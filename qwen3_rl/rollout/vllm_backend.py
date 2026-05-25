"""vLLM backend for fast rollout generation with sleep/wake lifecycle.

Wraps vLLM's offline LLM engine with:
- V1 engine in single-process mode (no EngineCore subprocess)
- Sleep/wake lifecycle with unsloth's CuMemAllocator patch (skips weights)
- LoRA adapter sync with stale-cache workaround (vLLM #42125)
- Prefix caching with mamba_cache_mode="align" for DeltaNet hybrid

Requires vLLM >= 0.20.0 and unsloth.  Import is lazy so the rest of
qwen3_rl works without vLLM installed.
"""

from __future__ import annotations

import asyncio
import itertools
import os
import sys
import threading
import warnings
from collections import OrderedDict
from contextlib import suppress
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import RolloutConfig


def _patch_vllm_decompose_size_nodes() -> None:
    """Skip x.size(dim) nodes in vLLM's torch.compile pass.

    vLLM's _decompose_size_nodes matches both x.size() (tuple) and
    x.size(dim) (scalar).  The scalar form needs no decomposition and
    crashes with "Tried to erase Node size but it still had N users".
    """
    try:
        import vllm.compilation.backends as _backends
    except ImportError:
        return

    import operator
    import torch
    import torch.fx as fx

    def _fixed_decompose_size_nodes(graph: fx.GraphModule) -> None:
        size_nodes = list(
            graph.graph.find_nodes(op="call_method", target="size")
        )

        for node in size_nodes:
            # x.size(dim) returns a scalar SymInt -- nothing to decompose.
            if len(node.args) > 1:
                continue

            tensor_node = node.args[0]
            ev = tensor_node.meta.get("example_value")
            assert ev is not None, (
                f"Tensor node '{tensor_node.name}' has no example_value "
                f"metadata. Cannot decompose size node '{node.name}'."
            )

            dims: list[fx.Node | int] = []
            with graph.graph.inserting_after(tensor_node):
                for i in range(ev.dim()):
                    dim_val = ev.shape[i]
                    if isinstance(dim_val, torch.SymInt):
                        dn = graph.graph.call_function(
                            torch.ops.aten.sym_size.int,
                            args=(tensor_node, i),
                        )
                        dn.meta["example_value"] = dim_val
                        dims.append(dn)
                    elif isinstance(dim_val, int):
                        dims.append(dim_val)
                    else:
                        raise AssertionError(
                            f"dim_val is either torch.SymInt or int, "
                            f"got {type(dim_val)} for dim {i} of "
                            f"'{node.name}'"
                        )

            for user in list(node.users):
                if (
                    user.op == "call_function"
                    and user.target is operator.getitem
                    and len(user.args) == 2
                    and user.args[0] is node
                ):
                    idx = user.args[1]
                    assert isinstance(idx, int), (
                        f"Expected literal int index for getitem on size(), "
                        f"got {type(idx).__name__}: {idx}"
                    )
                    user.replace_all_uses_with(dims[idx])
                    graph.graph.erase_node(user)
                else:
                    new_args = []
                    for arg in user.args:
                        if arg is node:
                            new_args.extend(dims)
                        else:
                            new_args.append(arg)
                    user.args = tuple(new_args)
            graph.graph.erase_node(node)

    _backends._decompose_size_nodes = _fixed_decompose_size_nodes


def _prepend_path(env_key: str, path: str) -> None:
    if not path or not os.path.isdir(path):
        return
    existing = os.environ.get(env_key, "")
    parts = [p for p in existing.split(os.pathsep) if p]
    if path not in parts:
        os.environ[env_key] = os.pathsep.join([path] + parts)


def _has_nvcc(cuda_home: str | None) -> bool:
    return bool(cuda_home) and os.path.isfile(os.path.join(cuda_home, "bin", "nvcc"))


def _find_pip_cuda_home() -> str | None:
    try:
        import importlib.util
        spec = importlib.util.find_spec("nvidia.cuda_nvcc")
        if spec is None or spec.submodule_search_locations is None:
            import site
            cu13_roots = [
                os.path.join(root, "nvidia", "cu13")
                for root in site.getsitepackages() + [site.getusersitepackages()]
            ]
            cu13_root = next(
                (
                    root for root in cu13_roots
                    if os.path.isfile(os.path.join(root, "bin", "nvcc"))
                ),
                None,
            )
            return cu13_root
        else:
            nvcc_pkg = list(spec.submodule_search_locations)[0]
            cu13_root = os.path.normpath(os.path.join(nvcc_pkg, "..", "..", "cu13"))
            nvcc_bin = os.path.join(cu13_root, "bin", "nvcc")
            if not os.path.isfile(nvcc_bin):
                return None
            return cu13_root
    except Exception:
        return None


def _ensure_cuda_home() -> None:
    """Set CUDA_HOME/PATH to a CUDA 13 toolkit that can JIT SM120 kernels."""
    cuda_home = os.environ.get("CUDA_HOME")
    if not _has_nvcc(cuda_home):
        cuda_home = _find_pip_cuda_home()
        if cuda_home is None:
            return
        os.environ["CUDA_HOME"] = cuda_home

    _prepend_path("PATH", os.path.join(sys.prefix, "bin"))
    _prepend_path("PATH", os.path.join(cuda_home, "bin"))
    _prepend_path("LD_LIBRARY_PATH", os.path.join(cuda_home, "lib"))


def _ensure_cudart_path() -> None:
    """Point vLLM at the real libcudart, not tilelang's stub.

    Unsloth's import chain loads tilelang which puts a minimal
    ``libcudart_stub.so`` into the process address space.  vLLM's
    ``CudaRTLibrary`` uses ``find_loaded_library("libcudart")`` which
    then returns the stub, causing ``cudaDeviceReset`` AttributeError.

    We patch the function reference in both ``system_utils`` (the
    source) and ``cuda_wrapper`` (which imports it at module level)
    to skip any path containing ``_stub.so``.
    """
    real_cudart = None
    try:
        import nvidia.cuda_runtime
        lib_dir = os.path.dirname(nvidia.cuda_runtime.__file__)
        for name in ("libcudart.so.12", "libcudart.so.13", "libcudart.so"):
            path = os.path.join(lib_dir, "lib", name)
            if os.path.isfile(path):
                real_cudart = path
                os.environ["VLLM_CUDART_SO_PATH"] = path
                break
    except ImportError:
        pass

    if real_cudart is None:
        try:
            import site
            candidates = []
            for root in site.getsitepackages() + [site.getusersitepackages()]:
                candidates.extend([
                    os.path.join(root, "nvidia", "cu13", "lib", "libcudart.so.13"),
                    os.path.join(root, "nvidia", "cu13", "lib", "libcudart.so"),
                    os.path.join(root, "nvidia", "cu12", "lib", "libcudart.so.12"),
                    os.path.join(root, "nvidia", "cu12", "lib", "libcudart.so"),
                ])
            for path in candidates:
                if os.path.isfile(path):
                    real_cudart = path
                    os.environ["VLLM_CUDART_SO_PATH"] = path
                    break
        except Exception:
            pass

    if real_cudart is None:
        return

    # Force-load the real libcudart with RTLD_GLOBAL so it appears
    # BEFORE tilelang's stub in /proc/self/maps.  vLLM's
    # find_loaded_library scans maps top-to-bottom and returns the
    # first match; ctypes.CDLL with RTLD_GLOBAL puts our library at
    # a higher address (later in maps), but we can also just patch
    # the function to skip stubs.
    import ctypes
    ctypes.CDLL(real_cudart, mode=ctypes.RTLD_GLOBAL)

    try:
        import vllm.utils.system_utils as _su
        _orig_find = _su.find_loaded_library

        def _find_no_stubs(lib_name: str):
            result = _orig_find(lib_name)
            if result and "_stub.so" in result:
                return real_cudart
            return result

        _su.find_loaded_library = _find_no_stubs

        # cuda_wrapper imports find_loaded_library at module level,
        # so we must also patch it there once it's loaded.
        import vllm.distributed.device_communicators.cuda_wrapper as _cw
        _cw.find_loaded_library = _find_no_stubs
    except (ImportError, AttributeError):
        pass


def _pre_import_env(config: RolloutConfig) -> None:
    """Set all env vars that must exist BEFORE ``import vllm``."""
    _ensure_cuda_home()
    _ensure_cudart_path()

    # Disable V1 EngineCore subprocess — it corrupts Triton CUDA state
    # in the parent process, causing CUDA_ERROR_MISALIGNED_ADDRESS during
    # FLA DeltaNet backward.  Unsloth uses the same fix.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")

    if config.language_model_only:
        os.environ["VLLM_LANGUAGE_MODEL_ONLY"] = "1"

    if config.enable_sleep_mode and not config.async_refill:
        os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
    if config.async_refill:
        os.environ["QWEN3_RL_PATCH_VLLM_CHILD"] = "1"


def _classify_finish(vllm_reason, gen_ids, stop_token_ids):
    if vllm_reason == "length":
        return "length"
    if vllm_reason == "stop":
        if gen_ids and gen_ids[-1] in stop_token_ids:
            return "eos"
        return "stop"
    return "eos"


def _lookup_logprob(lp_dict, token_id: int) -> float:
    item = lp_dict.get(token_id)
    if item is None:
        item = lp_dict.get(str(token_id))
    if item is None:
        # vLLM normally includes the sampled token even with logprobs=1.
        # Keep a fallback for older versions / stop-token edge cases.
        item = next(iter(lp_dict.values()))
    return float(item.logprob)


def _extract_generated_logprobs(output, gen_ids: list[int]) -> list[float] | None:
    if not output.logprobs:
        return None
    logprobs = []
    for token_id, lp_dict in zip(gen_ids, output.logprobs):
        if lp_dict:
            logprobs.append(_lookup_logprob(lp_dict, token_id))
        else:
            logprobs.append(0.0)
    return logprobs


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("VLLMBackend async methods cannot be called from a running event loop")


class VLLMBackend:

    def __init__(
        self,
        model_name: str,
        config: RolloutConfig,
        lora_path: str | None = None,
    ):
        _pre_import_env(config)

        # Unsloth's patch_vllm() must run before LLM is instantiated.
        # It patches CuMemAllocator.sleep/wake_up to skip weight tensors
        # (weights are shared with the HF model, not owned by vLLM).
        # We selectively apply only the patches we need: sleep mode +
        # inductor config.  We skip LoRA manager patching because unsloth's
        # replacement class is missing methods that vLLM 0.21+ expects
        # (e.g. get_dummy_lora_warmup_rank), and we handle LoRA sync
        # ourselves via load_lora_adapter().
        from unsloth_zoo.vllm_utils import (
            patch_vllm_set_inductor_config,
            patch_vllm_enable_sleep_mode,
            patch_vllm_graph_capture,
        )
        patch_vllm_set_inductor_config()
        if config.enable_sleep_mode and not config.async_refill:
            patch_vllm_enable_sleep_mode()
        patch_vllm_graph_capture()

        # Fix torch.compile crash in _decompose_size_nodes for hybrid
        # DeltaNet/GQA models (Qwen3.5). Must run before LLM instantiation.
        if not config.enforce_eager:
            _patch_vllm_decompose_size_nodes()

        from vllm import LLM, SamplingParams
        from vllm import TokensPrompt
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        try:
            from vllm import LoRARequest
        except ImportError:
            from vllm.lora.request import LoRARequest

        self._SamplingParams = SamplingParams
        self._TokensPrompt = TokensPrompt
        self._LoRARequest = LoRARequest
        self.config = config
        self._async_engine = config.async_refill
        self._engine_sleep_mode = config.enable_sleep_mode and not self._async_engine
        self._request_seq = itertools.count()
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._async_thread: threading.Thread | None = None

        # LoRA adapter tracking for stale-cache workaround
        self._current_lora: LoRARequest | None = None
        self._prev_adapter_name: str | None = None
        self._loaded_loras: OrderedDict[str, LoRARequest] = OrderedDict()

        self._stop_warned: bool = False

        # Build engine kwargs
        engine_kwargs: dict = dict(
            model=model_name,
            dtype="bfloat16",
            max_model_len=config.max_total_tokens,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enable_prefix_caching=config.enable_prefix_caching,
            enable_lora=config.enable_lora,
            max_lora_rank=config.max_lora_rank,
            enforce_eager=config.enforce_eager,
            enable_sleep_mode=config.enable_sleep_mode,
            hf_overrides={"mamba_cache_mode": config.mamba_cache_mode},
        )

        # Quantization
        if config.quantization == "fp8":
            engine_kwargs["quantization"] = "fp8"
        elif config.quantization == "nf4":
            engine_kwargs["quantization"] = "bitsandbytes"
            engine_kwargs["load_format"] = "bitsandbytes"

        if self._async_engine:
            async_kwargs = dict(engine_kwargs)
            async_kwargs["language_model_only"] = config.language_model_only
            async_kwargs["disable_log_stats"] = True
            # vLLM's async engine runs EngineCore in a child process.  The
            # sleep-mode CuMemAllocator patch is process-local, so keep async
            # rollouts on the standard allocator.
            async_kwargs["enable_sleep_mode"] = False
            engine_args = AsyncEngineArgs(**async_kwargs)
            self.llm = AsyncLLMEngine.from_engine_args(engine_args)
            self._async_loop = asyncio.new_event_loop()
            self._async_thread = threading.Thread(
                target=self._run_async_loop,
                name="qwen3-rl-vllm-async",
                daemon=True,
            )
            self._async_thread.start()
        else:
            self.llm = LLM(**engine_kwargs)

        # If an initial LoRA path is provided, load it as adapter_v0
        if lora_path is not None:
            self.sync_adapter(lora_path, iter_id=0)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _run_async_loop(self) -> None:
        assert self._async_loop is not None
        asyncio.set_event_loop(self._async_loop)
        self._async_loop.run_forever()

    def _run_on_async_loop(self, coro):
        if self._async_loop is None:
            return _run_async(coro)
        future = asyncio.run_coroutine_threadsafe(coro, self._async_loop)
        return future.result()

    def _sampling_params(
        self,
        *,
        max_new: int,
        stop_token_ids: list[int],
        stop: list[str],
        seed: int,
        return_logprobs: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ):
        kwargs = dict(
            max_tokens=max_new,
            temperature=self.config.temperature if temperature is None else temperature,
            top_p=self.config.top_p if top_p is None else top_p,
            stop_token_ids=stop_token_ids,
            stop=stop if stop else None,
            seed=seed,
        )
        if top_k is None:
            top_k = self.config.top_k
        if top_k is not None:
            kwargs["top_k"] = top_k
        if return_logprobs:
            kwargs["logprobs"] = 1
        return self._SamplingParams(**kwargs)

    def generate(
        self,
        token_ids: list[int],
        max_new: int,
        stop_token_ids: list[int],
        stop: list[str],
        seed: int,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        lora_request=None,
    ) -> tuple[list[int], str]:
        """Generate tokens from a token-id prefix.

        Returns (new_token_ids, finish_reason) where finish_reason is one
        of "eos", "length", "stop".
        """
        if self._async_engine:
            gen_ids, finish, _ = _run_async(self.generate_async(
                token_ids=token_ids,
                max_new=max_new,
                stop_token_ids=stop_token_ids,
                stop=stop,
                seed=seed,
                return_logprobs=False,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                lora_request=lora_request,
            ))
            return gen_ids, finish

        sampling = self._sampling_params(
            max_new=max_new,
            stop_token_ids=stop_token_ids,
            stop=stop,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        prompt = self._TokensPrompt(prompt_token_ids=token_ids)

        outputs = self.llm.generate(
            [prompt],
            sampling_params=sampling,
            lora_request=lora_request or self._current_lora,
        )
        output = outputs[0].outputs[0]

        # Token output is authoritative (design invariant: never decode
        # then re-encode).
        gen_ids = list(output.token_ids)
        finish = _classify_finish(output.finish_reason, gen_ids, stop_token_ids)

        if finish == "stop" and not self._stop_warned:
            self._stop_warned = True
            warnings.warn(
                "vLLM stop-string normalization fired — output "
                "tokens may not include the stop-string suffix",
                RuntimeWarning, stacklevel=2,
            )
        return gen_ids, finish

    def generate_batch(
        self,
        token_ids_list: list[list[int]],
        max_new: int,
        stop_token_ids: list[int],
        stop: list[str],
        seeds: list[int],
        return_logprobs: bool = False,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        lora_request=None,
    ) -> list[tuple[list[int], str, list[float] | None]]:
        """Generate from multiple prompts in a single vLLM batch.

        Returns list of (token_ids, finish_reason, logprobs_or_None).
        When ``return_logprobs=True``, each generated token's logprob is
        returned as a float list aligned with token_ids.
        """
        if self._async_engine:
            max_inflight = len(token_ids_list) or 1
            return self.generate_batch_refill(
                token_ids_list=token_ids_list,
                max_new=max_new,
                stop_token_ids=stop_token_ids,
                stop=stop,
                seeds=seeds,
                return_logprobs=return_logprobs,
                max_inflight=max_inflight,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                lora_request=lora_request,
            )

        prompts = [
            self._TokensPrompt(prompt_token_ids=ids) for ids in token_ids_list
        ]
        sampling_list = [
            self._sampling_params(
                max_new=max_new,
                stop_token_ids=stop_token_ids,
                stop=stop,
                seed=s,
                return_logprobs=return_logprobs,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            for s in seeds
        ]

        outputs = self.llm.generate(
            prompts,
            sampling_params=sampling_list,
            lora_request=lora_request or self._current_lora,
        )

        results = []
        for req_out in outputs:
            output = req_out.outputs[0]
            gen_ids = list(output.token_ids)
            finish = _classify_finish(output.finish_reason, gen_ids, stop_token_ids)

            logprobs = None
            if return_logprobs and output.logprobs:
                logprobs = _extract_generated_logprobs(output, gen_ids)

            results.append((gen_ids, finish, logprobs))
        return results

    async def generate_async(
        self,
        token_ids: list[int],
        max_new: int,
        stop_token_ids: list[int],
        stop: list[str],
        seed: int,
        *,
        request_id: str | None = None,
        return_logprobs: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        lora_request=None,
    ) -> tuple[list[int], str, list[float] | None]:
        """Generate one request on the async vLLM engine."""
        if not self._async_engine:
            batch = await asyncio.to_thread(
                self.generate_batch,
                [token_ids],
                max_new,
                stop_token_ids,
                stop,
                [seed],
                return_logprobs,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                lora_request=lora_request,
            )
            return batch[0]

        if self._async_loop is not None:
            running_loop = None
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                pass
            if running_loop is not self._async_loop:
                future = asyncio.run_coroutine_threadsafe(
                    self._generate_async_impl(
                        token_ids=token_ids,
                        max_new=max_new,
                        stop_token_ids=stop_token_ids,
                        stop=stop,
                        seed=seed,
                        request_id=request_id,
                        return_logprobs=return_logprobs,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        lora_request=lora_request,
                    ),
                    self._async_loop,
                )
                return await asyncio.wrap_future(future)

        return await self._generate_async_impl(
            token_ids=token_ids,
            max_new=max_new,
            stop_token_ids=stop_token_ids,
            stop=stop,
            seed=seed,
            request_id=request_id,
            return_logprobs=return_logprobs,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            lora_request=lora_request,
        )

    async def _generate_async_impl(
        self,
        *,
        token_ids: list[int],
        max_new: int,
        stop_token_ids: list[int],
        stop: list[str],
        seed: int,
        request_id: str | None,
        return_logprobs: bool,
        temperature: float | None,
        top_p: float | None,
        top_k: int | None,
        lora_request,
    ) -> tuple[list[int], str, list[float] | None]:
        sampling = self._sampling_params(
            max_new=max_new,
            stop_token_ids=stop_token_ids,
            stop=stop,
            seed=seed,
            return_logprobs=return_logprobs,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        prompt = self._TokensPrompt(prompt_token_ids=token_ids)
        req_id = request_id or f"qwen3-rl-{next(self._request_seq)}"

        final = None
        async for req_out in self.llm.generate(
            prompt,
            sampling_params=sampling,
            request_id=req_id,
            lora_request=lora_request or self._current_lora,
        ):
            final = req_out
        if final is None:
            return [], "eos", [] if return_logprobs else None

        output = final.outputs[0]
        gen_ids = list(output.token_ids)
        finish = _classify_finish(output.finish_reason, gen_ids, stop_token_ids)
        logprobs = _extract_generated_logprobs(output, gen_ids) if return_logprobs else None
        return gen_ids, finish, logprobs

    def generate_batch_refill(
        self,
        token_ids_list: list[list[int]],
        max_new: int,
        stop_token_ids: list[int],
        stop: list[str],
        seeds: list[int],
        return_logprobs: bool = False,
        max_inflight: int | None = None,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        lora_request=None,
    ) -> list[tuple[list[int], str, list[float] | None]]:
        """Generate requests with bounded async in-flight refill."""
        if not self._async_engine:
            return self.generate_batch(
                token_ids_list,
                max_new,
                stop_token_ids,
                stop,
                seeds,
                return_logprobs,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                lora_request=lora_request,
            )
        return self._run_on_async_loop(self._generate_batch_refill_async(
            token_ids_list=token_ids_list,
            max_new=max_new,
            stop_token_ids=stop_token_ids,
            stop=stop,
            seeds=seeds,
            return_logprobs=return_logprobs,
            max_inflight=max_inflight,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            lora_request=lora_request,
        ))

    async def _generate_batch_refill_async(
        self,
        *,
        token_ids_list: list[list[int]],
        max_new: int,
        stop_token_ids: list[int],
        stop: list[str],
        seeds: list[int],
        return_logprobs: bool,
        max_inflight: int | None,
        temperature: float | None,
        top_p: float | None,
        top_k: int | None,
        lora_request,
    ) -> list[tuple[list[int], str, list[float] | None]]:
        from collections import deque

        if len(token_ids_list) != len(seeds):
            raise ValueError("token_ids_list and seeds must have the same length")
        limit = max_inflight or len(token_ids_list) or 1
        limit = max(1, min(limit, len(token_ids_list) or 1))
        pending = deque(range(len(token_ids_list)))
        tasks: dict[asyncio.Task, tuple[int, str]] = {}
        results: list[tuple[list[int], str, list[float] | None] | None] = [
            None for _ in token_ids_list
        ]

        def submit(idx: int) -> None:
            req_id = f"qwen3-rl-{next(self._request_seq)}"
            task = asyncio.create_task(self.generate_async(
                token_ids=token_ids_list[idx],
                max_new=max_new,
                stop_token_ids=stop_token_ids,
                stop=stop,
                seed=seeds[idx],
                request_id=req_id,
                return_logprobs=return_logprobs,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                lora_request=lora_request,
            ))
            tasks[task] = (idx, req_id)

        try:
            while pending or tasks:
                while pending and len(tasks) < limit:
                    submit(pending.popleft())
                done, _ = await asyncio.wait(
                    tasks.keys(), return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    idx, _ = tasks.pop(task)
                    results[idx] = task.result()
        except BaseException:
            for task, (_, req_id) in list(tasks.items()):
                await self.llm.abort(req_id)
                task.cancel()
            raise

        return [r for r in results if r is not None]

    def abort_requests(self, request_ids: list[str]) -> None:
        if self._async_engine and request_ids:
            self._run_on_async_loop(self.llm.abort(request_ids))

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        if not request_ids:
            return
        if self._async_engine:
            await self.llm.abort(request_ids)

    def close(self) -> None:
        if not self._async_engine or self._async_loop is None:
            return
        self._run_on_async_loop(self._close_async())
        self._async_loop.call_soon_threadsafe(self._async_loop.stop)
        if self._async_thread is not None:
            self._async_thread.join(timeout=5)

    async def _close_async(self) -> None:
        handler = getattr(self.llm, "output_handler", None)
        if handler is not None and not handler.done():
            handler.cancel()
            with suppress(asyncio.CancelledError):
                await handler
        self.llm.shutdown(timeout=0)

    # ------------------------------------------------------------------
    # Sleep / wake lifecycle
    # ------------------------------------------------------------------

    def sleep(self) -> None:
        """Offload KV cache to CPU and free GPU memory for training.

        With unsloth's patched CuMemAllocator, weight tensors are skipped
        (they're shared with the HF model).  Only KV cache is offloaded.
        """
        if self._async_engine:
            if not self._engine_sleep_mode:
                return
            self.llm.sleep(level=self.config.sleep_level, mode="abort")
        else:
            self.llm.sleep(level=self.config.sleep_level)

    def wake(self) -> None:
        """Restore KV cache from CPU to GPU before rollout.

        Must NOT pass ``tags=["weights"]`` — that leaves the executor's
        ``is_sleeping`` flag True, causing a redundant wake inside
        ``generate()`` which double-maps already-mapped handles.
        """
        if not self._engine_sleep_mode and self._async_engine:
            return
        self.llm.wake_up()

    # ------------------------------------------------------------------
    # LoRA adapter sync
    # ------------------------------------------------------------------

    def current_lora_request(self):
        return self._current_lora

    def _max_loaded_loras(self) -> int:
        configured = getattr(self.config, "max_loaded_loras", None)
        if configured is not None:
            return max(1, int(configured))
        max_off_policy = getattr(self.config, "async_max_off_policy_steps", 0)
        return max(2, int(max_off_policy) + 2)

    def _remove_lora(self, name: str) -> None:
        if self._async_engine:
            remove = getattr(self.llm, "remove_lora", None)
            if remove is not None:
                self._run_on_async_loop(remove(name))
            return

        engine = getattr(self.llm, "llm_engine", None)
        remove = getattr(engine, "remove_lora", None)
        if remove is not None:
            remove(name)

    def _evict_old_loras(self) -> None:
        while len(self._loaded_loras) > self._max_loaded_loras():
            name, lora_req = next(iter(self._loaded_loras.items()))
            if lora_req is self._current_lora and len(self._loaded_loras) == 1:
                return
            self._loaded_loras.pop(name, None)
            self._remove_lora(name)

    def sync_adapter(self, lora_path: str, iter_id: int) -> None:
        """Load a new LoRA adapter checkpoint via the engine's add_lora API.

        Uses a bounded ring of names (``adapter_v{iter_id}``) so active
        async rollouts can remain pinned to their behavior-policy adapter while
        older, unreferenced versions are evicted.
        """
        new_name = f"adapter_v{iter_id}"
        lora_req = self._LoRARequest(new_name, int(iter_id) + 1, lora_path)
        if self._async_engine:
            self._run_on_async_loop(self.llm.add_lora(lora_req))
        else:
            self.llm.llm_engine.add_lora(lora_req)
        self._current_lora = lora_req
        self._prev_adapter_name = new_name
        self._loaded_loras[new_name] = lora_req
        self._loaded_loras.move_to_end(new_name)
        self._evict_old_loras()
