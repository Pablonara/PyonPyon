"""vLLM backend for batched rollout generation with torch.compile + CUDA graphs."""

from __future__ import annotations

import os
import warnings
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


def _ensure_cuda_home() -> None:
    """Set CUDA_HOME to pip-installed nvcc 13+ if system nvcc is too old for SM120."""
    if "CUDA_HOME" in os.environ:
        return
    try:
        import importlib.util
        spec = importlib.util.find_spec("nvidia.cuda_nvcc")
        if spec is None or spec.submodule_search_locations is None:
            return
        nvcc_pkg = list(spec.submodule_search_locations)[0]
        cu13_root = os.path.normpath(os.path.join(nvcc_pkg, "..", "..", "cu13"))
        if not os.path.isfile(os.path.join(cu13_root, "bin", "nvcc")):
            return
        os.environ["CUDA_HOME"] = cu13_root
        lib_dir = os.path.join(cu13_root, "lib")
        if os.path.isdir(lib_dir):
            existing = os.environ.get("LD_LIBRARY_PATH", "")
            if lib_dir not in existing:
                os.environ["LD_LIBRARY_PATH"] = (
                    f"{lib_dir}:{existing}" if existing else lib_dir
                )
    except Exception:
        pass


def _ensure_cudart_path() -> None:
    """Patch find_loaded_library to skip tilelang's libcudart_stub.so."""
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
        return

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
        import vllm.distributed.device_communicators.cuda_wrapper as _cw
        _cw.find_loaded_library = _find_no_stubs
    except (ImportError, AttributeError):
        pass


def _pre_import_env(config: RolloutConfig) -> None:
    """Set env vars that must exist before ``import vllm``."""
    _ensure_cuda_home()
    _ensure_cudart_path()
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
    if config.language_model_only:
        os.environ["VLLM_LANGUAGE_MODEL_ONLY"] = "1"
    if config.enable_sleep_mode:
        os.environ["UNSLOTH_VLLM_STANDBY"] = "1"


def _classify_finish(vllm_reason, gen_ids, stop_token_ids):
    if vllm_reason == "length":
        return "length"
    if vllm_reason == "stop":
        if gen_ids and gen_ids[-1] in stop_token_ids:
            return "eos"
        return "stop"
    return "eos"


class VLLMBackend:

    def __init__(
        self,
        model_name: str,
        config: RolloutConfig,
        lora_path: str | None = None,
    ):
        _pre_import_env(config)

        # Selective unsloth patches — skip LoRA manager (missing methods in 0.21+)
        from unsloth_zoo.vllm_utils import (
            patch_vllm_set_inductor_config,
            patch_vllm_enable_sleep_mode,
            patch_vllm_graph_capture,
        )
        patch_vllm_set_inductor_config()
        if config.enable_sleep_mode:
            patch_vllm_enable_sleep_mode()
        patch_vllm_graph_capture()

        if not config.enforce_eager:
            _patch_vllm_decompose_size_nodes()

        from vllm import LLM, SamplingParams, TokensPrompt
        try:
            from vllm import LoRARequest
        except ImportError:
            from vllm.lora.request import LoRARequest

        self._SamplingParams = SamplingParams
        self._TokensPrompt = TokensPrompt
        self._LoRARequest = LoRARequest
        self.config = config
        self._current_lora: LoRARequest | None = None
        self._prev_adapter_name: str | None = None
        self._stop_warned: bool = False

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
        if config.quantization == "fp8":
            engine_kwargs["quantization"] = "fp8"
        elif config.quantization == "nf4":
            engine_kwargs["quantization"] = "bitsandbytes"
            engine_kwargs["load_format"] = "bitsandbytes"

        self.llm = LLM(**engine_kwargs)

        if lora_path is not None:
            self.sync_adapter(lora_path, iter_id=0)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        token_ids: list[int],
        max_new: int,
        stop_token_ids: list[int],
        stop: list[str],
        seed: int,
    ) -> tuple[list[int], str]:
        sampling = self._SamplingParams(
            max_tokens=max_new,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            stop_token_ids=stop_token_ids,
            stop=stop if stop else None,
            seed=seed,
        )
        output = self.llm.generate(
            [self._TokensPrompt(prompt_token_ids=token_ids)],
            sampling_params=sampling,
            lora_request=self._current_lora,
        )[0].outputs[0]

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
    ) -> list[tuple[list[int], str, list[float] | None]]:
        """Generate from multiple prompts in one vLLM batch.

        Returns list of (token_ids, finish_reason, logprobs_or_None).
        """
        prompts = [self._TokensPrompt(prompt_token_ids=ids) for ids in token_ids_list]
        sampling_list = [
            self._SamplingParams(
                max_tokens=max_new,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                stop_token_ids=stop_token_ids,
                stop=stop if stop else None,
                seed=s,
                logprobs=1 if return_logprobs else None,
            )
            for s in seeds
        ]

        outputs = self.llm.generate(
            prompts, sampling_params=sampling_list, lora_request=self._current_lora,
        )

        results = []
        for req_out in outputs:
            output = req_out.outputs[0]
            gen_ids = list(output.token_ids)
            finish = _classify_finish(output.finish_reason, gen_ids, stop_token_ids)

            logprobs = None
            if return_logprobs and output.logprobs:
                logprobs = [
                    next(iter(lp.values())).logprob if lp else 0.0
                    for lp in output.logprobs
                ]
            results.append((gen_ids, finish, logprobs))
        return results

    # ------------------------------------------------------------------
    # Sleep / wake lifecycle
    # ------------------------------------------------------------------

    def sleep(self) -> None:
        """Offload KV cache to CPU and free GPU memory for training.

        With unsloth's patched CuMemAllocator, weight tensors are skipped
        (they're shared with the HF model). Only KV cache is offloaded.
        """
        self.llm.sleep(level=self.config.sleep_level)

    def wake(self) -> None:
        """Restore KV cache from CPU to GPU before rollout.

        Must NOT pass ``tags=["weights"]`` — that leaves the executor's
        ``is_sleeping`` flag True, causing a redundant wake inside
        ``generate()`` which double-maps already-mapped handles.
        """
        self.llm.wake_up()

    # ------------------------------------------------------------------
    # LoRA adapter sync
    # ------------------------------------------------------------------

    def sync_adapter(self, lora_path: str, iter_id: int) -> None:
        new_name = f"adapter_v{iter_id}"
        lora_req = self._LoRARequest(new_name, 1, lora_path)
        self.llm.llm_engine.add_lora(lora_req)
        self._current_lora = lora_req
        self._prev_adapter_name = new_name
