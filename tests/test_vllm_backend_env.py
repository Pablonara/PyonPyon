import os
from types import SimpleNamespace

from qwen3_rl.rollout import vllm_backend


def test_ensure_cuda_home_replaces_invalid_existing_cuda_home(tmp_path, monkeypatch):
    invalid = tmp_path / "missing_cuda"
    cuda_home = tmp_path / "cu13"
    (cuda_home / "bin").mkdir(parents=True)
    (cuda_home / "lib").mkdir()
    (cuda_home / "bin" / "nvcc").write_text("")

    monkeypatch.setenv("CUDA_HOME", str(invalid))
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.setattr(vllm_backend, "_find_pip_cuda_home", lambda: str(cuda_home))

    vllm_backend._ensure_cuda_home()

    assert os.environ["CUDA_HOME"] == str(cuda_home)
    assert os.environ["PATH"].split(os.pathsep)[0] == str(cuda_home / "bin")
    assert os.environ["LD_LIBRARY_PATH"].split(os.pathsep)[0] == str(cuda_home / "lib")


def test_ensure_cuda_home_keeps_valid_existing_cuda_home(tmp_path, monkeypatch):
    cuda_home = tmp_path / "valid_cuda"
    (cuda_home / "bin").mkdir(parents=True)
    (cuda_home / "lib").mkdir()
    (cuda_home / "bin" / "nvcc").write_text("")

    monkeypatch.setenv("CUDA_HOME", str(cuda_home))
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.setattr(vllm_backend, "_find_pip_cuda_home", lambda: None)

    vllm_backend._ensure_cuda_home()

    assert os.environ["CUDA_HOME"] == str(cuda_home)
    assert os.environ["PATH"].split(os.pathsep)[0] == str(cuda_home / "bin")


def test_extract_generated_logprobs_uses_sampled_token_id():
    output = SimpleNamespace(logprobs=[
        {
            111: SimpleNamespace(logprob=-9.0),
            222: SimpleNamespace(logprob=-0.5),
        },
        {
            333: SimpleNamespace(logprob=-0.25),
            444: SimpleNamespace(logprob=-8.0),
        },
    ])

    assert vllm_backend._extract_generated_logprobs(output, [222, 333]) == [
        -0.5,
        -0.25,
    ]


def test_sync_adapter_evicts_old_loras_in_bounded_ring():
    class FakeLoRARequest:
        def __init__(self, lora_name, lora_int_id, lora_path):
            self.lora_name = lora_name
            self.lora_int_id = lora_int_id
            self.lora_path = lora_path

    class FakeEngine:
        def __init__(self):
            self.added = []
            self.removed = []

        def add_lora(self, req):
            self.added.append(req.lora_name)

        def remove_lora(self, name):
            self.removed.append(name)

    backend = vllm_backend.VLLMBackend.__new__(vllm_backend.VLLMBackend)
    backend.config = SimpleNamespace(max_loaded_loras=2)
    backend._LoRARequest = FakeLoRARequest
    backend._async_engine = False
    backend._current_lora = None
    backend._prev_adapter_name = None
    backend._loaded_loras = vllm_backend.OrderedDict()
    engine = FakeEngine()
    backend.llm = SimpleNamespace(llm_engine=engine)

    backend.sync_adapter("/tmp/a0", 0)
    first = backend.current_lora_request()
    backend.sync_adapter("/tmp/a1", 1)
    backend.sync_adapter("/tmp/a2", 2)

    assert first.lora_name == "adapter_v0"
    assert backend.current_lora_request().lora_name == "adapter_v2"
    assert list(backend._loaded_loras) == ["adapter_v1", "adapter_v2"]
    assert engine.added == ["adapter_v0", "adapter_v1", "adapter_v2"]
    assert engine.removed == ["adapter_v0"]
