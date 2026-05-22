import os

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
