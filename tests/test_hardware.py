import pytest
from hardware.detection import detect_hardware, print_hardware_summary
from hardware.backend import select_backend, get_backend, to_cpu, to_device


def test_hardware_detection():
    report = detect_hardware()
    assert report.cpu_cores >= 1
    assert report.ram_gb > 0.0
    assert report.recommended_backend in ("cpu", "cuda", "npu (via onnx export)")


def test_backend_cpu_fallback():
    backend = select_backend("cpu")
    assert backend == "cpu"
    xp = get_backend()
    arr = xp.array([1, 2, 3])
    cpu_arr = to_cpu(arr)
    assert cpu_arr.tolist() == [1, 2, 3]
