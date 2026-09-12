"""Hardware detection for CPU, NVIDIA CUDA GPU, and NPU/DirectML/ONNX."""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class HardwareReport:
    platform_name: str
    python_version: str
    cpu_model: str
    cpu_cores: int
    ram_gb: float
    cuda_available: bool = False
    cuda_device_name: Optional[str] = None
    cuda_vram_gb: Optional[float] = None
    cupy_installed: bool = False
    npu_available: bool = False
    npu_providers: List[str] = field(default_factory=list)
    onnxruntime_available: bool = False
    recommended_backend: str = "cpu"


def detect_hardware() -> HardwareReport:
    plat = f"{platform.system()} {platform.release()} ({platform.machine()})"
    py_ver = sys.version.split()[0]
    cores = os.cpu_count() or 1
    cpu_model = platform.processor() or "x86_64 / ARM"

    # Memory
    ram_gb = 4.0
    try:
        if sys.platform == "linux":
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        ram_gb = int(line.split()[1]) / (1024 * 1024)
                        break
        elif sys.platform == "win32":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulonglong = ctypes.c_ulonglong
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', c_ulonglong),
                    ('ullAvailPhys', c_ulonglong),
                    ('ullTotalPageFile', c_ulonglong),
                    ('ullAvailPageFile', c_ulonglong),
                    ('ullTotalVirtual', c_ulonglong),
                    ('ullAvailVirtual', c_ulonglong),
                    ('ullAvailExtendedVirtual', c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                ram_gb = stat.ullTotalPhys / (1024 ** 3)
    except Exception:
        pass

    # CUDA / CuPy
    cuda_available = False
    cuda_device_name = None
    cuda_vram_gb = None
    cupy_installed = False

    try:
        import cupy as cp
        cupy_installed = True
        if cp.cuda.runtime.getDeviceCount() > 0:
            cuda_available = True
            dev = cp.cuda.Device(0)
            props = cp.cuda.runtime.getDeviceProperties(0)
            cuda_device_name = props["name"].decode("utf-8") if isinstance(props["name"], bytes) else str(props["name"])
            mem_info = dev.mem_info
            cuda_vram_gb = round(mem_info[1] / (1024 ** 3), 2)
    except Exception:
        pass

    # NPU / ONNX Runtime
    onnx_available = False
    npu_available = False
    npu_providers: List[str] = []

    try:
        import onnxruntime as ort
        onnx_available = True
        providers = ort.get_available_providers()
        npu_target_providers = [
            "DmlExecutionProvider",        # Windows DirectML (NPUs, GPUs)
            "OpenVINOExecutionProvider",   # Intel Core Ultra NPU / iGPU
            "QNNExecutionProvider",        # Qualcomm Snapdragon NPU
            "VitisAIExecutionProvider",    # AMD Ryzen AI NPU
        ]
        for p in providers:
            if p in npu_target_providers:
                npu_available = True
                npu_providers.append(p)
    except Exception:
        pass

    # Determine recommended backend
    if cuda_available:
        recommended = "cuda"
    elif npu_available:
        recommended = "npu (via onnx export)"
    else:
        recommended = "cpu"

    return HardwareReport(
        platform_name=plat,
        python_version=py_ver,
        cpu_model=cpu_model,
        cpu_cores=cores,
        ram_gb=round(ram_gb, 2),
        cuda_available=cuda_available,
        cuda_device_name=cuda_device_name,
        cuda_vram_gb=cuda_vram_gb,
        cupy_installed=cupy_installed,
        npu_available=npu_available,
        npu_providers=npu_providers,
        onnxruntime_available=onnx_available,
        recommended_backend=recommended,
    )


def print_hardware_summary(report: HardwareReport | None = None) -> None:
    if report is None:
        report = detect_hardware()

    print("=" * 60)
    print("           HARDWARE DETECTION & ACCELERATION REPORT")
    print("=" * 60)
    print(f"Platform:       {report.platform_name}")
    print(f"Python:         {report.python_version}")
    print(f"CPU:            {report.cpu_model} ({report.cpu_cores} threads/cores)")
    print(f"System RAM:     {report.ram_gb:.1f} GB")
    print("-" * 60)
    if report.cuda_available:
        print(f"CUDA GPU:       {report.cuda_device_name} ({report.cuda_vram_gb} GB VRAM)")
        print(f"CuPy Status:    Installed & Active")
    else:
        print("CUDA GPU:       Not detected or CuPy not installed")
        if not report.cupy_installed:
            print("  (Tip: for NVIDIA GPU acceleration, run: pip install cupy-cuda12x or cupy-cuda13x)")
    print("-" * 60)
    if report.npu_available:
        print(f"NPU/DirectML:   Available ({', '.join(report.npu_providers)})")
        print("  Inference:    Supported via ONNX export ('python numpy_gpt.py export')")
    else:
        print("NPU/DirectML:   No dedicated NPU execution provider detected in current environment")
        if report.onnxruntime_available:
            print("  (ONNX Runtime installed: CPU/DirectML export is ready)")
        else:
            print("  (Tip: install onnxruntime-directml on Windows to enable DirectML NPU/GPU inference)")
    print("-" * 60)
    print(f"Primary Training Backend:  {'CUDA (CuPy)' if report.cuda_available else 'NumPy (CPU)'}")
    print(f"Recommended Action:        {report.recommended_backend.upper()}")
    print("=" * 60)
