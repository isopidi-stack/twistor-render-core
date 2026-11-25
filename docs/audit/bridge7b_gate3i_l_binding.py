# Gate 3I-L: review-only evaluation-binding prototype.
# Never emits a backend certificate or changes repository files.
import sys
import os
import platform
import hashlib
import struct
from pathlib import Path
from dataclasses import dataclass
import threading

REPO = Path.home() / "twistor-render-core"
sys.path.insert(0, str(REPO))
import ctypes
import math
from fractions import Fraction as F

import torch
from torch.utils._python_dispatch import (
    _get_current_dispatch_mode_stack,
)

torch.set_num_threads(1)

class Rejected(Exception):
    pass

def require(condition, reason):
    if not condition:
        raise Rejected(reason)

capture = ctypes.WinDLL("ntdll").RtlCaptureContext
capture.argtypes = [ctypes.c_void_p]
capture.restype = None

def read_mxcsr():
    raw = ctypes.create_string_buffer(4096 + 15)
    addr = (ctypes.addressof(raw) + 15) & ~15

    capture(ctypes.c_void_p(addr))

    flags = ctypes.c_uint32.from_address(addr + 0x30).value
    mxcsr = ctypes.c_uint32.from_address(addr + 0x34).value
    saved = ctypes.c_uint32.from_address(addr + 0x118).value

    require(
        (flags & 0x00100008) == 0x00100008,
        "FP_CONTEXT_MISSING"
    )
    require(mxcsr == saved, "MXCSR_CONTEXT_MISMATCH")
    return mxcsr

EXPECTED_KEYS = (
    "DispatchKeySet(CPU, ADInplaceOrView, "
    "AutogradCPU, AutocastCPU)"
)

EXPECTED_INCLUDE = (
    "DispatchKeySet(BackendSelect, ADInplaceOrView)"
)

EXPECTED_EXCLUDE = (
    "DispatchKeySet(AutocastCPU, AutocastMTIA, "
    "AutocastMAIA, AutocastXPU, AutocastIPU, "
    "AutocastHPU, AutocastXLA, AutocastMPS, "
    "AutocastCUDA, AutocastPrivateUse1)"
)

def admission_guard(p, b):
    C = torch._C

    require(type(p) is torch.Tensor, "TENSOR_TYPE")
    require(type(b) is float, "B_TYPE")

    require(
        not C._is_torch_function_mode_enabled(),
        "TORCH_FUNCTION_MODE"
    )
    require(
        not _get_current_dispatch_mode_stack(),
        "PYTHON_DISPATCH_MODE"
    )
    require(
        not C._are_functorch_transforms_active(),
        "FUNCTORCH"
    )
    require(
        C._is_torch_function_enabled(),
        "TORCH_FUNCTION_STATE"
    )

    require(not torch.is_grad_enabled(), "GRAD_MODE")
    require(
        not torch.is_inference_mode_enabled(),
        "INFERENCE_MODE"
    )
    require(
        not torch.is_autocast_enabled("cpu"),
        "AUTOCAST"
    )
    require(torch.get_num_threads() == 1, "THREAD_COUNT")

    require(
        str(C._dispatch_tls_local_include_set())
        == EXPECTED_INCLUDE,
        "TLS_INCLUDE"
    )
    require(
        str(C._dispatch_tls_local_exclude_set())
        == EXPECTED_EXCLUDE,
        "TLS_EXCLUDE"
    )

    require(
        (read_mxcsr() & 0xFFC0) == 0x1F80,
        "MXCSR_CONTROL"
    )

    require(p.device.type == "cpu", "DEVICE")
    require(p.dtype == torch.float64, "DTYPE")
    require(p.layout == torch.strided, "LAYOUT")
    require(tuple(p.shape) == (3,), "SHAPE")
    require(p.is_contiguous(), "CONTIGUITY")
    require(p.stride() == (1,), "STRIDE")
    require(not p.requires_grad, "REQUIRES_GRAD")
    require(not p.is_conj(), "CONJUGATE_BIT")
    require(not p.is_neg(), "NEGATIVE_BIT")
    require(not p.is_inference(), "INFERENCE_TENSOR")

    require(
        str(C._dispatch_keys(p)) == EXPECTED_KEYS,
        "TENSOR_DISPATCH_KEYS"
    )

    values = (*p.tolist(), b)

    require(
        all(math.isfinite(v) for v in values),
        "NONFINITE_INPUT"
    )
    require(b != 0.0, "ZERO_B")

    x, y, z, br = map(F, values)
    m = x*x + y*y + z*z + br*br

    require(
        F(1, 2**1022) <= m <= F(2**1022),
        "EXACT_DOMAIN"
    )

    q = 12 * F(1, 2**53) * m
    upper = float(q)

    require(math.isfinite(upper), "BOUND_NONFINITE")

    if F(upper) < q:
        upper = math.nextafter(upper, math.inf)

    require(
        math.isfinite(upper) and F(upper) >= q,
        "OUTWARD_ROUNDING"
    )
    require(
        F(math.nextafter(upper, -math.inf)) < q,
        "BOUND_NOT_MINIMAL"
    )

    require(
        (read_mxcsr() & 0xFFC0) == 0x1F80,
        "MXCSR_CONTROL_AFTER"
    )

    return upper, values


from twistor import congruence as cg

PINS = {
    "torch_cpu.dll": "b8ec43a4ff24435c93a8a7710f9cf3fe1f246d923abbd7549064c8ba26afc642",
    "torch_python.dll": "9865b16e791fc6f91fe92b38a930004a9fedaa60b3d70645b872f5d640574f29",
    "c10.dll": "60cae8210a16da67910754008ff62266cf529363fdcd032cd50a6b949dc4ea21",
}

def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def verify_session():
    require(platform.system() == "Windows", "PLATFORM")
    require(platform.machine().upper() == "AMD64", "ARCH")
    require(sys.version_info[:3] == (3, 13, 1), "PYTHON")
    require(torch.__version__ == "2.15.0+cpu", "TORCH_VERSION")
    require(torch.version.git_version ==
            "d0f2af8b66b20da4baa50644c0cd3f887a2eae2e", "TORCH_REVISION")
    require(torch.backends.cpu.get_cpu_capability() == "AVX2", "CPU_CAPABILITY")
    require(all(os.environ.get(k) is None for k in
                ("ATEN_CPU_CAPABILITY", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "MKL_CBWR")),
            "ENV_OVERRIDE")
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = k.GetModuleHandleW
    handle.argtypes = [ctypes.c_wchar_p]
    handle.restype = ctypes.c_void_p
    filename = k.GetModuleFileNameW
    filename.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_wchar), ctypes.c_uint32]
    filename.restype = ctypes.c_uint32
    lib = Path.home() / "twistor-benchmark" / ".venv-bench" / "Lib" / "site-packages" / "torch" / "lib"
    for name, expected in PINS.items():
        h = handle(name)
        require(bool(h), "MODULE_NOT_LOADED:" + name)
        buf = ctypes.create_unicode_buffer(32768)
        n = filename(h, buf, len(buf))
        require(0 < n < len(buf), "MODULE_PATH:" + name)
        path = Path(buf.value).resolve()
        require(path == (lib / name).resolve(), "MODULE_PATH:" + name)
        actual = digest(path)
        require(actual == expected, "MODULE_HASH:" + name)
        print("PIN", name, actual)
    source = Path(cg.__file__).resolve()
    require(source == (REPO / "twistor" / "congruence.py").resolve(), "SOURCE_PATH")
    actual = digest(source)
    require(actual == "481e8f194ad225ca866e1b383e34a85433cc91957ef8a241db10e7c73cb68f9c",
            "SOURCE_HASH")
    print("PIN congruence.py", actual)

def words(values):
    return tuple(struct.pack(">d", v).hex() for v in values)

def check_observation(values, upper, actual):
    # Exact diagnostic; not a universal graph proof.
    require(type(actual) is complex, "RESULT_TYPE")
    require(math.isfinite(actual.real) and math.isfinite(actual.imag),
            "RESULT_NONFINITE")
    x, y, z, b = map(F, values)
    m = x*x + y*y + z*z + b*b
    require(b != 0 and F(1, 2**1022) <= m <= F(2**1022), "EXACT_DOMAIN")
    q = 12 * F(1, 2**53) * m
    require(math.isfinite(upper) and F(upper) >= q, "BOUND_UNDER")
    require(F(math.nextafter(upper, -math.inf)) < q, "BOUND_NOT_MINIMAL")
    er = F(actual.real) - (x*x+y*y+z*z-b*b)
    ei = F(actual.imag) - 2*z*b
    error_sq = er*er + ei*ei
    require(error_sq <= q*q, "OBSERVED_ERROR_EXCEEDS_12UM")
    return error_sq

@dataclass(frozen=True)
class Observation:
    input_words: tuple
    value: complex
    bound: float

def evaluate(p, b):
    tid = threading.get_native_id()
    # Caller must provide stable input during the initial snapshot read.
    upper, values = admission_guard(p, b)
    snapshot = torch.tensor(values[:3], dtype=torch.float64, device="cpu")
    bound2, actual_inputs = admission_guard(snapshot, b)
    require(words(actual_inputs) == words(values), "SNAPSHOT_BITS")
    require(bound2.hex() == upper.hex(), "SNAPSHOT_BOUND")
    require(threading.get_native_id() == tid, "THREAD_CHANGED")
    require((read_mxcsr() & 0xFFC0) == 0x1F80, "MXCSR_BEFORE_DELTA")
    result = cg.delta(snapshot, b)
    require((read_mxcsr() & 0xFFC0) == 0x1F80, "MXCSR_AFTER_DELTA")
    require(threading.get_native_id() == tid, "THREAD_CHANGED")
    bound3, after_inputs = admission_guard(snapshot, b)
    require(words(after_inputs) == words(values), "SNAPSHOT_CHANGED")
    require(bound3.hex() == upper.hex(), "BOUND_CHANGED")
    require(type(result) is torch.Tensor and result.shape == () and
            result.device.type == "cpu" and result.dtype == torch.complex128 and
            not result.is_conj() and not result.is_neg(), "RESULT_METADATA")
    actual = result.item()
    check_observation(values, upper, actual)
    require((read_mxcsr() & 0xFFC0) == 0x1F80, "MXCSR_FINAL")
    # Immutable Python result and input bit identifiers, not a mutable tensor.
    return Observation(words(values), actual, upper)

def main():
    print("GATE3I_L_BEGIN")
    verify_session()
    tiny, low, high = 2.0**-1074, 2.0**-511, 2.0**511
    cases = [
        ("ordinary", (0.1, 0.3, -1.25, 1.5), None),
        ("decimal_defect", (2.0**-60, 0., 0., 1.), None),
        ("lower_equal", (0., 0., 0., low), None),
        ("upper_equal", (0., 0., 0., high), None),
        ("subnormal_b", (1., 0., 1., tiny), None),
        ("subnormal_products", (2.0**-530, 2.0**-540, low, 2.0**-530), None),
        ("exact_cancellation", (0., 0., 1.5, 1.5), None),
        ("lower_below", (0., 0., 0., math.nextafter(low, 0.)), "EXACT_DOMAIN"),
        ("upper_above_hidden", (tiny, 0., 0., high), "EXACT_DOMAIN"),
        ("underflow_control", (2.0**-550, 0., 2.0**-550, 2.0**-550), "EXACT_DOMAIN"),
        ("overflow_control", (2.0**600, 0., 2.0**600, 2.0**600), "EXACT_DOMAIN"),
    ]
    failures = 0
    with torch.no_grad():
        for name, values, expected in cases:
            p = torch.tensor(values[:3], dtype=torch.float64, device="cpu")
            try:
                obs = evaluate(p, values[3])
                observed = None
                print("OBSERVATION", name, *obs.input_words, obs.value.real.hex(),
                      obs.value.imag.hex(), obs.bound.hex())
            except Rejected as exc:
                observed = str(exc)
            passed = observed == expected
            failures += int(not passed)
            print("CASE", name, "EXPECTED", expected, "OBSERVED", observed, "PASS", passed)

        # Deliberately bad inputs to the exact observation checker, not monkeypatches.
        v = (0.1, 0.3, -1.25, 1.5)
        obs = evaluate(torch.tensor(v[:3], dtype=torch.float64, device="cpu"), v[3])
        for name, bound, actual, expected in [
            ("underrounded_bound", math.nextafter(obs.bound, -math.inf), obs.value, "BOUND_UNDER"),
            ("nonminimal_bound", math.nextafter(obs.bound, math.inf), obs.value, "BOUND_NOT_MINIMAL"),
            ("wrong_result", obs.bound, obs.value + 1.0, "OBSERVED_ERROR_EXCEEDS_12UM"),
            ("nonfinite_result", obs.bound, complex(math.nan, 0), "RESULT_NONFINITE"),
        ]:
            try:
                check_observation(v, bound, actual)
                observed = None
            except Rejected as exc:
                observed = str(exc)
            passed = observed == expected
            failures += int(not passed)
            print("CHECKER_CONTROL", name, "EXPECTED", expected, "OBSERVED", observed, "PASS", passed)
    verify_session()
    print("TOTAL", 15)
    print("FAILURES", failures)
    print("GATE3I_L_STATUS", "PASS" if failures == 0 else "REVIEW_REQUIRED")
    print("SCOPE REVIEW_ONLY_BOUND_OBSERVATIONS")
    print("BRIDGE_LEMMA_7B UNRESOLVED")
    print("all_certified False")
    return 0 if failures == 0 else 2

if __name__ == "__main__":
    raise SystemExit(main())
