# Bridge Lemma 7B — pinned CPU exact-rational oracle checkpoint

**Recorded:** 2026-11-24  
**Status:** **Gate 2 empirical oracle complete** for the specified input set; **Bridge Lemma 7B remains UNRESOLVED**.  
**Overall:** `all_certified=False`; Bridge Lemma 6 frozen; narrowed 7A instruction evidence complete on its separately documented scope; 7C–7D unresolved.

This document records operator-provided output from an unmodified local Windows run. The full console transcript was supplied to the audit conversation; the actual log bytes are **not** committed here. No source files were modified by this documentation checkpoint.

## Exact source graph and test environment

At frozen source baseline `d0db6f43827218ef8b0ea5447edc1a385fffd5e6`, `twistor/congruence.py` evaluates:

```python
b_val = _validate_b(b)
pos64 = pos.to(dtype=torch.float64)
x, y, z = pos64[..., 0], pos64[..., 1], pos64[..., 2]
v = torch.complex(z, torch.full_like(z, b_val))
return v * v + (x * x + y * y).to(dtype=torch.complex128)
```

The mathematical exact reference for the **represented** binary64 inputs is:

```text
Delta_exact = (z*z - b*b + x*x + y*y) + 1j*(2*z*b)
M = x*x + y*y + z*z + b*b
u = 2**-53
```

The oracle converted each input to `fractions.Fraction.from_float`, evaluated the reference expressions over exact rationals, and compared **squared** complex error against squared candidate bound `(12*u*M)^2` over rational arithmetic. A float approximation was used only to *print* error/bound ratios, not to decide containment.

- Python `3.13.1`; torch `2.15.0+cpu`; `torch.set_num_threads(1)`; CPU `float64` / `complex128`.
- Windows launcher checked `torch_cpu.dll` SHA-256 equals `b8ec43a4ff24435c93a8a7710f9cf3fe1f246d923abbd7549064c8ba26afc642` before running.
- Scalar `(3,)` contiguous and stride-2 input vectors become **0D** `x,y,z` tensors when indexed; they do **not** exercise non-unit inner-kernel tensor strides.
- Batched `(8,3)` contiguous and stride-2-row input layouts yield coordinate views with element strides 3 and 6, plus eight independent `(3,)` calculations.

## Observed scalar results

The run included **29 named cases** × **2 input layouts** = **58 evaluations**.

| Result | Count | Interpretation |
| --- | ---: | --- |
| `WITHIN` candidate `12uM` | **54** | All 27 nonexceptional cases, twice |
| `EXCEEDS` | **2** | Deliberate underflow stress, twice |
| `NONFINITE_OUTPUT` | **2** | Deliberate overflow stress, twice |
| Contiguous-vs-strided bit mismatches | **0** | The tested scalar outputs |
| Max nonexceptional printed ratio | **0.0523783** | `|Delta_hat-Delta|/(12uM)` |

Underflow stress: `x=b=z=2**-550, y=0`; squared terms underflow and the error exceeds the candidate. Overflow stress: `x=b=z=2**600, y=0`; output is nonfinite. These are deliberate **negative controls**, not failures of a proof limited to nonexceptional arithmetic.

For the `z_near_b_fma_probe` with `z=nextafter(1.5,+inf)`, `b=1.5`, `x=y=0`, both scalar layouts reported `re_bits=3cd0000000000000` and ratio `0.0523783`. That real result matches a separately rounded `z*z` and subsequent subtraction of `b*b`; a direct fused multiply-subtract of the exact operands would instead round to `3cc8000000000000`. **This is not proof of a no-FMA kernel**: other machine arithmetic arrangements or FMA elsewhere can yield the same output.

## Observed batched results

- All **eight** batched rows were `WITHIN` `12uM`.
- `BATCH_0D_DIVERGENCES: 0`: both output components matched bit-for-bit across batched-contiguous, batched-strided, and individual 0D computations.
- Maximum printed batched-row ratio: **0.0740741**.
- Bit parity does **not** prove the native execution paths match or that AVX2 vectorization occurred. `torch.complex` builds a new complex tensor whose layout may differ from the input coordinate views.

## What remains unproved

- Native, executed FP64 instruction graph of complex128 `v*v`, real `x*x+y*y`, conversion and final complex addition (including any FMA and helper calls).
- Live MXCSR (rounding, FTZ, DAZ) at the relevant instructions. Gate 2 checked binary hash and thread count but did not trace MXCSR.
- A rigorous nonexceptional domain covering cancellation and intermediate underflow, and a formally conservative bound matching the executed graph.
- Outward-rounded bound implementation: simply constructing exact `u=2^-53` as a Decimal does not guarantee future Decimal arithmetic is upward-rounded without an appropriate context.

**Next gate:** isolate relevant ATen operations on the pinned runtime and map the actual complex128/real scalar instruction paths with CDB, using the above FMA discriminator as a cross-check. Keep `candidate_7b_discriminant_bound_float64` unresolved and `all_certified=False`. No unreviewed source or raw memory-address logs should be published on the public repository.


## Gate 3A: verified TensorIterator indirect call site (static; no execution claim)

On the pinned DLL (SHA-256 above), the operator ran a **read-only Capstone decoding script** using the independently pinned `bridge_lemma_7a_bounded_callgraph.py` helper and isolated Capstone installation. The decoded executable window `0x8DA1B70–0x8DA1BA5` ends with:

```asm
0x8DA1B99  mov rdx, rax
0x8DA1B9C  mov rcx, qword ptr [r12 + 8]
0x8DA1BA1  call qword ptr [r12]   ; return RVA 0x8DA1BA5
```

The script checked that the instruction ending at `0x8DA1BA5` is exactly one **indirect** `call` whose operand is memory, and that the call lies in the PE `.pdata` function range `0x8DA1B30–0x8DA1FE9`. It reported:

```text
L7B_INDIRECT_CALL_CONFIRMED 0x8da1ba1 call qword ptr [r12] return_rva= 0x8da1ba5 function_range= ('0x8da1b30', '0x8da1fe9')
```

The local Git worktree was again unchanged: five previously untracked files and no tracked modifications.

**Limit:** this is **static instruction identification only**. Neither the total number of callbacks through this site nor membership of each `delta()` operation has been established. A CDB execution trace with per-operation markers, callback target capture and MXCSR is the next test. The claim that *all* complex128 or conversion operations pass through this one site remains unproved.
