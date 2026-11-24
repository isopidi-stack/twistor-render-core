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


## Gate 3B callback discovery — runtime evidence (2026-11-24)

The operator supplied an approximately 898-line **CDB/PowerShell console transcript** for a pinned Windows `torch_cpu.dll` run (expected SHA-256 `b8ec43a4ff24435c93a8a7710f9cf3fe1f246d923abbd7549064c8ba26afc642`, validated by its PowerShell launch guard). The DLL loaded at base `0x31AF0000`. CDB set a breakpoint at `torch_cpu+0x8DA1BA1`, previously validated as `call qword ptr [r12]` in `TensorIteratorBase::serial_for_each`; the breakpoint command printed `r9`, `mxcsr`, and callback target `poi(@r12)-torch_cpu`.

The transcript also contains a **second PowerShell summary that reprints the same breakpoints**. Counting only original runtime lines gives **23 genuine `L7B_SERIAL_CALL` breakpoint hits**:

| Observed mode/phase | Hits | `r9` | Qualification |
| --- | ---: | ---: | --- |
| `scalar_3d_contig` | 7 | 1 | Scalar coordinate views (0D) |
| Setup between scalar modes | 1 | 3 | Occurs before `BEGIN_7B_MODE scalar_3d_strided`; consistent with a stride-2 buffer copy, but not individually tagged |
| `scalar_3d_strided` | 7 | 1 | Scalar coordinate views (0D) |
| `batched_8x3_contig` | 8 | 8 | One additional kernel callback versus scalar case |
| **Total** | **23** | | |

Every one of these **23** breakpoint stops reported `MXCSR=0x00001fa0` (round-to-nearest-even, FTZ and DAZ off, sticky inexact flag). This establishes the floating-point environment **at the indirect-call boundary**, not yet at every instruction inside each callback.

### Callback entry RVAs observed at the indirect-call site

| Callback entry RVA | Genuine hits | Grouping / limitations |
| --- | ---: | --- |
| `0x0D2F56F0` | 3 | One per main `delta()` evaluation |
| `0x0D3238A0` | 3 | One per main evaluation; entry is a jump thunk |
| `0x0D323840` | 6 | Two per main evaluation; entry is a jump thunk |
| `0x0CA3B670` | 3 | One per main evaluation; entry is a jump thunk |
| `0x0AEE9D90` | 3 | One per main evaluation |
| `0x0CA3B5D0` | 3 | One per main evaluation; entry is a jump thunk |
| `0x0D262BA0` | 1 | Inter-mode setup callback, `r9=3` |
| `0x0D2203F0` | 1 | Additional callback in batched case, `r9=8` |
| **Total** | **23** | |

The seven shared callbacks occur in the same order for both scalar modes and the batch. **Per-operation identifications are not yet individually tagged in the CDB stream**; based on the source expression, their order suggests complex construction, complex product, two real products, real sum, real→complex conversion, final complex addition. The batched-only extra callback may be initialization/fill but remains to be identified by operation-specific instrumentation. Several call targets are jump thunks, so disassembly of only their first eight instructions is insufficient to prove the arithmetic graph.

Some decoded callback addresses were assigned misleading nearby stripped-DLL symbolic names such as `torch::xpu::synchronize` or `torch::nnapi::bind`; these are **not evidence that an XPU/NNAPI device was used**. Prefer RVA plus verified instruction bytes.

### Completion and limitations

The three modes each reported matching output bits for the discriminating first-row input: real `3cd0000000000000`, imaginary `4012000000000001`.

The driver printed `ALL_7B_DISCOVERY_MODES_COMPLETED` and launcher `CDB exit code: 0`. Git status remained unchanged: five original untracked files and **no tracked modifications**.

**Gate 3B status:** complete for **callback discovery at this one serial-for-each call site**; not a full dynamic instruction proof. The script armed the breakpoint at module load and used an indirect executable-address watch (`ba e 1`), rather than an explicit DebugBreak after warm-up. Other execution pathways bypassing this site have not been excluded. No actual scalar/AVX2/FMA arithmetic opcode hit counters were collected.

**Next:** resolve observed callback RVA jump thunks, decode each resolved PE `.pdata`-bounded function, identify FP64 arithmetic instructions, then place targeted **runtime breakpoints at actual arithmetic instructions**, checking alternative branches/calls before certifying any error constant. `candidate_7b_discriminant_bound_float64` stays `status="unresolved"`; `all_certified=False`.


## Gate 3C pinned-DLL static arithmetic inventory (2026-11-24)

The operator ran the hash-pinned, read-only `bridge7b_gate3c_static.py` inspection and supplied its full UTF-16LE PowerShell transcript `lemma7b_gate3c_static_01.txt`. In the reviewing runtime, the actual uploaded file has size **9,770 bytes** and SHA-256 `024af2f724471812e684b547ee2b629232381ec82f9152468d672af3c240d4a2`. The helper script SHA-256 is `c3a84cbdfd04c97a1da460eebdc7d8c823d4b67bc8f3ad10ab2887344f74a31f`. The report confirms the loaded DLL hash `b8ec43a4ff24435c93a8a7710f9cf3fe1f246d923abbd7549064c8ba26afc642` and identifies Capstone under the isolated tools installation.

For eight dynamically discovered callback entries the script followed entry-point direct jump thunks to a PE `.pdata` function and linearly decoded **100% of the reported function byte ranges**. Key static findings:

| Candidate source operation | Callback entry -> resolved body RVA | Floating-point instruction candidates *in decoded body* |
| --- | --- | --- |
| complex128 `v*v` | `0xD3238A0 -> 0xD3C8140` | 4 `vmulpd`, 2 `vhsubpd`; **no FMA** in this decoded body |
| real `x*x` and `y*y` | `0xD323840 -> 0xD3C7120` | 4 `vmulpd`; **no FMA** in decoded body |
| real `x*x + y*y` | `0xCA3B670 -> 0xCA42980` | **2 `vfmadd213pd`** at `0xCA42A71` and `0xCA42A77` |
| final complex128 addition | `0xCA3B5D0 -> 0xCA40AF0` | 4 `vmulpd`, 2 `vhsubpd`, 2 `vaddpd`; **no FMA** in decoded body |
| complex construction / cast candidates | `0xD2F56F0` / `0xAEE9D90` | No arithmetic mnemonics in selected bodies, but **2 indirect calls each** |
| inter-mode setup / batched-extra callbacks | `0xD262BA0 -> 0xD2F2C40` / `0xD2203F0 -> 0xD223160` | No selected arithmetic mnemonics |

The result **does not prove** execution of either FMA. In particular, the real-add FMA candidates coincide with the previously unhit vector FMA site in the *7A n=3* audit; 7B's `n=8` case may take a distinct branch. No arithmetic instruction breakpoints were hit in this 3C run.

**Important limitation:** the script's `unresolved_items=0` means successful entry-thunk resolution and contiguous linear decoding of each *selected* `.pdata` range; **it is not transitive control-flow coverage**. The complex-multiplication callback shows packed arithmetic but no scalar FP instructions in its listed range, despite executed `n=1` tests. Scalar remainder/helper paths (including direct calls) and all conditional branches need to be traced explicitly. A successful contiguous decode also does not demonstrate the whole decoded byte range is traversed.

**Next Gate 3D:** set software instruction breakpoints at the specific packed `vmulpd` / `vhsubpd` / `vaddpd` and `vfmadd213pd` RVAs, retain the verified `serial_for_each` dispatch breakpoint, and compare actual n=1 versus n=8 hits and MXCSR. Follow separate scalar helpers if packed instructions are not reached. Do not use the observed `3cd0000000000000` probe or this static scan alone to certify a no-FMA complex128 path. Keep 7B `unresolved` and `all_certified=False`.
