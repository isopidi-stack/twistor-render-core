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


## Gate 3D executed FP instruction hits — runtime evidence (2026-11-24)

The operator supplied the complete PowerShell/CDB transcript of a pinned `torch_cpu.dll` run, using the same source driver as Gate 3B. The Windows launcher checked the DLL SHA-256 `b8ec43a4ff24435c93a8a7710f9cf3fe1f246d923abbd7549064c8ba26afc642`; the driver SHA-256 was reported as `37038c39990a6f02f7b7b7e7d3c245c09eca1f0b10453e3bf72ba068b62477ea`. The log-file SHA-256 was operator-reported as `35b5d9f843192cb4dd49e33d54d3dd20ea4bbcbbd8ca82dc9ca1227cd1be972b`. The raw `lemma7b_gate3d_arithmetic_01.log` is local only; the actual uploaded chat material is the console transcript, **not raw LOG bytes**.

**Initial CDB error messages did not invalidate the test**: all 21 breakpoints were first registered as deferred because `torch_cpu.dll` had not loaded. On module load the expressions resolved and actual breakpoint event markers were emitted at the expected instruction addresses.

Counting only genuine runtime event-marker lines, not breakpoint definitions or warnings:

| Observed phase / instruction group | Genuine hits |
| --- | ---: |
| `TensorIteratorBase::serial_for_each` callback dispatch | **23** |
| Complex128 vector multiplication: six `vmulpd` / `vhsubpd` sites, twice each | **12** |
| Real multiplication: four monitored `vmulpd` sites | **0** |
| Real addition: `vfmadd213pd` at `0xCA42A71` and `0xCA42A77` | **2** |
| Complex128 final vector addition: eight `vmulpd` / `vhsubpd` / `vaddpd` sites, twice each | **16** |
| **All 20 monitored arithmetic sites** | **30 hits** at **16 distinct sites** |

All genuine 23 dispatch and 30 arithmetic hits recorded `MXCSR=0x1fa0`. All **30 arithmetic hits occur in the batched `(8,3)` mode**; none of the 20 monitored *vector* sites fired in either scalar `(3,)` mode, although those modes executed and returned the expected result. The callback dispatch has `r9=1` on scalar modes, `r9=3` on the intervening setup callback, and `r9=8` on the batched mode.

The batched mode recorded these instruction orders:
1. Complex multiplication: the six selected vector FP instructions, **two passes** (`n=8`; this does not itself prove exactly which element lanes map to which inputs).
2. Real multiply callbacks, **zero hits** at the four chosen packed `vmulpd` sites; the executed real-multiplication instructions remain to be located.
3. Real-add callback: two **executed** `vfmadd213pd` instructions.
4. Final complex-add callback: eight selected packed FP instructions, **two passes**.

**Interpretation:**

- The executed `vfmadd213pd` instructions conclusively demonstrate vector FMA instructions on the selected `n=8` real-add code path. They do **not** prove that mathematical `x*x + y*y` is contracted across the separate PyTorch tensor operations: in generic ATen addition, an FMA can implement `a + alpha*b` where `alpha=1.0`. The scaling factor and exact operands must be checked at the live instruction.
- The observed complex multiplication `vmulpd` / `vhsubpd` vector sequence is consistent with separately rounded products before real/imaginary subtraction, explaining the previously measured `3cd0000000000000` FMA discriminator. The selected body contains no statically decoded FMA, but a complete no-FMA claim requires ruling out other branches/helpers.
- Zero `RMUL` hits do **not** mean multiplication was skipped; they show only that these particular packed sites were not exercised. The scalar path and the exercised real-mul arithmetic are still uncharacterized.
- The vector complex-add sequence includes multiplications and subtractions **before** vector adds. An exact-alpha=1 proof requires direct operand inspection; do not assume these are zero-error just from their names.

The driver reported matching scalar first-element and batched row-0 output bits (`re=3cd0000000000000`, `im=4012000000000001`), `ALL_7B_DISCOVERY_MODES_COMPLETED` and `CDB exit code: 0`. The local Git worktree still showed the same five pre-existing untracked files and no tracked changes.

**Gate 3D status:** **COMPLETE as a dynamic hit census of the selected 20 vector instruction sites**, **not** full instruction-level certification of `delta()`. Bridge Lemma 7B remains `status="unresolved"`; `all_certified=False`.

**Next Gate 3E:** resolve scalar-remainder/direct-call branches for complex multiplication, real multiplication, and final complex addition; inspect operands at executed real-add FMA sites (including exact alpha), and establish branch completeness before deriving or certifying `12uM`.


## Gate 3E bounded branch and callee inventory (2026-11-24)

The operator supplied the complete `lemma7b_gate3e_static_edges_01.txt` report generated by the read-only `bridge7b_gate3e_static_edges.py` helper. Independent inspection of the **actual uploaded file** confirms UTF-16LE, 33,964 bytes, 212 text lines, and SHA-256 `ba8f003d4c4bec1b8eb41bae976b47a1ca4afef607a14922135e2420c42af84e`. Its reported pinned DLL and 7A helper SHA-256 values matched the previously anchored digests. The inspecting source loops over four selected `.pdata`-bounded root functions, detects immediate direct calls, and prints selected FP arithmetic, conditional branches, and immediate callee bodies. No code was changed in the Git repository.

| Root body | Conditional branch edges | Unique direct-call targets | Additional direct-callee FP instructions |
| --- | ---: | ---: | --- |
| `COMPLEX_MUL 0xD3C8140` | 18 | 3 | None detected |
| `REAL_MUL 0xD3C7120` | 22 | 3 | `vmulpd` at `0xD342B60`, `0xD342B66` in immediate callee `0xD342A90` |
| `REAL_ADD_FMA 0xCA42980` | 11 | 3 | `vfmadd231pd` at `0xCA3CA3E`, `0xCA3CA49` in immediate callee `0xCA3C960` |
| `COMPLEX_ADD 0xCA40AF0` | 18 | 3 | None detected |

The four **new** FP sites above were not instrumented in Gate 3D. The two new vector FMA sites show that more than one *static* FMA body may implement the real-add kernel. Gate 3D demonstrated execution of the original `vfmadd213pd` pair at `0xCA42A71` and `0xCA42A77` in the `n=8` test; it did **not** demonstrate whether the newly found `vfmadd231pd` sites execute in any tested mode.

The inspected four root bodies each contain conditional dispatch paths and external direct-call destinations; for each root the tool decoded three immediate direct callees. `unresolved_items=0` only means successful bounded static decoding of **those selected targets**. The script does **not** recursively follow second-level calls, indirect jumps, possible CPU-specific dispatch, or collect dynamic branch coverage. It only treats SIMD/SSE-style mnemonics ending in `sd/pd/ss/ps` (and FMA names) as FP arithmetic; unsupported FP sequences could be omitted from that inventory.

**Unresolved scalar problem:** Gate 3D's 20 selected packed arithmetic sites recorded no hits in the `n=1` modes, even though the driver completed and produced valid output. The immediate-callee list by itself does not identify the executed scalar arithmetic instructions. Avoid asserting the scalar real/complex arithmetic graph or generalizing a no-FMA statement from the `n=8` selected sites.

**Next experiment:** instrument the newly discovered four arithmetic RVAs **and** the immediate-callee entry RVAs on the pinned build, with explicit mode and MXCSR log attribution. For scalar path proof, analyze further dynamic branch/control flow from the callback bodies, following indirect/tail targets as necessary. Maintain `candidate_7b_discriminant_bound_float64` as `unresolved` and `all_certified=False`.


## Gate 3F attempted run — breakpoints not armed (2026-11-24)

The operator supplied the actual debugger log `lemma7b_gate3f_helpers_02.log` and matching PowerShell console transcript. The downloaded log was independently SHA-256 verified as `8e7e3cbbe72127abae0d4bc72f5206b044c0510c63f99ad59b4c985cb575ff6c`, exactly matching the operator-reported hash. The log shows `torch_cpu.dll` loading successfully at the pinned image base and CDB's direct disassembly at `torch_cpu+0xD342B60` producing `vmulpd ymm0,ymm2,[rdx+rcx-0x20]`; later disassembly of `torch_cpu+0xCA3CA3E` produced `vfmadd231pd`. However, the CDB command file containing the breakpoints was **never invoked** at the DLL-load stop. The operator entered `g` immediately after the first successful `u`; the driver then completed all three modes with the expected bits, without instrumentation.

Neither the actual debugger log nor the console transcript contains `L7B_GATE3F_SETUP`, `L7B3F_DISPATCH`, `L7B3F_FP_`, or `L7B3F_HELPER_` markers. Thus **zero successful Gate 3F breakpoint hits** were collected. The recorded CDB exit code 0 does not imply breakpoint-trace success.

**Status:** `Gate 3F NOT EXECUTED` (invalid instrumentation attempt; DLL load and two static disassembly instructions only). A correct retry must invoke the already-preserved `lemma7b_gate3f_helpers_01.cdb` at the `1:005` stop immediately after `ModLoad ... torch_cpu.dll`, confirm its setup marker and 14 enabled resolved breakpoints, and only then issue `g`. Preserve the failed run log and do not change `all_certified=False`.
