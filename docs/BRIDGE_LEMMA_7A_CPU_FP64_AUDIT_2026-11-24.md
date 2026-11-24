# Bridge Lemma 7A — pinned CPU float64 execution checkpoint

**Recorded:** 2026-11-24  
**Scope decision:** Option A: lock the three-element, single-thread CPU execution evidence; broader tensor lengths and parallelism require separate audits.  
**Narrowed 7A execution evidence:** **COMPLETE** for the four measured modes and pinned build.  
**Full 7A certification:** **NOT_CERTIFIED**, pending adoption of the explicitly limited contract.  
**Overall certification:** `all_certified=False`; Bridge Lemmas 7B–7D unresolved.

This is an evidence manifest and conditional floating-point argument, **not** a backend-independent theorem or a claim of arbitrary-precision rendering. The original Windows CDB logs and local scripts remain on the audit machine; only their operator-reported cryptographic digests are recorded here.

## Provenance and reproducibility

- Repository baseline (frozen Bridge Lemma 6): `d0db6f43827218ef8b0ea5447edc1a385fffd5e6`.
- Python `3.13.1`, PyTorch `2.15.0+cpu`, x64 CDB `10.0.28000.2957`; one CPU thread.
- Pinned `torch_cpu.dll` SHA-256: `b8ec43a4ff24435c93a8a7710f9cf3fe1f246d923abbd7549064c8ba26afc642`.
- Selected runtime MXCSR: `0x00001fa0` (round-to-nearest-even, FTZ off, DAZ off; precision/inexact sticky flag set).
- Tested `float64` vectors of length `n=3`; strides `(1,)` (contiguous) or `(2,)` (strided); scalar `h` supplied as Python float or tensor; exact additive scaling factor `alpha=1.0`.
- Four modes: `contiguous_tensor_h`, `strided_tensor_h`, `contiguous_python_h`, `strided_python_h`.
- The observed runs reported all four `END_MODE` markers and `ALL_FOUR_MODES_COMPLETED`, matching product/output bit vectors across modes, and CDB exit code 0.
- These are specific finite test vectors (driver includes `h=1+2**-52`), **not** exhaustive coverage of every possible representable operand.

## Dynamic instruction-level findings

All counts below are genuine CDB breakpoint event markers, distinguished from echoed definitions, in the submitted Windows console/log transcripts.

| Checkpoint | Observed |
| --- | --- |
| Initial ADD/MUL dispatch entrypoints | 8/8 hits |
| Four conditional branch-target sites | 16/16 hits |
| Dispatch tails and callback trampolines | 16/16 hits |
| `ScalarType::Double` guard-success sites and post-trampoline bodies | 16/16 hits |
| Three-element remainder helper entrypoints | 8/8 hits |
| Monitored vector MUL at RVA `0xD3C7214` | 0 hits |
| Monitored vector FMA at RVA `0xCA42A71` | 0 hits |
| Scalar product MUL `vmulsd` at RVA `0xCE2D675` | **12 hits** (3 per mode) |
| Addition-alpha MUL `vmulsd` at RVA `0xCA3B572` | **12 hits** (3 per mode) |
| Final scalar ADD `vaddsd` at RVA `0xCA3B577` | **12 hits** (3 per mode) |
| Exact alpha binary64 at `[r8]` | **12/12** reads of `0x3ff0000000000000` |

At each arithmetic instruction, the live `r9` sequence was `3,2,1` per mode, with `MXCSR=0x1fa0`. The observed multiplication sequence preceded the paired addition-alpha MUL and final ADD instruction hits. This accounts for all three vector elements in every tested mode.

### Complete short-path disassembly

On the pinned DLL, a `.pdata`-bounded Capstone walk decoded:

- MUL epilogue `0xCE2D64B`–`0xCE2D692`: **71/71 bytes**, one scalar `vmulsd`, zero calls, zero indirect jumps, no FMA.
- ADD epilogue `0xCA3B54A`–`0xCA3B5AC`: **98/98 bytes**, scalar `vmulsd` followed by `vaddsd`, zero calls, zero indirect jumps, no FMA.

Selected instructions (RVA relative to the pinned `torch_cpu.dll`):

```asm
0xCE2D675   vmulsd xmm1, xmm0, qword ptr [rdx+rdi]  ; product
0xCA3B572   vmulsd xmm0, xmm0, qword ptr [r8]       ; alpha * product
0xCA3B577   vaddsd xmm1, xmm0, qword ptr [rcx+rbp]  ; final x + product
```

The monitored vector `vfmadd213pd` address was not hit in the tested `n=3` path. This **does not** exclude vector FMA for other tensor sizes.

## Conditional two-effective-rounding argument

For the executed scalar sequence, let `u=2^-53` be binary64 unit roundoff and `fl` round-to-nearest-even. Under finite, non-overflowing operations with no underflow invalidating the usual relative-error model:

```text
p_hat_i = fl(h*k_i)              = (h*k_i)*(1+delta1_i)
s_hat_i = fl(1.0*p_hat_i)        = p_hat_i
y_hat_i = fl(x_i+s_hat_i)        = (x_i+p_hat_i)*(1+delta2_i)
|delta1_i|, |delta2_i| <= u
```

The multiplication by exactly `1.0` is representationally exact for finite binary64 values (including finite subnormals when FTZ/DAZ are off). There is no FMA contraction between the initial `h*k_i` and final `+x_i` along the observed short scalar code path.

With `gamma_2 = 2u/(1-2u)`, the conditional **absolute forward error** satisfies

```text
|y_hat_i - (x_i + h*k_i)| <= gamma_2*(|x_i| + |h*k_i|).
```

**Do not convert this into an unconditional relative bound divided by `|x_i+h*k_i|`**, which may be tiny under cancellation. Input representation, overflow, NaN/Inf, and arithmetic underflow need explicit treatment if a wider domain is claimed.

## Operator-reported raw evidence inventory

These exact artifact sizes and SHA-256 values were produced by `Get-FileHash -Algorithm SHA256` on the Windows audit host on 2026-11-24. They are a **manifest**, not evidence that the raw files were uploaded to GitHub.

| Local artifact basename | Bytes | SHA-256 |
| --- | ---: | --- |
| `bridge_lemma_7a_cdb_four_modes.py` | 2,807 | `a13774b3f8aec40706496700eb38582a67edb6fd2ecf9094d72c1064682118db` |
| `bridge_lemma_7a_bounded_callgraph.py` | 16,008 | `0d9b3c3d9b74f3c3c86b2df734d5aa5b2415ee146eaedd4595ce9492f2d53d93` |
| `lemma7a_cdb_four_modes_unbuffered.log` | 36,804 | `9e0dd0e12350b6acac0d767a76e05ae5efc61c9b4ef251fe3d0b73c0add5c51f` |
| `lemma7a_cdb_branch_targets_02.log` | 61,233 | `224698b58d4cd3075a0e833e55f97d8a0b73543c551d0771ddbabd5cf94b0823` |
| `lemma7a_cdb_inner_loops_01.log` | 39,387 | `c5fd1a0e012d4a82f7a92c504777cbb494ee8a2601351c4940c36e75ce0b4dc5` |
| `lemma7a_cdb_guards_body_01.log` | 58,890 | `42e347cf392903ffaf52d2a3bc57a73e1d24e9f449f15022d0939b749d41d1f4` |
| `lemma7a_cdb_vector_vs_remainder_01.log` | 73,215 | `66a41e13cd35042375a697c71be26af2498057305bf8a03566bf803b8da98b80` |
| `lemma7a_cdb_scalar_fp64_exec_01.log` | 30,107 | `9a7fc2079318c807c1ca4efda2536df9895ea264de4982e4fe70b125848f114b` |

The raw logs and scripts are still stored under `%USERPROFILE%\Downloads` on the audit host; they are **not** represented as byte-for-byte GitHub files by this commit. Review their incidental paths and debugger data before any public upload.

## Open scope and restart checklist

**Out of scope:** `n=4..7` four-way-unrolled scalar loops; `n>=8` vectorized/FMA execution; arbitrary shapes/strides; other CPU builds or ISA dispatch configurations; multithreading and MXCSR of worker threads (potentially beyond a `0x8000`-element grain); nonfinite/overflow/underflow cases; and full-renderer correctness.

**Local worktree at last report:** no tracked changes; five existing, untracked files. The operator's 2026-11-24 PowerShell transcript now confirms each was byte-for-byte copied into a local checkpoint staging directory, but **none was committed to GitHub**:

```text
argon_geometry_audit.py
argon_v44_adversarial_audit.py
argon_v4_adversarial_audit.py
tests/test_backend_certification.py
twistor/backend_certification.py
```

**Resume:** (1) verify the checkpoint ZIP and preserve an independent off-machine copy; (2) inspect the five untracked source files and eight raw artifacts before deciding which are suitable for Git; (3) keep Bridge Lemma 6 frozen; (4) adopt the limited 7A contract or open a separate wider-domain audit; (5) proceed to 7B–7D. Until that happens, `all_certified=False` and **full 7A remains NOT_CERTIFIED**.


## Local checkpoint creation confirmed (2026-11-24)

An operator-provided PowerShell transcript records successful source-to-staging SHA-256 equality checks for **13 files**: the eight evidence artifacts listed above and the five previously untracked repository files. It also records successful creation of the `SHA256_MANIFEST.csv` file and of the local ZIP archive:

- Archive: `%USERPROFILE%\\Downloads\\argon_7a_checkpoint_20261124.zip`
- Archive SHA-256 (operator-reported): `fcbb0f2224b34d21b7b2f4ebcbe13f5d20bf2161fac610a7686ada186488034d`
- Archive contents intended: `repo/` (five files), `evidence/` (eight files), and `SHA256_MANIFEST.csv`

**Evidence limitations:** only the console transcript was provided in chat, **not the ZIP bytes**. This records a successful local packaging action, not independent verification of ZIP-entry integrity or an off-device/cloud backup. The 13 file bytes remain absent from this GitHub branch. Do not report the source code or raw logs as pushed until they are reviewed and uploaded intentionally.

## Archive integrity independently verified (uploaded ZIP)

The actual ZIP bytes were subsequently uploaded to the audit conversation and inspected in a filesystem-backed Python runtime. Validation was **not** limited to ZIP directory metadata:

- Full ZIP byte length: **74,150**; SHA-256: `fcbb0f2224b34d21b7b2f4ebcbe13f5d20bf2161fac610a7686ada186488034d`, matching the operator's recorded digest.
- All **14 ZIP members readable**; CRC integrity scan reported no corrupt member.
- Uncompressed payload size: **413,858 bytes**.
- `SHA256_MANIFEST.csv`: **13 data rows**.
- All **13/13 decompressed payload SHA-256 hashes matched their manifest rows**.
- No unexpected or missing archive entries. ZIP member names use Windows backslash separators.

The ZIP is confirmed intact **as supplied to the conversation**. It is **not** stored in this public GitHub branch, and an independent durable/off-machine copy remains a separate action.

### Source review — provisional, not a backend certification

All five Python files in `repo\\` were parsed and compiled without syntax error in the reviewing Python 3.13 environment. The three older Argon audit scripts appear to be separate, frozen/adversarial gates; the raw SHA-256 of `argon_geometry_audit.py` was verified to match the literal baseline pinned by `argon_v44_adversarial_audit.py`, and its BOM/CRLF-normalized hash checks also matched.

Review of `twistor/backend_certification.py` and `tests/test_backend_certification.py` identified an important **model-vs-runtime certification gap**:

1. `verify_7a_graph_capability` matches a declared `GraphSpec` by string/field comparison; `capture_runtime_backend` records environment details but does **not** enforce the pinned DLL SHA-256, PyTorch `2.15.0+cpu`, CPU-only device, one-thread regime, or MXCSR parity.
2. `bound_7a_ray_advance_euclidean_float64` can label its result `analytic_proven` based on the declared graph and arithmetic-range check alone. This must be represented as a **conditional model bound**, not automatically as live backend certification. It also silently converts arbitrary real input tensor dtypes to float64 instead of enforcing the narrow tested input contract.
3. The normal/subnormal rejection compares the **upper endpoint** of an absolute-product/sum interval against the minimum normal value. A fail-closed implementation should reject whenever the interval might intersect a nonzero subnormal regime, not only when its upper bound is already subnormal. This must be reviewed before promoting the gate.
4. `candidate_7b_discriminant_bound_float64` explicitly marks its scale-based `12uM` estimate `unresolved`. The partial `BackendErrorBudget` keeps 7B–7D unresolved and `all_certified=False`; this is correct and must be preserved.
5. The bundled unit tests cover cancellation/underflow/overflow, legacy-file hash, GraphSpec mismatch, Decimal outward conversion and non-promotion of unresolved 7B–7D, but **do not prove the pinned runtime instruction path**, which was established separately by CDB.

**Commit policy:** do not silently promote these preliminary source files to the production branch. Review/repair the runtime-versus-model certification distinction and add negative tests first. Keep the three frozen Argon scripts byte-for-byte intact. No raw debugger logs are published in this documentation-only PR.

**7B next experiment:** inspect and bound the exact `twistor.congruence.delta` operation graph (complex128 `(z+ib)^2` plus real `x*x+y*y`), including the backend's complex multiplication/addition semantics, before attempting to upgrade `12uM` from candidate to certified. Use a high-precision exact-input oracle, adversarial cancellation cases, and fail-closed overflow/underflow checks. Retain `all_certified=False` until all obligations pass.

### CDB logfile versus console-output coverage

Direct inspection of the uploaded **raw** `lemma7a_cdb_scalar_fp64_exec_01.log` confirms 36 genuine hardware breakpoint events, exactly 12 at each of three scalar arithmetic instructions; 12 exact `alpha=1.0` reads; `MXCSR=0x1fa0` at all 36 hits; and four repeated nine-event groups (3 MUL, 3 pairs of ADD-MUL/ADD). This raw CDB `-logo` file does **not** itself contain Python child-stdout labels `BEGIN_MODE`, `END_MODE`, or `ALL_FOUR_MODES_COMPLETED`, nor the PowerShell-reported CDB exit code. Those mode-boundary and completion observations came from the separate **terminal transcripts** supplied during the audit. The four repeated groups align with the driver's known sequential four-mode order, but **per-mode attribution is not independently encoded in the raw `-logo` bytes**. Keep the console transcript with the archived evidence; for future work, capture both debugger log and child stdout in a single durable report when possible. No further CDB run is required solely to re-establish the already reviewed narrowed 7A path.
