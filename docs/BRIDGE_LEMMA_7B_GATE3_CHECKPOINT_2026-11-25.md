# Bridge Lemma 7B — Gate 3 consolidated checkpoint (2026-11-25)

**Review only; not a backend certification.** Supersedes the *current status* at the end of [the earlier Gate 2/3 audit](BRIDGE_LEMMA_7B_GATE2_ORACLE_2026-11-24.md), retaining its historical failed runs. Frozen production main: `d0db6f43827218ef8b0ea5447edc1a385fffd5e6`. Draft PR #2. No production source changes authorized or applied. Bridge 6 remains conditional and frozen; 7A retains its earlier narrow model disposition. 7B, 7C, 7D unresolved; `all_certified=False`.

## 1. Pinned environment and scope

Windows AMD64; Python 3.13.1; torch 2.15.0+cpu; reported PyTorch source revision `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`; AVX2, one intra-op thread (16 inter-op threads in survey). Proposed initial contract: ordinary CPU strided contiguous float64 tensor of shape (3,), stride (1,), no grad, exact built-in float b finite/nonzero, no override modes/functorch/inference/autocast, pinned dispatcher/runtime, RN-even gradual underflow, stable privately owned input snapshot. No general batch certification. Explicit trusted-process assumptions are required; hashes of backing files and pre/post snapshots do not authenticate executable pages or establish continuous FP state.

| Pinned file | SHA-256 |
|---|---|
| torch_cpu.dll | b8ec43a4ff24435c93a8a7710f9cf3fe1f246d923abbd7549064c8ba26afc642 |
| torch_python.dll | 9865b16e791fc6f91fe92b38a930004a9fedaa60b3d70645b872f5d640574f29 |
| c10.dll | 60cae8210a16da67910754008ff62266cf529363fdcd032cd50a6b949dc4ea21 |
| twistor/congruence.py | 481e8f194ad225ca866e1b383e34a85433cc91957ef8a241db10e7c73cb68f9c |
| twistor/certified_ray.py | eb63abb0c90fc5677f511734474c634ee18b7315742b8963530e8ed33dc7d150 |
| twistor/certified_selector.py | 2d53d7058a77265d4df9a8087ac49ec3e52424ff0aef5360dad352cbc1f877fd |
| twistor/interval_bounds.py | 15f6017305430397a94638f81fe0a3d04b6d88332e6da0fc049558aba5a3ba9c |
| local untracked twistor/backend_certification.py | 84ae48d6af4949c2374382db66357b530e0aee963d91ff17490a05d3b341ac7d |
| local untracked tests/test_backend_certification.py | 28ee17985b96f50aeac962ff95678f4e0b0008cdb6f30d8ef6c8ef2492c196f6 |
| discovery driver | bfd201ce5f18e6b6266202e95cc8971b11bfaf5b58a1970c989f42e1d9189a4d |

Preserve all five pre-existing untracked files: argon_geometry_audit.py, argon_v44_adversarial_audit.py, argon_v4_adversarial_audit.py, tests/test_backend_certification.py, twistor/backend_certification.py. No git clean, no force push, no merge to main. Worktree-status comparison alone is not content attestation.

## 2. Gate 3G — two-stream first-hit survey accepted (bounded)

Initial boundary pass count bare 16 was hexadecimal (22 decimal). Corrected to 0n16. Run 20261125_015412 had 355 resolved breakpoint lines, zero unresolved, 118 events (53 scalar, 65 batch), but strict acceptance failed because CDB -logo omitted child stdout markers. V2 uses separate debugger and child stdout streams. Run 021818 incomplete: batch bl inventory missing and two premature unresolved disassembly commands. Preserve all failed artifacts. Final run 20261125_023649_v2: exact batch boundary r9=8 callback 0xD2203F0, 178 scalar and 177 batch resolved inventory, 118 events (98 edges, 18 entries, 2 FMA), Python exit zero and debugger exit zero; composite PASS_BOUNDED_FIRST_HIT_SURVEY_TWO_STREAM. Original native analyzer exit 2 due stdout separation; cross-stream chronological ordering not established. Accepted as bounded survey only. The operator logged closure.

Reported SHA-256 for original 023649_v2 artifacts (not independently rehashed here):
- debugger.log: 5fc725518859c7f66abc8764b1c6a216ef2e54d093ad3b7111d19f28303164d2
- python_stdout.txt: 2e224f3542b7d6d2dc08056502c83b6c38607e2befe4101e5e34713c807da6d0
- native_summary.json: 61d03272cee0074a6fbd008d845c6177bdda55fdae1c968eec0693fbfc687701
- evidence.txt: d8717ef160e836eb7868d00bc03ba86cc3a827cbe5220d4d413c666daf730b4c

## 3. Gate 3H — native arithmetic evidence

A: initial bounded .pdata disassembly clipped helpers at unwind boundaries; A2 expanded real arithmetic windows (21 FP sites); A3 complex root/helper branch-following (84 FP sites, six roots, no unresolved intraprocedural transfers); A4 specialized roots and sign mask resolved, raw REVIEW_REQUIRED retained because two security-failure path transfers unresolved. Mask at 0x10454240: lanes +0 sign mask, sign bit, +0, sign bit. Normal cookie return is conditional, not full interprocedural closure.

Observed scalar helpers (RVAs relative to torch_cpu.dll):
- CMUL CE34F30, remainder CE35118..CE35165: four separately rounded multiplications, subtraction of real products, addition of cross products, two stores.
- RMUL CE2D4B0, scalar tail CE2D670..CE2D67E: vmulsd and store.
- RADD CA3B370, tail CA3B56D..CA3B580: alpha=1 multiplication, vaddsd and store.
- CADD CA3B690, tail CA3B8B0..CA3B919: complex alpha=(1,0), four products, subtraction/addition and addition to first operand.
- Complex packing D2F5A00, scalar tail D2F5B60..D2F5BA0: bit-preserving loads/stores.
- Float64-to-complex128 conversion AEE9D90, integer copy loop AEE9E8C..AEE9EA6: copy real bits and write positive zero imaginary.
- Batch CMUL D3C8140; RADD CA42980 uses vfmadd213pd at CA42A71 and CA42A77 with exact unity coefficient. This is RN(py*1+px), not contraction across separate square operations. Other static FMA sites CA3CA3E/49 and CA3C83E/49 not established as dynamically executed in these modes. No global no-FMA claim.

B: discriminator run 20261125_030216, 24 dispatches, 163/163 recorded MXCSR snapshots 0x1FA0. Scalar/batch arithmetic captured; first scalar conversion not captured, one CADD checkpoint missing, x/y zero. Three debugger command errors, three Ctrl+C events, three breakpoint redefinitions; retain raw log, not zero-error claim. Python exit zero confirmed; original wrapper post-report missing; recovered status/hash evidence. Native batch stores and conversion/CADD checkpoints reconciled; bounded accepted with qualifications.

C: nonzero eight-row run 20261125_055407_176, 56 resolved breakpoints, 203 arithmetic/capture markers, 23 dispatch/return pairs, all observed MXCSR 0x1FA0. 18 arithmetic return buffers (90 qwords) checked against exact rational rounding graph in prior review; Python stdout 10/10 graph/bound passes, four rows distinguish stepwise rounded graph from once-rounded exact polynomial. No complete upstream coverage; setup/fill/copy output provenance not exhaustively captured. Original wrapper post.json missing, recovered read-only integrity evidence. Bounded nonzero evidence accepted with qualifications.

Byte hashes independently recalculated from the actual uploaded B/C artifacts during this consolidation:
| Artifact | Bytes | SHA-256 |
|---|---:|---|
| lemma7b_gate3h_b_20261125_030216_debugger.log | 209649 | 1b13289472c8a4251bca23b90e7369966148f5a4fe50f54d094d44e47128a2d9 |
| lemma7b_gate3h_b_20261125_030216_python_stdout.txt | 441 | 2e224f3542b7d6d2dc08056502c83b6c38607e2befe4101e5e34713c807da6d0 |
| lemma7b_gate3h_c_20261125_055407_176_debugger.log | 82899 | e565b32a73254601e8b98eb45f3c79248634d262ba2ffd53ed04266d51fe7a30 |
| lemma7b_gate3h_c_20261125_055407_176_python_stdout.txt | 5497 | fd8b8013e73af6d238e79d1f225c5ab29f61774ed45009faae308465a4a91aab |
| lemma7b_gate3h_c_20261125_055407_176_pre.json | 3060 | 5f1da330491dce7ef0529dde14dd4767e01b1c451f093004d01b4bf64aeea797 |
| bridge7b_gate3h_c.zip | 7050 | 1266c35019f44a0347ce39b7624c19fb127a4b239ac2700dbb863009fdf0c390 |
| bridge7b_gate3i_l_binding.py | 11631 | 1a726e20c9ea90b8a39121ce2cf0aa73d67c0793b5101bc8859f9726dd9a7f07 |

## 4. Gate 3I — conditional theorem

For the represented binary64 x,y,z,b, define Delta=x²+y²+z²-b²+i(2zb), M=x²+y²+z²+b², u=2^-53, eta=2^-1075. Candidate bound |Delta_hat-Delta| <= 12uM. Exact admission predicate 2^-1022 <= M <= 2^1022, b finite/nonzero. No claim about original unrounded physical inputs.

Assuming the observed scalar graph and RN-even with gradual underflow:
```
px=RN(x*x); py=RN(y*y); pz=RN(z*z); pb=RN(b*b)
t=RN(z*b); s=RN(px+py); r=RN(pz-pb)
Delta_hat.re=RN(r+s); Delta_hat.im=RN(t+t)
```
Exact unity scaling (and zero products) preserves finite magnitudes, allowing signed-zero differences only. Product absolute error <= u|ab|+eta. Addition/subtraction with a subnormal exact result is exact (integer multiple of the least subnormal); otherwise relative bound u, given no overflow. On the stated domain eta<=uM; products and intermediate sums have headroom below overflow (bound absolute contributions by their exact magnitudes plus rounding, each aggregate <2M). Doubling t is exact when finite. For gamma3=3u/(1-3u), a conservative real error <= gamma3*M+4(1+u)^2*eta <9uM; imaginary error <=uM+2eta<=3uM. Hence Euclidean error <=sqrt(90)uM <12uM. This proof is **conditional on the graph and runtime premises**, not proof that every admitted invocation reaches it. Exceptional values outside the domain rejected; cancellation and subnormal products allowed inside.

D: existing candidate computes u via Decimal.from_float(ulp(1))/Decimal(2) in ambient context (28 digits by default), potentially rounding down. Exact reproducer x=2^-60,y=z=0,b=1: existing converted bound 0x1.8000000000000p-50 is smaller than exact 12uM; minimal outward binary64 is 0x1.8000000000001p-50. This is an enclosure construction defect, not observed violation of the actual delta error. Fix reference using Fraction.from_float, exact domain and exact q=12uM; convert to float then nextafter toward +inf if Fraction(f)<q; verify prevfloat(f)<q<=f. Reject invalid inputs/intervals, do not use assertions as production guards. 7A ambient-context issue noted separately without changing its frozen disposition.

E1-E6: scalar tails and stride/count branches partially reconciled, no arithmetic in metadata pack/copy. Runtime-function unwind ranges are not full function boundaries. Import IAT slots resolved statically: FAD2BC8 VCRUNTIME140.dll memcpy; FAD4878 c10.dll SmallVectorBase<unsigned int>::grow_pod; FAD2E18 api-ms-win-crt-heap-l1-1-0.dll free. External behavior and live IAT integrity remain trusted/unproven. A4 security failure paths retain raw REVIEW_REQUIRED; normal execution assumption. No universal closure claim.

## 5. Gate 3I-F through P — accepted bounded checks

| Gate | Disposition | Qualification |
|---|---|---|
| F | Runtime metadata/10 registrations captured | Manual reconstruction, not actual cg.delta; no live MXCSR |
| G | 7/7 actual production evaluations, 4/4 domain rejects, exit 0 | External exact domain/enclosure reference, not integrated |
| H | Loaded DLL backing paths and hashes, reported source revision | Source-level consistency, not binary equivalence |
| I | 13 metadata checks, six internal APIs available | _get_dispatch_mode not invoked; enabled TorchFunction mechanism is not active override |
| J | Calling-thread MXCSR snapshot positive/negative/restore tests | 1FA0 -> 9FE0 with FTZ/DAZ -> 1FA0, mask FFC0 equals 1F80 required; same thread; no transient-state proof |
| K | Combined guard 22/22 (5 admissions,17 rejects) | Includes override-mode traps, FTZ/DAZ rejection; not production |
| L | Binding prototype 15/15 (7 evaluations,4 domain +4 checker rejects) | Pins, snapshot, immutable observation, exact squared residual; trusted runtime/stable initial read |
| M | Integration inventory | No direct tracked caller of backend candidate/budget found by git grep; not exhaustive indirect search |
| N | Downstream source review | Double delta evaluation found |
| O | Synthetic mismatch reproduced on Windows | 2 delta calls, reported -1, candidate 10j, valid True and zero bound; mock only, no spontaneous nondeterminism |
| P | In-memory single-evaluation correction passes | One baseline case, two injected controls, source hash unchanged, exit 0; no actual patch |

Gate L numerical observations independently checked from the seven hexadecimal records during consolidation (exact Fraction squared residual, exact M and minimal outward B):
| case | outward B (hex) | error / (uM), approximate |
|---|---|---|
| ordinary | 0x1.779999999999ap-48 | 0.0383386581469648567 |
| decimal_defect | 0x1.8000000000001p-50 | 6.7762635780344e-21 |
| lower_equal | 0x0.0000000000006p-1022 | 0 |
| upper_equal | 0x1.8000000000000p+972 | 0 |
| subnormal_b | 0x1.8000000000001p-49 | 1.0993325529478e-631 |
| subnormal_products | 0x0.0000000000007p-1022 | 0.0312499999997726262 |
| exact_cancellation | 0x1.b000000000000p-48 | 0 |

All seven satisfy error² <= (12uM)², prevfloat(B)<12uM<=B. The subnormal-b error is tiny but nonzero (b²); earlier displays rounded it to zero.

Reported original diagnostic report SHA-256 (not independently authenticated, original report files remain on operator machine):
| Gate | SHA-256 |
|---|---|
| F | e9b506242efc420e47049e4ced304788e82d92d1aad342a0a1121f377f5f54bd |
| G | 5fd4adfbfa4237e5ff44a7a3113298111cc85db151f7e04fe03b16d3b7288050 |
| J | 3c114f3dea76ae70c578f98cc7095cd6b807b7b22cac6790289bb5dfb43f645d |
| K | 6778a663f8613d5891b4c184115dbd27a8bf710f2132832306ccfa2659f91e44 |
| L | ca57b867e2124b1834c54c9e49049f40e204c3a666e21937b618f1d00f1d5ade |
| P | 0ec9bdd20d49a804db06fb6a51a5bfe370e71cc667d99f78ca4017269d28718d |

Actual uploaded cumulative P transcript independently hashed: 671b7fccbda04fc50819560b1a8b9c737c08c1eeebc736ec94a87ec0186fcc51.

## 6. Integration gap and reviewed correction (not applied)

Existing certified_ray.py propagate_ray_step_enclosure computes sqrt(cg.delta(pos_next64,b)) and separately calls cg.delta(...).item() for its interval check. Existing directed-rounding distance check rejects observed delta exceeding fl_delta_err; it checks the second evaluation, not necessarily the first. Gate O intentionally violates cross-call consistency with -100 then -1; root 10j while reported delta -1, valid True and eps_cand.hi=0.00. Actual backend nondeterminism not established; conditional Bridge 6 theorem unchanged.

Gate P in-memory shadow replaces only the two lines:
```python
delta64 = cg.delta(pos_next64, b)
w_cand64 = torch.sqrt(delta64).to(dtype=torch.complex128)
delta_hat_c = complex(delta64.item())
```
Baseline parity True, bad_then_good: one call, delta -100, root 10j, valid False, reason computed_delta_exceeds_fl_delta_budget. good_then_bad: one call, delta -1, root 1j, valid True. Production function identity and certified_ray.py hash unchanged. The fix still assumes stable input/output and ordinary nonmutating sqrt; enforce exact ordinary 0D CPU complex128 before consumers in future authorized patch. Bind endpoint/result bits to budget, don't recompute endpoint separately or silently treat a standalone bound as certificate. Frozen dataclass with mutable provenance dict is insufficient; all_certified currently checks status strings without validating numerical fields. Preserve conditional Bridge 6 API, and do not supply unresolved fields as zeros. Per-evaluation exact residual verification is a distinct fallback, not a universal backend proof.

## 7. Finite closure checklist

Stop exploratory integration gates. Remaining universal proof: establish every admitted scalar invocation reaches the graph (operator registration -> TensorIterator routing, fill/scalar conversion, dispatch keys and reachable helpers; all alternative or exceptional paths excluded or proved). State precise trusted runtime, input ownership and FP-state assumptions; pre/post MXCSR snapshots cannot exclude transient changes. Then separately authorize reviewable integration patch, exact domain/bound, evaluation binding, runtime fingerprints, immutable valid provenance, negative controls and existing regression suite. Do not promote before both proof and implementation are complete. Next work is consolidated dispatch proof, not another broad survey. 7C sqrt and 7D predictor/comparison remain separate.

```python
GATE3G_BOUNDED_SURVEY_ACCEPTED_TWO_STREAM = True
GATE3H_B_BOUNDED_ARITHMETIC_EVIDENCE_ACCEPTED_WITH_QUALIFICATIONS = True
GATE3H_C_BOUNDED_NONZERO_ARITHMETIC_EVIDENCE_ACCEPTED_WITH_QUALIFICATIONS = True
GATE3I_CONDITIONAL_MATHEMATICAL_BOUND = True
GATE3I_OUTWARD_ROUNDING_DEFECT_IDENTIFIED = True
GATE3I_EXACT_DOMAIN_REFERENCE_VERIFIED = True
GATE3I_OUTWARD_ENCLOSURE_REFERENCE_VERIFIED = True
GATE3I_LIVE_MXCSR_GUARD_VERIFIED = True
GATE3I_ADMISSION_PROTOTYPE_VERIFIED = True
GATE3I_EVALUATION_BINDING_PROTOTYPE_PASSED = True
GATE3I_N_DOWNSTREAM_REVIEW_COMPLETE = True
GATE3I_BINDING_GAP_REPRODUCED_ON_PINNED_RUNTIME = True
GATE3I_P_ACCEPTED = True
GATE3I_P_SHADOW_CORRECTION_VERIFIED = True
GATE3I_EXPLORATORY_INTEGRATION_REVIEW_COMPLETE = True
GATE3I_UPSTREAM_DISPATCH_CLOSURE = False
GATE3I_NATIVE_SCALAR_PATH_CLOSURE = False
GATE3I_RUNTIME_ATTESTATION_VERIFIED = False
GATE3I_ENCLOSURE_IMPLEMENTATION_VERIFIED = False
BRIDGE_LEMMA_7B = "UNRESOLVED"
BRIDGE_LEMMA_7C = "UNRESOLVED"
BRIDGE_LEMMA_7D = "UNRESOLVED"
all_certified = False
```

Evidence policy: public repo contains technical audit summary and review-only diagnostics, not unrestricted raw machine logs or untracked production candidates. Raw originals, failed runs and cumulative console transcripts should be retained in a separate private evidence archive; normalized published copies are not byte-identical originals. Missing locally retained artifacts must not be represented as uploaded. This consolidation is not a new Windows test run or formal certification.
