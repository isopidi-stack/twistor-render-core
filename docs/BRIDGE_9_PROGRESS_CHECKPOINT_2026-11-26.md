# Bridge 9 research recovery checkpoint — 2026-11-26, before JEDEC restart

## START HERE — owner and auditor handoff

Owner rejected BIOS flashing and external software installation to protect the sole home workstation. Auditor approved revised zero-flash Branch B; technical lead concurs. Only reversible memory configuration is authorized: record original BIOS 0402 settings, disable XMP, use SPD JEDEC DDR4-2400 at 1.20 V, auto timings, leave unrelated CPU/boot/storage/security settings unchanged. Do not load Optimized Defaults, use EZ Flash, update ME, clear TPM, or change CPU voltages/power limits. Check F10 change list; if unclear stop and ask. Back up critical worktrees and evidence off-machine. As of this checkpoint no post-change inventory has arrived: JEDEC is a target, NOT an observed state. No fresh campaign has run. Resume by reviewing BIOS photos and post-restart inventory, then run exactly one full R1 campaign. No retry-until-green.

This is documentation only, separate from frozen local implementation. Do not pull or merge this branch into frozen worktrees. Previous detailed checkpoint: [2026-11-25](BRIDGE_9_PROGRESS_CHECKPOINT_2026-11-25.md). GitHub main is not the latest reviewed executable implementation.

## Accepted scope and progress wall

| Scope | Status |
|---|---|
| 7B–7D | Conditional T1–T4 and a posteriori bounds; native a priori proof deferred |
| 8F | Owner accepted at 76e6267a63d6e3ccde873959edecf3f3ce27aa83 |
| 9A–F | Exact algebraic mathematics, inverse geometry, sharp degree-four bound, explicit sheet, rational obstruction — accepted |
| 9H | Standalone reference, simulation only — accepted; 14 tests, 7193 reported predicates |
| 9I-A | R2 owner accepted; qualified Q1–Q4 native affine trace deferral |
| 9I-B | R3 initialization owner accepted; 13 accepted / 19 rejected of 32, 248 assertions |
| 9I-C | R1 owner accepted; 7 verified transitions / 259 predicates |
| 9I-D | R1 owner accepted and closed; 70 verified / 2590 predicates / 57 continuity / 355 historical receipt checks, 6 tests |
| 9I-E.1–3 | Additive adaptive implementation, split nonpinned diagnostics and offline reconciliation complete; not promotion |
| 9I-E.4 | Original pinned campaign FAILED group 27; 8/9 groups; NOT_ACCEPTED |
| 9I-E.5 | OPEN; diagnostic composites only, no complete campaign reconciliation |
| Forensic 0 | Source audit auditor accepted |
| Forensic 1 | 200/200 rounds, 600/600 DLL comparisons; auditor accepted diagnostic only |
| Forensic 2 | Instrumented test 27 pass; independent reconciliation, Branch B no reproduction |
| 3A/3B | Hardware inventory complete; firmware remediation cancelled; memory-only revised protocol current |
| 9I-F, 9I-G, 9J | Not authorized; separate review/owner gates |
| Renderer/video pipelines, standalone builds, promotion, aggregate certification | Deferred |

## E R1 implementation and ledger

New source only twistor/certified_adaptive_session_9.py and tests/test_bridge9_adaptive_review.py plus separate review infrastructure. Existing accepted code unchanged. Single operation lease and shared Budget identity, Engine-local caches only; original algebraic anchor and complete historical replay. No persistent history cache, resource-policy increase or algorithm optimization. Exact target/remainder and lattice checks, robust clearance, 8F representability and shadow allocation. Strict retry strings ray_endpoint_drift_exceeds_budget and dual_certificates_not_both_satisfied with full 7BCD_REJECT prefix; injected recovery not genuine native endpoint recovery. Final observed publication is not verified arrival if postpublication reconciliation fails; poison. Seven positives (four quartics b ±1 sheets ±1, one exterior and two interior quadratics), both irrational crossings; genuine noninjected subdivision at s=1/16. Resource caps 74019 authenticated preflight controlled exhaustion, 80000 historical replay poison, 65000 post-evaluator poison; preserve exact caps. Bounded termination, not guaranteed arrival. Local split diagnostics real numerical but nonpinned; combined local run timed out, not a combined pass.

Original Windows run 2026-11-26 14:59 local: groups 19–26 pass, group 27 second evaluator returned INTEGRITY_ABORT:7BCD_REJECT:MODULE_HASH:torch_cpu.dll. No second-step publication; last verified s=1/8, target 1/4, poisoned. Original archive pre/post six payloads each reconciled, 697 files / 26 pins / package unchanged, exit 2. Groups 19–26 alone 99 verified, 3663 predicates, 83 continuity; including failed group 27 first step 100/3700/83. Isolated test 27 then passed two transitions / 74 predicates / one binding, exact s=1/4, remaining 0, resource [30728,3944,501], selectors [1,1]. Composite 99+2=101 / 3737 / 84 is DIAGNOSTIC ONLY, no single campaign claim. Keep failed archive forever.

Stage 0: backend_certification.py digest at 438–440 uses context-managed hashlib.file_digest, actual comparison 470–471, no ordinary I/O exceptions mapped to MODULE_HASH. GetModuleHandleW/GetModuleFileNameW explicit signatures, 32768 wchar buffer, use_last_error=True; distinct lookup/path rejection; no direct harness contamination identified. Five internal call sites: certified_ray_7bcd.py 93, backend_certification.py 527 and 548, certified_ray_7bcd.py 203 and 234. Original actual wrong digest/call site not exported, cause unresolved.
Stage 1: 200 original verification calls, 600 DLL comparisons, 36.42s, handles 196 constant, commit 966594560 bytes constant, hot cache/no interleaved native compute; 4 selected event queries empty. Stage 2: 38 calls (28 session and each five internal twice), 114 comparisons, 2 verified transitions, 74 predicates, one continuity; no mismatch, entry/exit MXCSR 1fa0/1fa2; pre/post archive hashes matched; trace original byte counts unavailable for ALL 114, double-hash not triggered. Stage 2 composite diagnostic unchanged verifier passes. No fresh 697-file envelope for isolated or Stage 2. No causal attribution. Earlier 24 successful standalone DLL checks and empty selected event logs also only nonreproduction. The original event is permanently an unresolved transient authentication failure.

## Host and zero-flash limits

Intel i7-13700K 16 cores/24 logical; ASUS ROG STRIX B760-G GAMING WIFI D4 BIOS 0402 dated 2022-11-04; microcode reported 0x10E on all 24; 32 GiB non-ECC Kingston KF432C16BBK2/32, reported 3200 MT/s/1200 mV before change (actual XMP and voltage not established). JEDEC target 2400/1200, timings Auto. Firmware predates Intel voltage mitigations; owner defers flash, ME update and external tools because sole machine and irreversible risk. Memory-only change reduces possible XMP instability, does not establish hardware reliability or repair silicon. No WHEA matches in seven-day inventory. Four SATA drives: two Kingston A400 960 GB SSDs (C, Q), Samsung 500 GB D, Seagate 2 TB E. All SMART PredictFailure false, HDD read uncorrected zero; SSD read/write error fields unavailable, not zero; C max read latency 7285 ms no timestamp/cause. C/Q/E fully decrypted, D omitted; TPM ready; no clear/reset. No direct storage-driver events in prior returned events (IDs provider-specific). Windows 11 Home build 26200; preserve Defender/VBS/HVCI. ME interface driver 2406.5.5.0 and WMI provider 2408.5.4.0 signed; actual firmware ME version unknown. Full ATA SMART and external stress tests deferred, not passed. Original evidence must remain visible; no guarantee of universally stable hardware.

## Exact resume

1. After restart provide original BIOS photos/settings and post-change WMI both DIMMs, BIOS/microcode, available storage counters, Python version, R1 SHA. Confirm target JEDEC actually reached, unrelated settings not changed; if not, resolve without flash. Record unavailable fields. Preserve local/external backup.
2. Return to normal non-elevated PowerShell. Run once in a fresh timestamped Downloads directory, exact R1 ZIP SHA below. The unchanged runner performs fresh pre/post preservation and source/runtime authentication. Capture full console stdout/stderr; retain result regardless of success. Do not modify scripts, use old extracted folders, or change security policies.
3. Hard stop on any unexpected authentication/corruption failure: preserve archive and logs, no rerun. Built-in memory diagnostic or reversible isolation only with separate reviewed plan. No causal speculation. If runner/observer tooling fails classify separately, not numerical pass.
4. On clean run independently rehash R1 + evidence, ZIP CRC, manifest 104 + manifest itself (105 snapshot files), nested pre/post six payloads each, compare frozen baseline, run unchanged verify_stage_e_evidence.py according to packaged interface; verify full group records against EXPECTED_STAGE_E.json, mathematical predicates and continuity, auditor watchlist (lease, D0 authority, clearance contradiction, scheduling negatives, strict retry, genuine subdivision, shared budget and final poison). Require actual 9/9 groups, 101 verified / 3737 predicates / 84 continuity, 697/697, 26/26, no unexpected abort, exit 0 and 9I_E_OFFLINE_RECONCILIATION_PASS. Expected deliberate integrity aborts=13, observed unreconciled publication=1, 17 fault controls; do not misrepresent those as unexpected. Counts: 38 init/published, 118 session attempts, 106 real evaluator entries, 103 raw valid, 102 observed publications, 101 verified, 129 considered, 11 preflight, 25 controlled rejects, 101 pinned / 0 nonpinned. 53 operations. Check all seven positives. Auditor review and explicit owner closure still required, separately authorize F. No guarantee of future or universal hardware reliability.

Pinned command (run only AFTER baseline reviewed):
```powershell
& {
  $ErrorActionPreference = "Stop"
  Set-StrictMode -Version Latest
  $D = "$env:USERPROFILE\Downloads"
  $Py = "$env:USERPROFILE\twistor-benchmark\.venv-bench\Scripts\python.exe"
  $Zip = Join-Path $D "bridge9i_e_adaptive_review_R1.zip"
  $Expected = "c14fda5a52c8ff7a52effa8293d7d5b6a952669bc1d8ac151be5a4d255b336f9"
  if ((Get-FileHash -LiteralPath $Zip -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Expected) { throw "R1_HASH_MISMATCH" }
  $id = [Security.Principal.WindowsIdentity]::GetCurrent()
  $p = New-Object Security.Principal.WindowsPrincipal($id)
  if ($p.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw "NON_ELEVATED_REQUIRED" }
  $Stamp = Get-Date -Format "yyyyMMdd_HHmmss_ffffff"
  $Dest = Join-Path $D "bridge9i_e_jedec_review_$Stamp"
  Expand-Archive -LiteralPath $Zip -DestinationPath $Dest
  $Log = Join-Path $D "bridge9i_e_jedec_console_$Stamp.txt"
  Write-Host "CAMPAIGN_START = $(Get-Date -Format o)"
  $ErrorActionPreference = "Continue"
  & $Py -I -B (Join-Path $Dest "run_bridge9i_e_windows_review.py") 2>&1 | Tee-Object -FilePath $Log
  $RC = $LASTEXITCODE
  Write-Host "CAMPAIGN_END = $(Get-Date -Format o)"
  Write-Host "9I_E_REVIEW_EXIT_CODE = $RC"
  Write-Host "CONSOLE_LOG = $Log"
}
```
Wrapper is a proposed recovery instruction, not executed at checkpoint.

## Artifact registry (SHA256 rechecked against available bytes on 2026-11-26)

| Artifact | SHA-256 |
|---|---|
| bridge9i_a_final_review_candidate_R2.zip | 3fab2ca87002258a630623c6e961658fbd32da6c6e26376471e0d07a9a839711 |
| bridge9i_b_initialization_review_R3.zip | 26166c55542df082c1ac43261b4e65b7e78865e3feb14f2c3ce54a8f0d6567f1 |
| bridge9i_b_evidence_20261125_164626_939675.zip | 10c646046b4322cc6a259967ea910de3e8e014e3d25313c597b1c39375d5cfd3 |
| bridge9i_c_single_step_review_R1.zip | b455e76fa452977095eaa32045227208413477e0a8cc3adfb46c2adc056b1adf |
| bridge9i_c_evidence_20261125_170636_224304.zip | 2390cac6cc08414486a02bd65ceb5589b3541c857a5840f53e01828be35dd1fe |
| bridge9i_d_recursive_review_R1.zip | b920ed6a0f1f6ffa99bf64c9905a57f23d9f46686ccf52e71f906de8683c4701 |
| bridge9i_d_evidence_20261126_132649_834684.zip | b961df54b0649c06006f3f78a6aa6b9cb6c18dbf3c40080f1195db08bb27146a |
| bridge9i_e_adaptive_review_R1.zip | c14fda5a52c8ff7a52effa8293d7d5b6a952669bc1d8ac151be5a4d255b336f9 |
| bridge9i_e_evidence_20261126_145928_520056.zip (FAILED) | 339afdb19fdde3288e8027acb43ba2355c2773f75b49911917543b8dd53abb23 |
| bridge9i_e_evidence_isolated27_20261126_153006_902566.zip | cbf414cf35ea6317a9e132b0dbe6cf57f3471107edbdddee88cf3f3bbadfe454 |
| bridge9i_e_stage1_evidence_20261126_155903_907035.zip | 08446bbde9a267c35123fe64efe82926abbf53935be53402b5f1588da721c994 |
| bridge9i_e_evidence_stage2_20261126_160831_027620.zip | aec91510377f06265a2872026a8b1058d6a652a107731605ddc80b67b9ee0f2e |
| bridge9i_e_stage1_dll_diagnostic.py | 0b781c4d7e5b59e7aa2eb958aa8e556afde5174aaf28498f1854528f0939ba9f |
| bridge9i_e_stage2_forensic_diagnostic.py | dac0b021d3d7cb6b1e7169c3c8edb88de1ed79a52680eb0925fa47ce6f8166fd |
| 9id_offline_verification.log | b7718c31c7eeda1f8d265a915baa509c3bb7717b992c21de3f8389fb9e21c2ea |
| 9ie_partial_reconciliation.log | c02e22b78729f5b7eab50c05eaa2a8f174f676db3f77e8bbb1c1584237fa432b |
| 9ie_isolated27_composite_reconciliation.log | 5dcee54cc25af3f68c2ff771362d957e74dcafe14679da0a241f9bbace92334f |
| 9ie_stage2_independent_reconciliation.log | 6e9a1c554a68d753017abe33904abf155714f8f9c63b289ba8cfd2b365066007 |

Raw host transcripts and ZIPs retained separately, not published to this documentation branch. This registry authenticates available artifacts, not a new live scan. Original 9H ZIP 26d2ac647c5e334e87cd26db60d0ac40e64451daf229aa52eb2bb2ef5e1b2ee9, launcher 59c8320b7800f7ae8f794f2b83cf79641cb2f9abca748d62a86b9b52c8ca3e16 (prior recorded identities). Preservation snapshot 21b17b6d75d317adf3a3104171205a15894774c74ff320be5574b0eacec38e90. Six worktrees counts 103+104+111+118+127+134=697, 26 inherited pins. Python 3.13.1, PyTorch 2.15.0+cpu revision d0f2af8b66b20da4baa50644c0cd3f887a2eae2e, AVX2, MXCSR control mask 0xffffffc0 == 0x1f80; DLL pins torch_cpu b8ec43a4ff24435c93a8a7710f9cf3fe1f246d923abbd7549064c8ba26afc642, torch_python 9865b16e791fc6f91fe92b38a930004a9fedaa60b3d70645b872f5d640574f29, c10 60cae8210a16da67910754008ff62266cf529363fdcd032cd50a6b949dc4ea21.

## Locks

GATE_9I_E=NOT_ACCEPTED; GATE_9I_E_5=NOT_ACCEPTED; ROOT_CAUSE_ESTABLISHED=False; BIOS_FLASH_AUTHORIZED=False; EXTERNAL_SOFTWARE_INSTALL_AUTHORIZED=False; JEDEC_POST_CHANGE_VERIFIED=False; _7B_INTEGRATION_ACCEPTED=False; all_certified=False. GATE3I_UPSTREAM_DISPATCH_CLOSURE=False; GATE3I_NATIVE_SCALAR_PATH_CLOSURE=False; GUARANTEED_ARRIVAL_ESTABLISHED=False; FULL_DISCOVERY_ALL_GREEN=False; GATE8F_GENUINE_ENDPOINT_BUDGET_RECOVERY_VERIFIED=False; GATE7C_NATIVE_APRIORI_PROOF_DEFERRED=True. T1–T4 conditional and Q1–Q4 qualified deferral remain. No accepted executable changes, pin replacement, integration, promotion, merge, or downstream authorization. Only this documentation publication authorized. 
