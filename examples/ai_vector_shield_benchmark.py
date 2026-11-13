import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "2"

import sys, time, torch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ai_vector_shield import AIVectorShield

def metrics(preds, labels):
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1

def run():
    print("=" * 96)
    print(" TWISTOR-VISION v0.9.0: ULTRA-HARD TIE-X STEALTH & OOD POSIONING BENCHMARK")
    print("=" * 96)
    print(f"{'Scale (N)':<10} | {'Attack Class':<32} | {'Latency (ms)':<12} | {'Throughput':<14} | {'Prec.':<6} | {'Rec.':<6} | {'F1':<6}")
    print("-" * 96)

    shield = AIVectorShield(dim=512, threshold=0.65)
    for n in [1000, 5000, 10000, 20000]:
        torch.manual_seed(42)
        # Clean embeddings with coherent CP33 phase geometry
        base_phase = torch.linspace(0, 3.14, 512).unsqueeze(0).repeat(n, 1)
        mags = torch.randn(n, 512) * 0.15 + 1.0
        clean = torch.polar(mags, base_phase + torch.randn(n, 512) * 0.08)

        # Test 1: 60% OOD Outlier Poison
        n_ood = int(n * 0.6)
        ood = torch.polar(torch.rand(n_ood, 512) * 0.4, torch.rand(n_ood, 512) * 6.28)
        data_1 = torch.cat([clean, ood], dim=0)
        lab_1 = torch.cat([torch.ones(n), torch.zeros(n_ood)])
        t0 = time.perf_counter()
        p1 = shield.verify(data_1)
        dt1 = (time.perf_counter() - t0) * 1000.0
        pr1, rc1, f11_s = metrics(p1, lab_1)
        print(f"{n:<10} | {'OOD Outliers (60% Contam.)':<32} | {dt1:<12.3f} | {data_1.shape[0]/(dt1/1000):<11.1f} Hz | {pr1:<6.3f} | {rc1:<.3f} | {f11_s:<.3f}")

        # Test 2: 40% Tier-X Manifold-Tangent Stealth (Same magnitude, adversarial phase twirl)
        n_st = int(n * 0.4)
        st_mags = torch.randn(n_st, 512) * 0.15 + 1.0
        st_phase = base_phase[:n_st] + torch.randn(n_st, 512) * 1.8
        stealth = torch.polar(st_mags, st_phase)
        data_2 = torch.cat([clean, stealth], dim=0)
        lab_2 = torch.cat([torch.ones(n), torch.zeros(n_st)])
        t0 = time.perf_counter()
        p2 = shield.verify(data_2)
        dt2 = (time.perf_counter() - t0) * 1000.0
        pr2, rc2, f12_s = metrics(p2, lab_2)
        print(f"{n:<10} | {'Tier-X Tangent Stealth (40%)':<32} | {dt2:<12.3f} | {data_2.shape[0]/(dt2/1000):<11.1f} Hz | {pr2:<6.3f} | {rc2:<.3f} | {f12_s:<.3f}")
    print("=" * 96)

if __name__ == "__main__":
    run()
