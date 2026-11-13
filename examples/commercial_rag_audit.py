import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "2"

import sys, torch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ai_vector_shield import AIVectorShield

def roc_auc(labels, scores):
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    return (pos.unsqueeze(1) > neg.unsqueeze(0)).float().mean().item()

def metrics(preds, labels):
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1

def run_honest_audit():
    print("=" * 90)
    print(" HONEST COMMERCIAL RAG AUDIT: FULL-RANK SPECTRUM & UNSEEN DOMAIN SHIFT")
    print("=" * 90)

    torch.manual_seed(40)
    D = 768
    # Full-rank power-law spectrum (1/k^0.6) mimicking real transformer embeddings
    U, _ = torch.linalg.qr(torch.randn(D, D))
    sing_vals = 1.0 / (torch.arange(1, D + 1, dtype=torch.float32) ** 0.6)
    spectrum = U * sing_vals.unsqueeze(0)

    # 4 semantic clusters: 0-2 = Finance/Engineering (calibrated), 3 = Unseen HR/Legal
    centers = torch.matmul(torch.randn(4, D), spectrum.T) * 2.5

    def sample_clusters(cluster_ids, n_per_cluster):
        out = []
        for cid in cluster_ids:
            z = torch.matmul(torch.randn(n_per_cluster, D), spectrum.T) + centers[cid]
            out.append(z)
        return torch.cat(out, dim=0)

    cal_data = sample_clusters([0, 1, 2], 400)
    shield = AIVectorShield(dim=D, latent_rank=32, quantile=0.95)
    shield.fit(cal_data)

    # Test 1: Garbled / OOD Outliers (Finance/Eng vs isotropic noise)
    clean_in = sample_clusters([0, 1, 2], 200)
    ood = torch.randn(400, D) * 0.55
    x1 = torch.cat([clean_in, ood], dim=0)
    y1 = torch.cat([torch.ones(len(clean_in)), torch.zeros(len(ood))])
    p1 = shield.verify(x1)
    s1 = -shield._projective_residual(shield._to_complex_projective(x1))
    pr1, rc1, f11 = metrics(p1, y1)
    print(f"[TEST 1: OOD Garbled Injection]       ROC-AUC: {roc_auc(y1, s1):.3f} | F1: {f11:.3f} | Prec: {pr1:.3f} | Rec: {rc1:.3f}")

    # Test 2: Unseen Business Domain (HR/Legal clean docs - False Positive check)
    clean_unseen = sample_clusters([3], 200)
    p_unseen = shield.verify(clean_unseen)
    pass_rate = p_unseen.float().mean().item()
    print(f"[TEST 2: Unseen HR/Legal Clean Docs]   Acceptance Rate: {pass_rate*100:.1f}% (False Positive Blocks: {(1-pass_rate)*100:.1f}%)")

    # Test 3: Stealth PoisonedRAG (Clean doc + small adversarial payload epsilon=0.25)
    payload = torch.matmul(torch.randn(1, D), spectrum.T)
    poisoned = sample_clusters([0, 1, 2], 200) + 0.25 * payload
    x3 = torch.cat([clean_in, poisoned], dim=0)
    y3 = torch.cat([torch.ones(len(clean_in)), torch.zeros(len(poisoned))])
    p3 = shield.verify(x3)
    s3 = -shield._projective_residual(shield._to_complex_projective(x3))
    pr3, rc3, f13 = metrics(p3, y3)
    print(f"[TEST 3: Stealth PoisonedRAG (eps=0.25)] ROC-AUC: {roc_auc(y3, s3):.3f} | F1: {f13:.3f} | Prec: {pr3:.3f} | Rec: {rc3:.3f}")
    print("=" * 90)

if __name__ == "__main__":
    run_honest_audit()
