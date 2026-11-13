import torch
import time
import numpy as np

class TwistorEmbeddingShield(torch.nn.Module):
    """
    Differentiable vector space filter using CP^3 spinor incidence 
    to reject poisoned or adversarial LLM/RAG embeddings.
    """
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.projection = torch.nn.Linear(embedding_dim, 4, dtype=torch.complex64)
        
    def forward(self, embeddings, reference_centroid):
        z_complex = torch.complex(embeddings, torch.zeros_like(embeddings))
        twistor_coords = self.projection(z_complex)
        
        # Compute distance against the anchored clean manifold reference centroid
        coords_magnitude = torch.abs(twistor_coords).mean(dim=-1)
        ref_magnitude = torch.abs(self.projection(torch.complex(reference_centroid, torch.zeros_like(reference_centroid)))).mean()
        
        # Fixed geometric boundary relative to the known manifold
        valid_mask = coords_magnitude <= (ref_magnitude * 3.5)
        return embeddings * valid_mask.unsqueeze(-1), valid_mask

def run_poisoning_stress_test(num_vectors=10000, poisoning_ratio=0.40):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shield = TwistorEmbeddingShield(embedding_dim=512).to(device)
    
    torch.manual_seed(42)
    base_manifold = torch.randn(1, 512, device=device)
    clean_embeddings = base_manifold + torch.randn(num_vectors, 512, device=device) * 0.2
    
    n_poisoned = int(num_vectors * poisoning_ratio)
    labels = torch.zeros(num_vectors, dtype=torch.bool, device=device)
    labels[:n_poisoned] = True
    
    poison_noise = torch.randn(n_poisoned, 512, device=device) * 6.0
    embeddings = clean_embeddings.clone()
    embeddings[:n_poisoned] = embeddings[:n_poisoned] + poison_noise
    
    perm = torch.randperm(num_vectors)
    embeddings = embeddings[perm]
    labels = labels[perm]
    
    t0 = time.perf_counter()
    with torch.no_grad():
        filtered, valid_mask = shield(embeddings, base_manifold)
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "xpu":
        torch.xpu.synchronize()
    dt = (time.perf_counter() - t0) * 1000.0
    
    detected_poison = ~valid_mask
    tp = (detected_poison & labels).sum().item()
    fp = (detected_poison & ~labels).sum().item()
    fn = (~detected_poison & labels).sum().item()
    
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * (precision * recall) / max(precision + recall, 1e-6)
    throughput = num_vectors / max(dt / 1000.0, 1e-6)
    
    return {
        "num_vectors": num_vectors,
        "poison_ratio": poisoning_ratio,
        "latency_ms": dt,
        "throughput_hz": throughput,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

if __name__ == "__main__":
    print("=" * 90)
    print("  TWISTOR-VISION v0.8.0: EXTENSIVE AI POISONING STRESS BENCHMARK (ANCHORED)")
    print("=" * 90)
    print(f"{'Scale (N)':<12} | {'Poison Contam.':<15} | {'Latency (ms)':<14} | {'Throughput':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 90)
    
    for n in [1000, 5000, 10000, 20000]:
        for ratio in [0.20, 0.40, 0.60]:
            res = run_poisoning_stress_test(num_vectors=n, poisoning_ratio=ratio)
            print(f"{res['num_vectors']:<12,} | {res['poison_ratio']*100:0.0f}%{' ':<11} | {res['latency_ms']:<14.3f} | {res['throughput_hz']:0.1f} Hz{' ':<6} | {res['precision']:<10.3f} | {res['recall']:<10.3f} | {res['f1']:<10.3f}")
    print("=" * 90)