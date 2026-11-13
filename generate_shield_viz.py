import numpy as np
import matplotlib.pyplot as plt

def generate_manifold_visualization():
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, 2 * np.pi, 50)
    U, V = np.meshgrid(u, v)
    X = (2 + 0.5 * np.cos(V)) * np.cos(U)
    Y = (2 + 0.5 * np.cos(V)) * np.sin(U)
    Z = 0.5 * np.sin(V) + 0.2 * np.sin(2*U)
    ax.plot_surface(X, Y, Z, color='cyan', alpha=0.15, edgecolor='deepskyblue', linewidth=0.3, antialiased=True)
    
    np.random.seed(42)
    n_clean = 300
    r_clean = 2.0 + np.random.normal(0, 0.08, n_clean)
    theta_clean = np.random.uniform(0, 2 * np.pi, n_clean)
    phi_clean = np.random.normal(0, 0.1, n_clean)
    ax.scatter(r_clean * np.cos(theta_clean), r_clean * np.sin(theta_clean), phi_clean, c='cyan', s=20, alpha=0.8, label='Clean Inliers (Valid RAG)')
    
    n_stealth = 80
    r_stealth = 2.0 + np.random.normal(0, 0.02, n_stealth)
    theta_stealth = np.random.uniform(0, 2 * np.pi, n_stealth)
    ax.scatter(r_stealth * np.cos(theta_stealth), r_stealth * np.sin(theta_stealth), np.random.normal(0.45, 0.02, n_stealth), c='#FFAA00', s=25, marker='s', alpha=0.9, label='Tier-X Manifold-Tangent Stealth')

    xp = np.random.uniform(-4, 4, 60)
    yp = np.random.uniform(-4, 4, 60)
    zp = np.random.uniform(-2, 2, 60)
    mask = (xp**2 + yp**2 > 3.5) | (np.abs(zp) > 0.8)
    ax.scatter(xp[mask], yp[mask], zp[mask], c='#FF3366', s=30, marker='^', alpha=0.9, label='OOD Adversarial Outliers')
    
    xx, yy = np.meshgrid(np.linspace(-3.5, 3.5, 10), np.linspace(-3.5, 3.5, 10))
    ax.plot_surface(xx, yy, np.zeros_like(xx) + 0.4, color='magenta', alpha=0.1, edgecolor='none')

    ax.set_title(r"Twistor-Vision Core: $\\mathbb{CP}^3$ Tier-X Manifold Shield", fontsize=14, color='white', pad=20, fontweight='bold')
    ax.axis('off')
    ax.view_init(elev=30, azim=45)
    leg = ax.legend(loc='upper left', frameon=True, facecolor='#111111', edgecolor='#333333')
    for text in leg.get_texts():
        text.set_color("white")
    plt.tight_layout()
    plt.savefig("assets/cp3_manifold_shield.png", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    print("Successfully generated assets/cp3_manifold_shield.png!")

if __name__ == "__main__":
    generate_manifold_visualization()
