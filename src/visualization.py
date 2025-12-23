import matplotlib.pyplot as plt
import numpy as np

def plot_projection(X_proj: np.ndarray, labels: list, title: str, save_path: str = None):
    # 2d scatter plot
    unique = list(set(labels))
    cmap = plt.get_cmap('tab10')
    
    plt.figure(figsize=(10, 8))
    
    for i, label in enumerate(unique):
        idx = [j for j, x in enumerate(labels) if x == label]
        plt.scatter(
            X_proj[idx, 0], X_proj[idx, 1], 
            color=cmap(i), label=label, alpha=0.7, edgecolor='k'
        )
    
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_correlation_circle(corr: np.ndarray, feats: list, save_path: str = None):
    # correlation circle viz
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # unit circle
    ax.add_artist(plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--'))
    
    for i in range(len(feats)):
        x, y = corr[i, 0], corr[i, 1]
        ax.arrow(0, 0, x, y, head_width=0.03, head_length=0.03, fc='r', ec='r')
        ax.text(x * 1.15, y * 1.15, feats[i], color='black', ha='center', va='center')
        
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.axhline(0, color='k', linewidth=0.5)
    plt.axvline(0, color='k', linewidth=0.5)
    plt.title("Correlation Circle")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()