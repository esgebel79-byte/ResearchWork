"""
Visualization utilities for PR-Patch model internals: patch attention, forecast trajectories.
Ensures outputs are compatible with the real patch-based and multi-step architectures.
"""
from __future__ import annotations

from typing import Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def extract_attention_weights(model: torch.nn.Module, x: torch.Tensor, ews: Optional[torch.Tensor] = None, device: str = 'cpu') -> Optional[np.ndarray]:
    """Extract attention weights from the model's cross-attention layer during inference.
    
    FIX #2: Properly extract stored attention weights from model.last_attn_weights after forward pass.
    Shape returned: (batch, n_head, n_query_tokens, n_key_tokens)
    """
    if not hasattr(model, 'cross_attn'):
        return None

    model = model.to(device)
    model.eval()
    x = x.to(device)
    if ews is not None:
        ews = ews.to(device)

    try:
        with torch.no_grad():
            # Forward pass: model stores attention weights in self.last_attn_weights
            _ = model(x, ews=ews)
            
            # Retrieve stored attention weights (FIX #2)
            if hasattr(model, 'last_attn_weights') and model.last_attn_weights is not None:
                attn_weights = model.last_attn_weights
                if torch.is_tensor(attn_weights):
                    return attn_weights.detach().cpu().numpy()
    except Exception as e:
        print(f"Warning: Could not extract attention weights: {e}")
        return None
    return None


def visualize_patch_attention(model: torch.nn.Module, x: torch.Tensor, ews: Optional[torch.Tensor] = None,
                              save_path: str = "artifacts/plots/patch_attention.png", device: str = 'cpu') -> None:
    """Visualize the patch-level cross-attention matrix.

    Creates a heatmap of attention weights from component patches to EWS patches.
    """
    attn_weights = extract_attention_weights(model, x, ews=ews, device=device)
    if attn_weights is None or attn_weights.size == 0:
        print("No attention weights extracted.")
        return

    batch_size = attn_weights.shape[0]
    n_query = attn_weights.shape[1]
    if n_query == 0:
        print("Attention matrix has zero query dimension.")
        return

    mean_attn = attn_weights.mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mean_attn, cmap='viridis', aspect='auto')

    n_patches = getattr(model, 'n_patches', 8)
    n_inputs = getattr(model, 'n_inputs', 3)
    patch_labels = []
    for i_comp in range(n_inputs):
        for i_patch in range(n_patches):
            patch_labels.append(f"C{i_comp}_P{i_patch}")

    ews_labels = [f"EWS_P{i}" for i in range(n_patches)]

    ax.set_xticks(range(len(ews_labels)))
    ax.set_yticks(range(min(len(patch_labels), n_query)))
    ax.set_xticklabels(ews_labels, rotation=45, ha='right')
    ax.set_yticklabels(patch_labels[:min(len(patch_labels), n_query)])
    ax.set_xlabel("EWS Patch Tokens (Key/Value)")
    ax.set_ylabel("Component Patch Tokens (Query)")
    ax.set_title("Cross-Attention: Component Patches → EWS Patches")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Attention Weight")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Patch attention heatmap saved to {save_path}")


def compare_attention_focus(model: torch.nn.Module, x_list: list, ews: Optional[torch.Tensor] = None,
                            save_path: str = "artifacts/plots/attention_focus_comparison.png", 
                            device: str = 'cpu') -> dict:
    """Compare attention focus across different input samples.
    
    FIX #2: Validate that attention weights are actually DATA-DEPENDENT:
    outputs for x1 != x2 should have DIFFERENT attention patterns.
    
    Args:
        model: PRPatchModel instance
        x_list: list of (batch, seq_len, n_inputs) input tensors to compare
        ews: optional EWS tensor
        save_path: where to save comparison heatmap
        device: 'cpu' or 'cuda'
    
    Returns:
        dict with attention focus statistics
    """
    attn_list = []
    labels = []
    
    for i, x in enumerate(x_list):
        attn = extract_attention_weights(model, x, ews=ews, device=device)
        if attn is not None:
            attn_list.append(attn)
            labels.append(f"Sample {i}")
    
    if len(attn_list) < 2:
        print("Not enough attention weight arrays for comparison.")
        return {}
    
    # Attention shape is typically (batch, n_query, n_key)
    # Compare attention focus: compute differences between attention patterns
    attn_diffs = []
    for i in range(len(attn_list) - 1):
        diff = np.abs(attn_list[i] - attn_list[i+1]).mean()
        attn_diffs.append(diff)
    
    mean_diff = np.mean(attn_diffs)
    max_diff = np.max(attn_diffs)
    
    # Create visualization
    fig, axes = plt.subplots(1, len(attn_list), figsize=(5*len(attn_list), 4))
    if len(attn_list) == 1:
        axes = [axes]
    
    for idx, attn in enumerate(attn_list):
        # Average over batch dimension
        mean_attn = attn.mean(axis=0) if attn.ndim >= 3 else attn
        
        # Ensure we have a 2D array for imshow
        if mean_attn.ndim == 1:
            mean_attn = mean_attn.reshape(-1, 1)
        
        ax = axes[idx]
        im = ax.imshow(mean_attn, cmap='viridis', aspect='auto')
        ax.set_title(f"{labels[idx]}\n(Mean Attn)")
        ax.set_xlabel("Key Tokens")
        ax.set_ylabel("Query Tokens")
        plt.colorbar(im, ax=ax)
    
    fig.suptitle(f"Attention Focus Comparison (Mean Diff = {mean_diff:.4f}, Max Diff = {max_diff:.4f})")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"Attention comparison saved to {save_path}")
    print(f"  Mean attention difference: {mean_diff:.6f}")
    print(f"  Max attention difference: {max_diff:.6f}")
    
    return {
        'mean_attn_diff': mean_diff,
        'max_attn_diff': max_diff,
        'num_samples': len(attn_list)
    }



def visualize_forecast_trajectory(y_true: np.ndarray, y_pred: np.ndarray, horizon: int = 14,
                                  save_path: str = "artifacts/plots/forecast_trajectory.png") -> None:
    """Visualize observed vs predicted future trajectory over the forecast horizon.

    y_true: (n_samples, horizon, n_components) or (n_samples, horizon)
    y_pred: Same shape as y_true but model predictions.
    """
    if y_true.ndim == 2 and y_true.shape[1] == horizon:
        y_true_plot = y_true[0, :]
        y_pred_plot = y_pred[0, :]
    elif y_true.ndim == 3 and y_true.shape[1] == horizon:
        y_true_plot = y_true[0, :, 1] if y_true.shape[2] >= 2 else y_true[0, :, 0]
        y_pred_plot = y_pred[0, :, 1] if y_pred.shape[2] >= 2 else y_pred[0, :, 0]
    else:
        print(f"Unexpected shape: y_true {y_true.shape}, y_pred {y_pred.shape}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    time_steps = np.arange(len(y_true_plot))

    ax.plot(time_steps, y_true_plot, 'o-', label='Observed', color='#1f77b4', linewidth=2, markersize=6)
    ax.plot(time_steps, y_pred_plot, 's--', label='Predicted', color='#ff7f0e', linewidth=2, markersize=5)
    ax.fill_between(time_steps, y_true_plot, y_pred_plot, alpha=0.1, color='gray')

    ax.set_xlabel("Forecast Horizon (Days)")
    ax.set_ylabel("Target Value (I Component)")
    ax.set_title(f"H={horizon}-Step Forecast Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Forecast trajectory saved to {save_path}")


def visualize_patch_encoding_breakdown(model: torch.nn.Module, x: torch.Tensor,
                                       save_path: str = "artifacts/plots/patch_breakdown.png") -> None:
    """Show how each input component is encoded into patches.

    Visualizes the patch encoder output before attention fusion.
    """
    if not hasattr(model, 'patch_enc'):
        print("Model does not have patch_enc attribute.")
        return

    b, seq_len, c = x.shape
    n_patches = model.n_patches
    patch_size = model.patch_size
    n_inputs = model.n_inputs

    fig, axes = plt.subplots(n_inputs, 1, figsize=(12, 3 * n_inputs))
    if n_inputs == 1:
        axes = [axes]

    for i_comp in range(n_inputs):
        comp = x[:, :, i_comp].cpu().numpy() if torch.is_tensor(x) else x[:, :, i_comp]
        comp_sample = comp[0, :]

        ax = axes[i_comp]
        ax.plot(comp_sample, marker='o', linewidth=2, label='Input signal')

        colors = plt.cm.viridis(np.linspace(0, 1, n_patches))
        for p_idx in range(n_patches):
            start = p_idx * patch_size
            end = (p_idx + 1) * patch_size
            ax.axvspan(start, end, alpha=0.1, color=colors[p_idx])
            ax.text(start + patch_size/2, ax.get_ylim()[1] * 0.95, f"P{p_idx}", 
                   ha='center', fontsize=9, fontweight='bold')

        ax.set_ylabel(f"Component {i_comp}")
        ax.set_title(f"Patch Breakdown: Component {i_comp} (n_patches={n_patches}, patch_size={patch_size})")
        ax.grid(True, alpha=0.2)
        ax.legend()

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Patch breakdown saved to {save_path}")


def plot_lambda_sweep_results(df: dict | None = None, save_dir: str = "artifacts/plots") -> None:
    """Create comprehensive lambda sensitivity plots from the sweep results.

    If df is None, tries to load from the default metrics directory.
    """
    try:
        if df is None:
            metrics_path = Path("artifacts/metrics/lambda_sensitivity.csv")
            if metrics_path.exists():
                import pandas as pd
                df = pd.read_csv(metrics_path)
            else:
                print(f"Lambda sensitivity CSV not found at {metrics_path}. Skipping plot.")
                return
        else:
            import pandas as pd
            if isinstance(df, dict):
                df = pd.DataFrame(df)
    except Exception as e:
        print(f"Error loading lambda sensitivity data: {e}")
        return

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    try:
        lambdas = df['lambda'].to_numpy()
        mse_bif = df['mse_bifurcation'].to_numpy()
        pce_bif = df['pce_bifurcation'].to_numpy()

        ax = axes[0]
        mask_valid = ~(np.isnan(mse_bif) | np.isnan(pce_bif))
        ax.scatter(mse_bif[mask_valid], pce_bif[mask_valid], s=100, alpha=0.6, edgecolors='k')
        for i, lam in enumerate(lambdas[mask_valid]):
            ax.annotate(f"λ={lam:.3f}", (mse_bif[mask_valid][i], pce_bif[mask_valid][i]),
                       fontsize=9, xytext=(5, 5), textcoords='offset points')
        ax.set_xlabel("MSE at Bifurcation")
        ax.set_ylabel("PCE at Bifurcation")
        ax.set_title("MSE–PCE Trade-off")
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(lambdas, mse_bif, 'o-', label='MSE', linewidth=2, markersize=6)
        ax.set_xlabel("Lambda (Physics Regularization Strength)")
        ax.set_ylabel("MSE (left)")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(lambdas, pce_bif, 's-', color='red', label='PCE', linewidth=2, markersize=6)
        ax2.set_ylabel("PCE (right)", color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.legend(loc='upper right')

        ax.set_title("Lambda Sensitivity: Both Metrics")

    except Exception as e:
        print(f"Error plotting lambda sweep: {e}")

    fig.tight_layout()
    save_path = Path(save_dir) / "lambda_sweep_combined.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Lambda sweep plots saved to {save_path}")


if __name__ == '__main__':
    print("visualization.py module loaded. Use functions for PR-Patch model visualization.")
