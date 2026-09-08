import numpy as np
import torch
import logging
from src.explainability import explain_shap_prpatch, explain_lime_instance
from src.evaluation import plot_shap_summary

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('smoke')

# Toy model: returns scalar per sample
class ToyModel(torch.nn.Module):
    def __init__(self, seq_len, n_features):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.linear = torch.nn.Linear(seq_len * n_features, 1)
    def forward(self, x):
        b = x.shape[0]
        y = x.reshape(b, -1)
        return self.linear(y)

seq_len = 8
n_features = 4
bg = 12

np.random.seed(0)
X_bg = np.random.randn(bg, seq_len, n_features)
inst = np.random.randn(seq_len, n_features)

model = ToyModel(seq_len, n_features)

# SHAP prpatch (fast mode)
try:
    shap_arr, expl = explain_shap_prpatch(
        model_or_predict_fn=lambda x: model(torch.tensor(x, dtype=torch.float32)).detach().cpu().numpy(),
        X_background=X_bg,
        X_instance=inst.reshape(1, seq_len, n_features),
        seq_len=seq_len,
        patch_size=2,
        target_name='occupied_beds_calculated',
        background_size=10,
        nsamples=50,
        device='cpu',
        feature_names=[f"ch{c}_lag{l}" for c in range(n_features) for l in range(seq_len)],
        out_basename='shap'
    )
    logger.info(f"SHAP computed: shape={getattr(shap_arr, 'shape', None)}")
except Exception as e:
    logger.exception(f"SHAP explain failed: {e}")

# LIME
try:
    exp, out = explain_lime_instance(
        predict_fn=lambda x: model(torch.tensor(x, dtype=torch.float32)).detach().cpu().numpy(),
        X_train=X_bg,
        instance=inst,
        n_patches=4,
        n_features=n_features,
        feature_names=[f"ch{c}_lag{l}" for c in range(n_features) for l in range(seq_len)],
        target_name='occupied_beds_calculated',
        num_features=8,
        save=True,
        out_basename='lime'
    )
    logger.info(f"LIME explained; out_plot={out}")
except Exception as e:
    logger.exception(f"LIME explain failed: {e}")

# SHAP summary via evaluation helper
try:
    plot_shap_summary(model, X_bg, inst.reshape(1, seq_len, n_features), [f"ch{c}_lag{l}" for c in range(n_features) for l in range(seq_len)], save_path='artifacts/plots/test_shap_summary.png', seq_len=seq_len)
    logger.info("plot_shap_summary finished")
except Exception as e:
    logger.exception(f"plot_shap_summary failed: {e}")

print('SMOKE TEST COMPLETE')
