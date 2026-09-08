import numpy as np
import torch
import logging
from src.explainability import explain_lime_instance

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('lime-smoke')

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
bg = 20
np.random.seed(0)
X_bg = np.random.randn(bg, seq_len, n_features)
inst = np.random.randn(seq_len, n_features)
model = ToyModel(seq_len, n_features)

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

print('LIME TEST COMPLETE')
