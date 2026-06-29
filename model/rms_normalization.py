import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        x_np = np.array(x);
        rms = np.sqrt(np.dot(x_np, x_np)/len(x_np) + eps)
        x_np /= rms
        x = x_np.tolist()
        return [np.round(x_i*gamma_i, 4) for x_i, gamma_i in zip(x, gamma)]
