from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple

def pso_optimize_entropy_aprp(
    params: Dict[str, Any],
    n_particles: int = 80,
    n_iter: int = 200,
    seed: int = 1
) -> Tuple[np.ndarray, float, float, float]:
    """
    Sesuai judul:
    - Prediksi deret waktu -> menghasilkan hhat (harga acuan dari SRA)
    - Entropy pasar -> masuk fungsi optimasi (objective), bukan hanya tampil

    Objective (MINIMIZE):
      fit = -revenue + w_entropy * sum(entropy_i * rel_dev_i) + penalty

    dimana:
      revenue = sum(q_i * x_i)
      rel_dev_i = |x_i - hhat_i| / max(1, hhat_i)

    Constraints (penalty):
      - h_i <= x_i <= H_i
      - (1-e_i)*hhat_i <= x_i <= (1+e_i)*hhat_i
      - revenue*(1-r) >= (TC+Profit)

    Output:
      best_x, revenue, entropy_score, penalty
    """

    rng = np.random.default_rng(seed)

    n_var = int(params["n_var"])
    q = np.asarray(params["q"], dtype=float)

    h = np.asarray(params["h"], dtype=float)
    H = np.asarray(params["H"], dtype=float)

    TC = float(params["TC"])
    Profit = float(params["Profit"])
    biaya_total = TC + Profit

    r = float(params["r"])

    hhat = np.asarray(params["hhat"], dtype=float)
    e_pred = np.asarray(params["e_pred"], dtype=float)

    entropy = np.asarray(params.get("entropy", [0.0] * n_var), dtype=float)
    if entropy.shape[0] != n_var:
        entropy = np.zeros(n_var, dtype=float)

    w_entropy = float(params.get("w_entropy", 1.0))

    lo_pred = (1.0 - e_pred) * hhat
    hi_pred = (1.0 + e_pred) * hhat

    X = rng.uniform(h, H, size=(n_particles, n_var))
    V = rng.normal(0, 1, size=(n_particles, n_var)) * (H - h) * 0.05

    pbest = X.copy()
    pbest_fit = np.full(n_particles, np.inf)

    gbest = None
    gbest_fit = np.inf

    w = 0.72
    c1 = 1.49
    c2 = 1.49

    def fitness(x: np.ndarray) -> Tuple[float, float, float, float]:
        x = np.asarray(x, dtype=float)

        revenue = float(np.sum(q * x))

        rel_dev = np.abs(x - hhat) / np.maximum(1.0, hhat)
        entropy_score = float(np.sum(entropy * rel_dev))

        penalty = 0.0

        penalty += 1e6 * float(np.sum(np.maximum(0.0, h - x)))
        penalty += 1e6 * float(np.sum(np.maximum(0.0, x - H)))

        penalty += 1e6 * float(np.sum(np.maximum(0.0, lo_pred - x)))
        penalty += 1e6 * float(np.sum(np.maximum(0.0, x - hi_pred)))

        lhs = biaya_total - revenue * (1.0 - r)
        penalty += 1e6 * max(0.0, float(lhs))

        fit = (-revenue) + (w_entropy * entropy_score) + penalty
        return fit, revenue, entropy_score, penalty

    for i in range(n_particles):
        fit, _, _, _ = fitness(X[i])
        pbest_fit[i] = fit
        if fit < gbest_fit:
            gbest_fit = fit
            gbest = X[i].copy()

    for _ in range(n_iter):
        R1 = rng.random((n_particles, n_var))
        R2 = rng.random((n_particles, n_var))

        V = w * V + c1 * R1 * (pbest - X) + c2 * R2 * (gbest - X)
        X = X + V

        X = np.minimum(np.maximum(X, h), H)

        for i in range(n_particles):
            fit, _, _, _ = fitness(X[i])

            if fit < pbest_fit[i]:
                pbest_fit[i] = fit
                pbest[i] = X[i].copy()

            if fit < gbest_fit:
                gbest_fit = fit
                gbest = X[i].copy()

    final_fit, final_rev, final_ent, final_pen = fitness(gbest)
    return gbest, float(final_rev), float(final_ent), float(final_pen)
