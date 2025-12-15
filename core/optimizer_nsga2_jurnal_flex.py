from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize


def nsga2_optimize_flex(
    params: Dict[str, Any],
    pop_size: int = 160,
    n_gen: int = 250,
    seed: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MOEOP (Multi-Objective Entropy Optimization Problem)
    Decision variables: harga tiap produk (x_i)

    Objectives (MINIMIZE):
    F1: biaya_total - revenue   (semakin kecil semakin baik; bisa negatif)
    F2: risiko (deviasi dari harga acuan + penalti pelanggaran)
    F3: entropy * deviasi       (pasar volatil => deviasi lebih "mahal")

    Constraints (<= 0):
    - x_i <= HG_i
    - monotonic (optional)
    - |mean(x)-h_bar| <= eps (optional)
    - |D-S| <= eps_q per produk (optional)
    """

    n_var = int(params["n_var"])
    q = np.asarray(params["q"], dtype=float)

    h = np.asarray(params["h"], dtype=float)
    H = np.asarray(params["H"], dtype=float)
    HG = np.asarray(params["HG"], dtype=float)

    TC = float(params["TC"])
    Profit = float(params["Profit"])
    biaya_total = TC + Profit

    r = float(params["r"])

    hhat = np.asarray(params["hhat"], dtype=float)
    ehat = np.asarray(params["ehat"], dtype=float)

    mono_pairs = list(params.get("mono_pairs", []))
    mono_delta = float(params.get("mono_delta", 0.0))

    h_bar = float(params.get("h_bar", np.mean(hhat)))
    eps = float(params.get("eps", 0.0))

    eps_q = float(params.get("eps_q", 0.0))
    aD = np.asarray(params.get("aD", [10000]*n_var), dtype=float)
    bD = np.asarray(params.get("bD", [1]*n_var), dtype=float)
    aS = np.asarray(params.get("aS", [1000]*n_var), dtype=float)
    bS = np.asarray(params.get("bS", [1]*n_var), dtype=float)

    entropy = np.asarray(params.get("entropy", [0.0]*n_var), dtype=float)
    if entropy.shape[0] != n_var:
        entropy = np.zeros(n_var, dtype=float)

    w_entropy = float(params.get("w_entropy", 1.0))


    n_con = n_var  
    if mono_pairs:
        n_con += len(mono_pairs)
    if eps > 0:
        n_con += 1
    if eps_q > 0:
        n_con += n_var

    class MOEOPProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(
                n_var=n_var,
                n_obj=3,
                n_constr=n_con,
                xl=h,
                xu=H
            )

        def _evaluate(self, x, out, *args, **kwargs):
            x = np.asarray(x, dtype=float)

            # ========== F1: Pendapatan score
            revenue = float(np.sum(q * x))
            f1 = float(biaya_total - revenue)

            # ========== F2: Risiko score
            rel_dev = np.abs(x - hhat) / np.maximum(1.0, hhat)
            over_tol = np.maximum(0.0, rel_dev - ehat)

            risk_dev = float(np.sum(rel_dev) + 10.0 * np.sum(over_tol))

            gov_violate = np.maximum(0.0, x - HG)
            risk_gov = float(np.sum(gov_violate))

            f2 = float(risk_dev + 5.0 * risk_gov + r * risk_dev)

            # ========== F3: Entropy score (explicit)
            f3 = float(w_entropy * np.sum(entropy * rel_dev))

            # ========== Constraints
            g = []

            # 1) x_i <= HG_i  -> x_i - HG_i <= 0
            g.extend(list(x - HG))

            # 2) monotonic (optional)
            for (i_lo, i_hi) in mono_pairs:
                g.append((x[i_lo] + mono_delta) - x[i_hi])

            # 3) target rata-rata (optional)
            if eps > 0:
                g.append(abs(float(np.mean(x)) - h_bar) - eps)

            # 4) keseimbangan pasar (optional)
            if eps_q > 0:
                D = aD - bD * x
                S = aS + bS * x
                gap = np.abs(D - S)
                g.extend(list(gap - eps_q))

            out["F"] = np.array([f1, f2, f3], dtype=float)
            out["G"] = np.array(g, dtype=float)

    problem = MOEOPProblem()
    algorithm = NSGA2(pop_size=int(pop_size))
    termination = get_termination("n_gen", int(n_gen))

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=int(seed),
        save_history=False,
        verbose=False
    )

    X = res.X
    F = res.F
    if X is None or F is None:
        return np.empty((0, n_var)), np.empty((0, 3)), np.empty((0, n_con))

    # compute CV
    CV = []
    for x in np.atleast_2d(X):
        out = {}
        problem._evaluate(x, out)
        g = out["G"]
        CV.append(np.maximum(0.0, g))
    CV = np.asarray(CV)

    return np.asarray(X), np.asarray(F), CV
