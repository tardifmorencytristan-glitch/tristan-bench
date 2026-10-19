"""Synthetic vectorized Pareto benchmark: three problems, ten policies, 289 candidates.

Requires NumPy. No HTTP, secrets, file writes, or external effects.
Pareto sets are exact only relative to the evaluated finite grid.
"""
import json
import numpy as np

REFERENCE = [
    [21, 21, 17, 14, 14, 21, 21, 21, 17, 21],
    [27, 27, 17, 18, 18, 27, 27, 27, 17, 25],
    [22, 26, 16, 18, 12, 22, 22, 26, 16, 22],
]
SHARED_REFERENCE = [100, 100, 49, 56, 54, 100, 100, 100, 49, 94]
TARGETS = np.array([
    [[-1, .75], [1, -.75]],
    [[-1.25, -1], [1.25, 1]],
    [[-.5, 1.25], [1.25, -.5]],
], dtype=float)
POLICIES = (
    "base_two", "base_three", "energy_budget_1", "x_ge_0", "x_le_0",
    "robust_axis_025", "robust_axis_050", "robust_axis_025_three",
    "robust_axis_050_budget_1", "energy_budget_2",
)

def nondominated(values, feasible):
    """Return feasible nondominated mask for a minimization problem."""
    non_worse = np.all(values[:, None] <= values[None, :] + 1e-10, axis=-1)
    strictly_better = np.any(values[:, None] < values[None, :] - 1e-10, axis=-1)
    dominates = non_worse & strictly_better & feasible[:, None] & feasible[None, :]
    return feasible & ~np.any(dominates, axis=0)

def evaluate():
    axis = np.linspace(-2, 2, 17)
    x = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    difference = x[None, :, None, :] - TARGETS[:, None, :, :]
    losses = np.sum(difference**2, axis=-1)
    energy = np.sum(x**2, axis=-1)
    energies = np.broadcast_to(energy, (3, len(x)))
    base = np.concatenate((losses, energies[..., None]), axis=-1)
    def robust(eps):
        d = losses + eps**2 + 2 * eps * np.max(np.abs(difference), axis=-1)
        return np.concatenate((d, energies[..., None]), axis=-1)
    source = [base] * 5 + [robust(.25), robust(.50), robust(.25),
                           robust(.50), base]
    scores = np.stack(source, axis=1)
    three = {1, 7}
    for policy in range(10):
        if policy not in three:
            scores[:, policy, :, 2] = 0
    masks = []
    for policy in range(10):
        ok = np.ones(len(x), dtype=bool)
        if policy in (2, 8):
            ok = energy <= 1
        elif policy == 9:
            ok = energy <= 2
        elif policy == 3:
            ok = x[:, 0] >= 0
        elif policy == 4:
            ok = x[:, 0] <= 0
        masks.append(ok)
    counts = [[int(nondominated(scores[p, f], masks[f]).sum())
               for f in range(10)] for p in range(3)]
    shared = []
    for f in range(10):
        # Reverse view: same policy simultaneously evaluated over all problems.
        stacked = scores[:, f, :, :2].transpose(1, 0, 2).reshape(len(x), 6)
        shared.append(int(nondominated(stacked, masks[f]).sum()))
    assert counts == REFERENCE, (counts, REFERENCE)
    assert shared == SHARED_REFERENCE, (shared, SHARED_REFERENCE)
    return {
        "tensor_shape": list(scores.shape), "problems": 3,
        "policies": list(POLICIES), "candidates": len(x),
        "front_counts": counts, "shared_front_counts": shared,
        "verified_against_reference": True,
    }

if __name__ == "__main__":
    print(json.dumps(evaluate(), sort_keys=True))
