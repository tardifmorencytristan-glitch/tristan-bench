# R15 — Tensorized multi-front optimization (synthetic benchmark)

**Status:** reproducible finite-grid experiment; not a scientific claim of novelty, global optimization, autonomous agents, or a deployed service.

## Idea

One optimization problem can be examined through multiple **Pareto policies** (different objective sets, constraints, and perturbation models). Conversely, the **same policy** can be applied to several problems to find solutions that are nondominated jointly over all their objectives. These are different maps, not mathematical inverses.

- Parameter candidates: 17 × 17 grid in [-2, 2]² = **289 candidates**.
- Synthetic problems: **3**, each with two different target points.
- Pareto policies: **10** per problem.
- Candidate scores: NumPy tensor **[3, 10, 289, 3]**.
- Forward evaluation: **30** single-problem policy fronts.
- Reverse/cross-problem evaluation: **10** shared-parameter fronts with **6 objectives** (2 per problem).

Each target loss is a squared Euclidean distance. An optional third objective is parameter energy, `x²+y²`. Constraints include energy budgets and half-spaces. Robust objectives use the exact worst score among four axis-aligned perturbations of radius epsilon:

```text
robust_target_loss = squared_distance + epsilon²
                     + 2 * epsilon * max(abs(dx), abs(dy))
```

A front is the set of feasible candidates for which no other feasible candidate is at least as good on every active objective and strictly better on one. All objectives are minimized. For policies with only two active objectives, the third score is set to zero.

## Reproduction

Requires Python >= 3.11 and NumPy (not currently a core dependency of the repository).

```bash
python examples/multifront_tensor_r15.py
```

The script performs no network requests or disk writes. It checks its result against fixed reference counts and prints structured JSON.

### Reference nondominated-set cardinalities

Columns are policies in the order listed by the script.

| Problem | Base 2 | Base 3 | Budget 1 | x ≥ 0 | x ≤ 0 | Robust .25 | Robust .50 | Robust .25 + energy | Robust .50 + budget | Budget 2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 21 | 21 | 17 | 14 | 14 | 21 | 21 | 21 | 17 | 21 |
| B | 27 | 27 | 17 | 18 | 18 | 27 | 27 | 27 | 17 | 25 |
| C | 22 | 26 | 16 | 18 | 12 | 22 | 22 | 26 | 16 | 22 |

Shared-parameter fronts (same policy applied across all three problems), by policy:

```text
[100, 100, 49, 56, 54, 100, 100, 100, 49, 94]
```

## Verification and limits

- Generated using a vectorized NumPy implementation, then checked on the full 289-state grid with an independently structured NumPy dominance oracle.
- Independently reproduced with **integer arithmetic**: all grid coordinates and target coordinates are exact multiples of 1/4, so all benchmark objective values can be scaled by 16 to obtain integers. The independent integer oracle matched all **30 + 10** cardinalities.
- The standalone publishable script was executed successfully in three separate environments and matched all reference counts.
- Agreement on cardinalities alone does not prove equality of all Pareto membership masks. The comparison is scoped to the tested synthetic benchmark.
- These fronts are exact **only within the chosen finite grid**. No continuous optimum, runtime speedup, physical performance, validity of external models, or novelty claim is made.
- Some robust policies yield the same front cardinalities in this benchmark. A change in the objective formula does not guarantee a new nondominated set.
- No private research files, credentials, local system details, public ingress, external API calls, charges, or automated production publications are involved.

## Relationship to established tools

For larger-scale experiments, prefer assessing existing frameworks rather than replacing them: [pymoo](https://pymoo.org/algorithms/moo/nsga2.html) for multiobjective evolutionary optimization, [Optuna](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/002_multi_objective.html) for study orchestration, and [BoTorch](https://botorch.org/docs/v0.17.2/tutorials/multi_objective_bo) for parallel Bayesian optimization.

**Next gates before operational use:** reproduce full membership masks; add null/random and strong baseline comparisons; test constraint edge cases, noisy measurements, and incompatible objectives; evaluate privacy, authority, reversibility, and deployment costs; only then consider integration into existing orchestration. It is valid to select NO_ACTION.
