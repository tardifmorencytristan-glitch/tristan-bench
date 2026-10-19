# R16 — Full Pareto membership verification and asymmetric controls

**Status:** exact finite-grid synthetic test, executed successfully on three locally authorized computers on October 19, 2026. Complements [R15](multifront_tensor_r15.md); does not claim global or physical optimality.

## Motivation and corrections to R15

R15 checked front **cardinalities**, which alone do not prove that two solvers returned the same members. R15's symmetric objectives also led to a common minimax candidate `(0,0)` under all 10 policies.

R16 adds:

1. Full nondominated-membership comparisons for **80** masks: 3 problems × 10 front policies + 10 common six-objective fronts in each of two worlds.
2. A second algorithm using explicit Python integer-tuple comparisons, independent from NumPy broadcasting.
3. A second asymmetric synthetic world with altered targets to force differences among policies.
4. A negative control rejecting the hypothesis that every feasible candidate is Pareto-optimal.
5. SHA-256 of all 40 masks per world (289 indicator bytes per mask in fixed order).

## Reproduce

Requires Python >=3.11 and NumPy:

```bash
python examples/validate_multifront_r16.py
```

The script is offline, deterministic and makes no disk writes. It fails with an assertion if the independent Pareto-membership algorithms disagree.

All objective calculations use integer values equal to **16 times** the physical squared-loss or robustness metric on the quarter-unit grid, so no floating comparison tolerance is required.

## Verification results

| Control | R15 symmetric | R16 asymmetric |
|---|---:|---:|
| Single-problem fronts | 30 | 30 |
| Shared-policy cross-problem fronts | 10 | 10 |
| Full membership masks compared | 40 | 40 |
| SHA-256 of serialized masks | `fc3cfe5a67bba6256fefe4a5f294045433ea9aa2e701ad1d26609a16bf57523e` | `417f51dd1f17e17f14683b75239662c2c492fadd173c38b9e572428ad6611dd8` |
| Common minimax candidates | All policies select (0,0) | Multiple distinct candidates |

R16 asymmetric shared-front cardinalities by policy:

```text
[114, 114, 49, 82, 43, 113, 115, 113, 49, 84]
```

Asymmetric per-problem cardinalities:

```text
A: [18,18,14,7,13,18,18,18,14,17]
B: [8,30,3,8,2,8,8,30,3,3]
C: [6,24,2,6,3,6,6,24,2,4]
```

In the asymmetric world, the selected *shared* minimax candidate is:

- `(0.25, 0.25)` under the baseline two-objective policy;
- `(0.0, 0.25)` under the `x <= 0` policy;
- `(0.25, 0.0)` under the axis-robustness 0.25 policy.

This demonstrates real variation between **finite-grid** policy choices, not a guarantee about continuous optima.

## Read-back provenance

The same R16 test file was written into isolated R16 experiment directories on FORGE, OAK and VERIFIER. All three produced the expected output and exact matching membership digests. SHA-256 of the locally executed source on each was identical:

```text
36726E8FE5C7B067BE529F6B6CB7F11ECECEED64E4B5A6A52CAFC479FDDBEABD
```

Distinct computers executing the same code does not constitute three independent algorithms: the **two explicit membership algorithms** inside the file are what independently check each mask. No continuous runtime service, private ingress, unreviewed autonomous publishing or spending was activated.

**Scope of publication:** only synthetic benchmark code and public documentation, with no private archive content or credentials.
