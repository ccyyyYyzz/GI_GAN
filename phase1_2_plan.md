# Phase 1.2 Plan and Execution Notes

1. Restore Phase79 Rad-5/64 operator and checkpoint exactly.
2. Run alignment smoke tests for zero-noise determinism and stochastic P0 response.
3. Run E2a candidate coverage on legacy dev split with Kmax=32.
4. If coverage passes, build 64px train/val candidate caches with K=16.
5. Train/evaluate scalar pair, sum-image, scratch dual, raw-FCC, DM-FCC, and structural DM-FCC selectors on shared candidate pools.
6. Do not run final locked test in this run because final selector configuration has not been formally frozen.
