# tests/ — handoff test surface

The handoff includes three layers of testing:

1. pure-Python fixed-K=4 protocol tests;
2. backend contract tests;
3. the V3 pre-experiment correctness gate and its recorded PASS evidence.

The canonical Harness V3 CPU regression is in
`source/harness/rescue_harness/test_harness.py`.
The K4 tests cover full acceptance, rejection/rollback, EOS, short final blocks and boundaries.

Historical failed gate attempts are evidence only; do not treat them as current failures.
The accepted V3 gate passed 10/10 required gates.
