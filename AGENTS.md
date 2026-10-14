# RESCUE session entry

For paper writing, research discussion, or continuation of the attention-guided
speculative decoding work, first read [docs/PAPER_HANDOFF.md](docs/PAPER_HANDOFF.md).
It records the user's latest architecture, implemented experiments, evidence
limits, and paths to the source reports. A ready-to-use writing prompt is in
[docs/NEW_SESSION_PROMPT.md](docs/NEW_SESSION_PROMPT.md).

Keep these distinctions explicit:

- Module numbers describe the paper: 1 = context selection, 2 = selective
  content injection, 3 = draft-length control. Experiment groups A/B/C instead
  mean selection / length / their joint decision.
- Attention-weighted content coverage is a proposed selector with a synthetic
  cost benchmark. It has no model-quality or end-to-end decoding evaluation.
- Read-preservation selection is the completed older experiment, with a
  different objective. Its results cannot validate the new coverage selector.
- Performance-supervised V injection has a positive acceptance signal against
  its matched selection baseline. Precise token-content correspondence and the
  independent value of learned gates remain unproved.
- Current length/joint policies have not shown net acceleration; do not write
  the complete paper hypothesis as an established empirical conclusion.

Use the original per-experiment protocols and reports for numbers. Keep
different datasets, budgets, numerical precisions, and output limits separate.
The latest explicit user request determines whether to write, implement, run
experiments, or publish changes; a handoff alone is not a new experiment request.
