# Conditional Behavior Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether a learned conditional policy uses personal reference behavior across games and unseen players.

**Architecture:** Frozen full-channel KataGo features feed shared state/history and reference encoders. Compare no-context, global-context, and attention-context policies with identical data and budgets.

**Tech Stack:** Existing Python environment, NumPy, PyTorch, pytest, Slurm.

**Spec:** `docs/superpowers/specs/2026-11-16-conditional-behavior.md`

## Global Constraints

- No fold 0, official query data, production changes or writes to sibling worktrees.
- Fixed 64/16/20 player split; reference/query games disjoint; both seeds evaluated together.
- One GPU, <=60 allocated minutes including failures; <=5.5 GiB cache; >=8 GiB /work free.
- No test-driven parameter changes; findings are exploratory mechanism evidence.

## Review Focus

- Current target action or future moves leaking into the conditioning path.
- Different players or games crossing the model selection boundary.
- Padding, pass, illegal actions, and nonfinite log-policy producing misleading loss.
- Reference permutation ablation accidentally changing the global-summary control.
- Test evaluation selecting hyperparameters, seeds or checkpoints.

### Task 1: Model and invariants

Files: create `scripts/behavior_model.py`, `scripts/behavior_data.py`, `tests/test_behavior.py`.
Interfaces: `BehaviorPolicy(mode).forward(phi, logp, state, history, ref_state, ref_history, ref_value, ref_mask, ablation=None)` returns query x candidate x action log probabilities; no query label argument. `player_split(rows)`, `prefix_history(moves,t,target)`, `position_indices(n)` and `episode_games(groups,rng)` enforce data boundaries.

- [ ] Write tests for masks/pass, reference and action equivariance, non-personal arm invariance, ablations, prefix causality, player/game separation and equal-marginal context transfer.
- [ ] Run missing-module RED; implement; run project suite to GREEN.
- [ ] Commit only pilot files.

### Task 2: Immutable full-feature extraction

Files: create `scripts/pid_behavior_extract.py`; extend data tests.
Interfaces: `--prepare --run-dir` freezes source/script hashes and split; `--extract` creates validated resumable per-game NPZ and completion inventory. Reuse checked board initialization and frozen engine loader read-only.

- [ ] Test protocol repeatability, refusal to overwrite mismatched provenance, short-game sampling, illegal targets and cross-split duplicate SGFs.
- [ ] Implement deterministic 16-position selection and full legal-action feature export with memory/disk guards.
- [ ] Run tests, synthetic model check, and frozen extraction; verify all 1000 games and metadata.

### Task 3: Training and mechanistic evaluation

Files: create `scripts/pid_behavior_train.py`, `tests/test_behavior_eval.py`; report under `results/final/`.
Interfaces: train six fixed arm/seed models, validation-only checkpoint selection, test once after all selections; JSON contains per-player paired metrics, CI, verdict, hashes and resources.

- [ ] Test episode disjointness, candidate/truth indexing, paired player bootstrap and explicit gate classification on hand-computed fixtures.
- [ ] Implement bounded training, progress/checkpoint writes and test permutation/no-action controls.
- [ ] Run full suite, fresh whole-branch review, fix correctness findings, then execute frozen real-data pilot.
- [ ] Read result artifacts, verify saved scores reproduce metrics, report success or failure without claiming R1 rank improvement.

Execution ruling: user explicitly said start testing after reviewing the design; proceed inline without another confirmation. A single fresh reviewer at the end is required by executing-plans. Preserve research code and results for reproducibility; no merge.
