# Reference-conditioned policy Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans, native single implementer, followed by independent review. User approved the discussed conditional-policy model and explicitly asked to start experiments; no repeated approval gate.

**Goal:** Train and falsify a reference-conditioned policy that predicts decisions of previously unseen adapter-training players, distinguishing personalization from context alignment.

**Architecture:** Frozen humanv0 exposes full 384-channel spatial action features plus pass. A shared learned context key and residual-value encoder read each candidate's reference games before predicting all legal query actions. Hierarchical attention pools within games then averages games; a null memory slot permits baseline fallback. A uniform-memory arm removes context alignment.

**Tech Stack:** existing PyTorch/NumPy, checked SGF replay, R1 Python, production Slurm.

**Spec:** Approved conversational design; precise bounded pilot specification below. This is a mechanism pilot, not an official baseline-improvement experiment.

## Global Constraints

- Existing worktree/branch; no edits to frozen choice/exact/event protocol sources. New refpolicy subtree in existing run root.
- Continue cumulative 2 concurrent GPUs / 6 GPUh / 1 GiB study artifacts; current consumption 4691 GPU seconds and 786.3 MB. Budget ledger remains centralized; this plan has its own progress ledger. No fold0 or official questions. No adoption/submission.
- All computation/tests on production Slurm. Large action tensors live only in allocation RAM/GPU; do not write a multi-GB cache. Save manifest, small model checkpoints, metrics and provenance.
- Fixed pilot: 200 training players from folds1/2, 50 development fold3, 50 test fold4; deterministic hash selection among owners with >=10 distinct valid games and >=32 subject moves in first256plies. Ten games each, ordered by hash; dev/test first7 reference/last3 query. Full-game observation duplicates across selected splits must be rejected (no shared game from opposite-color records). Owner IDs are opaque; not model inputs.
- 32 deterministic evenly spaced subject positions from first256plies per game. This subsampling bounds pilot; conclusions do not cover full-game performance. Keys use only pre-action state, not played action, future turns, game IDs or owner IDs. Actual actions enter reference values and query loss only.
- Context = mean spatial trunk384 + pre-move value/entropy/ply/subject color (388). Action feature = spatial trunk384 + pass indicator (385). Reference value = played feature minus policy expectation. Query logits = frozen normalized policy logp + bounded learned correction2*tanh(action_projection dot attended_value/sqrt64). Full362 legal normalization, no sampled-action normalizer. Reference-only attention pergame includes a fixed null logit0/value0, then averages referencegames equally. Zero reference values must recover baseline exactly.
- Context standardization and action/value RMS fitted only on first7games of training players; fixed for development/test. Model hidden64, bias-free value/action projections (zero inputs map to zero); shared context MLP388->64->64. Uniform arm uses identical value/action path with uniform token/null weights, omitting learned context matching. Not advertised as equal-effective-parameter control.
- Two seeds17/29, two arms attention/uniform. AdamW lr3e-4 weight_decay1e-4;4000steps, gradientclip1. Training16owners/episode,3references+1disjointquerygame each,4random stored query positions perowner. First1000steps random owners; then half episodes draw8seeds+their nearest remaining training-owner residual-profile neighbors. These are frozen-feature hard negatives, not full-fusion error mining. Same episode RNG per paired arms.
- Loss = identity crossentropy of mean played-move log likelihood over4positions (temperature10) + .1 true-owner move NLL. Candidate count16 in training,50 in dev/test; whole games never shared between reference/query in an episode. Reference player slots permute equivariantly; no identity lookup table.
- Development every1000steps, select lowest identityCE+.1moveNLL over50owners x3querygames (32positions each,7refs); fixed tie earlier. Test read only after checkpoint selection for all runs; no tuning after test.
- Report test perseed Top1/rank score, moveNLL, own-vs-next-owner support advantage, key-permutation and zero-value ablations, uniform arm, null mass. Fixed hash permutation of context keys within each referencegame preserves values. This inference intervention alone is not proof of context semantics; trained uniform control is necessary.
- Primary gate: both seeds own-vs-wrong move-LL advantage owner-bootstrap95%lower>0 AND paired attention-minus-uniform rankscore95%lower>0. If only first passes: personalization supported, alignment not established. If neither: mechanism not established. Bootstrap5000owners, reference/checkpoints fixed; no moves treated as independent samples. Scores equal-weight1/2/3querygames; deterministic first2/all3 plus3singles. Not the full fusion, not official gains. Regardless of result preserve failures and scope.
- Before full experiment run synthetic learnability, legal normalizer, action-blind key, support/query leakage, reference permutation, split-selection, source-integrity tests and a64game resource pilot. Reuse checked legacy replay. Pilot must fit remaining budget; change budgets only by explicit new user authorization.

## Review Focus

1. Query played move cannot affect memory selection; action used only after distributions exist.
2. Same game cannot leak across player splits or support/query; duplicate physical games ignore subject for audit.
3. Illegal logits stay zero probability, pass is legal, zero-memory returns normalized baseline.
4. Controls share query positions/references/checkpoint rules; identity order equivariance; no oracle hard-negative evaluation shortlist.
5. Evaluation normalization uses training data only; test never selects parameters/checkpoints; source changes invalidate continuation.

## Task1: Mechanism and synthetic tests

Files: src/aicup_pid/reference_policy.py; tests/test_reference_policy.py.
Interfaces: ReferencePolicy(context_dim,action_dim,hidden=64,mode='attention').forward(qcontext,actions,base_logp,reference_context,reference_value) -> logp[Q,C,A], null_mass[Q,C]. Reference inputs [C,G,T,D]; no query action argument. episodic_loss(logp,played,owners) and summarize_scores(scores[owner,3,candidate],move_advantage,draws).

- [ ] Write RED tests for full legal normalization/null fallback, action-blind context path, reference-permutation equivariance, context-action synthetic opposite identities; run Slurm.
- [ ] Implement minimal attention/uniform model and paired-owner summaries, synthetic overfit verification on Slurm; commit.

## Task2: Data and experiment runner

Files: scripts/pid_reference_policy.py; scripts/nano4/pid_reference_policy.sh; tests/test_reference_pipeline.py.
Interfaces: select_games(rows,folds,counts), position_indices(n,count=32), validate_games(), extract manifest to in-memory tensors, train/evaluate/save checkpoint/report. Manifest has asset/source hashes and fixed settings. No large action-feature files.

- [ ] RED deterministic disjoint splits/duplicate game/leakage/source tests. Implement runner using checked replay and existing backbone hooks.
- [ ] Run full suite,64game throughput/memory pilot; record projected full-run time and storage.
- [ ] Freeze source protocol. Run3000game extraction + fixed four training runs; stream checkpoints/metrics and guard runtime/storage.

## Task3: Scientific decision and verification

- [ ] Inspect completeness/numerics and paired-owner confidence intervals; compare against predeclared gate without substituting arms.
- [ ] Independent fresh-context code/results review, fix material defects with tests.
- [ ] Update METHOD_NOTES and small final JSON; verify tests/jobs/budget; commit. If pilot supports mechanism, next required milestone is separately designed full-baseline complementarity on candidate-only holdouts, not automatic model adoption.
