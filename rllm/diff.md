# Changes relative to upstream `rllm`

This folder is a snapshot of [rllm](https://github.com/rllm-org/rllm) with two small, surgical modifications required to train and evaluate the MAGE shadow-memory judge. All other files are unchanged from the upstream baseline.

## Summary

Two files modified, ~109 net lines added:

| File | Change |
| ---- | ------ |
| `rllm/engine/agent_workflow_engine.py` | +1 line |
| `rllm/trainer/verl/agent_workflow_trainer.py` | +108 / −5 lines |

No files deleted, no public APIs removed.

## `rllm/engine/agent_workflow_engine.py`

Forward the engine's current mode (`train` / `val`) into each rollout invocation via `kwargs`, so downstream workflows can branch on it (e.g., to swap datasets or toggle logging).

```diff
+ kwargs = {**kwargs, "mode": self.current_mode}
```

## `rllm/trainer/verl/agent_workflow_trainer.py`

Adds three pieces of metric logic needed to monitor the shadow-memory judge during training and validation. None of the changes alter the training loop's control flow — they only add metrics and a final-checkpoint save.

### 1. Step-wise classification metrics for the judge

Workflows emit per-step `judge_tp`, `judge_fp`, `judge_tn`, `judge_fn` counts. Averaging them per-episode produces a biased estimate because episodes have different step counts. Instead, sum TP/FP/TN/FN across the batch (or validation set) and derive accuracy, precision, recall, and F1 from the totals.

- New helper: `_stepwise_classification_metrics_from_sums(tp, fp, tn, fn) -> dict`
- New constant: `WORKFLOW_CONFUSION_KEYS = frozenset({"judge_tp", "judge_fp", "judge_tn", "judge_fn"})`
- In the **training** loop: confusion counts are summed over the batch and turned into `batch/judge_accuracy`, `batch/judge_precision`, `batch/judge_recall`, `batch/judge_f1`. Non-confusion workflow metrics keep the original per-episode mean.
- In the **validation** loop: the same treatment is applied per `data_source`, producing `val/{data_source}/judge_{accuracy,precision,recall,f1}`.

### 2. Trajectory-level ASR and BU

ASR (Attack Success Rate) and BU (Benign Utility) are defined at trajectory granularity — a harmful trajectory where the judge never issued DENY counts as an attack success, a benign trajectory where the judge never issued DENY counts as correctly allowed. Episodes terminating in `ERROR` or `MAX_RESPONSE_LENGTH_EXCEEDED` are excluded from the denominators.

- New helper: `_compute_trajectory_asr_bu(metrics_list, extra_info_list, termination_reasons_list) -> {"asr", "bu"}`
- Definition follows `memagent/core/testv2.py` so training-time metrics match the offline evaluator.
- Training loop logs `batch/asr` and `batch/bu`.
- Validation loop logs `val/{data_source}/asr` and `val/{data_source}/bu`.

Input labels are read from each episode's `extra_info["labels"]`; if stored as a JSON string they are deserialized. A trajectory is considered harmful if any step's label is `1`.

### 3. Final checkpoint save

After the final validation at end-of-training, the trainer now calls `self._save_checkpoint()` explicitly so the last-step weights are always on disk regardless of the configured save frequency.

```diff
+ # Save final checkpoint
+ print(f"Saving final checkpoint at step {self.global_steps}")
+ self._save_checkpoint()
```

## How to regenerate this diff

From the source `rllm` repository:

```bash
git diff master memrl
```

`master` is the initial snapshot of upstream `rllm`; `memrl` is the branch exported into this submission.
