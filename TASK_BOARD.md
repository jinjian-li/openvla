# Task Board

Updated: 2026-05-29

## Roles

### User

- Owns final priorities, budget decisions, remote GPU rental, and external accounts.
- Provides missing credentials only through local environment variables or private files.
- Reviews major direction changes before expensive training runs.

### Codex

- Owns technical decisions, task decomposition, architecture, code review, and complex implementation.
- Defines task cards that Claude Code can execute mechanically.
- Reviews Claude Code outputs before they become project direction or git history.
- Keeps sensitive information out of tracked files.

### Claude Code

- Executes explicit task cards from this board or `handoff/` notes.
- Handles long mechanical work: batch edits, log summarization, command execution, result table cleanup, and documentation passes.
- Reports exact commands, changed files, outputs, and blockers in a handoff note.
- Does not invent new project scope without user or Codex approval.

## Current Task Cards

| ID | Owner | Status | Task | Done When |
|----|-------|--------|------|-----------|
| T001 | Codex | Done | Convert `PROJECT_PLAN.md` to internal execution plan | Plan uses direct execution language, PI0 port is `6009`, Isaac Sim is optional |
| T002 | Codex | Done | Add agent workflow files | `TASK_BOARD.md` and `handoff/HANDOFF_TEMPLATE.md` exist and are tracked safely |
| T003 | Codex | Done | Summarize current OpenVLA baseline run | Summary written to ignored handoff note; deterministic 3-task result is available locally |
| T004 | Codex | Ready | Review PI0 action interface | Decide controller strategy and define the next implementation card |
| T005 | Claude Code | Blocked | Download or verify Libero SFT dataset | Waits for Codex/user to specify dataset source and storage path |
| T006 | User | Ready | Confirm next paid GPU window | Decide when to run SFT/RL jobs and expected budget ceiling |
| T007 | Codex | Done | Harden OpenVLA base eval metadata and init-state handling | Client records reproducibility metadata and uses bundled Libero `.pruned_init` files |

## OpenVLA Baseline Result

Latest deterministic small baseline:

```bash
/media/li/新加卷/isaacsim/libero_env/bin/python3 -u libero_openvla_client.py \
  --episodes 1 \
  --task-limit 3 \
  --output openvla_base_det_3tasks_1ep.json
```

Summary:

- Commit: `55770e8`
- Server decode: `do_sample_false`
- Init states: bundled Libero `.pruned_init`, 50 per task
- Image path: camera `256`, resize `224`, `rotate180`
- Result: `0/3`, reward `0.0`, all three episodes ran 230 steps
- Raw gripper: `0.0` throughout, converted to close commands

Detailed note: `handoff/2026-05-29-openvla-baseline.md` (ignored by git).

## Next Claude Code Card

### T004 Assist: PI0 Action Interface Evidence Pack

Goal: gather evidence for Codex's PI0 action-interface review without changing controller behavior.

Steps:

1. Inspect `libero_pi0_client.py`, `pi0_libero_inference.py`, and any PI0 server/client logs.
2. Identify the action vector shape, units, and whether outputs look like joint targets, joint deltas, or Cartesian deltas.
3. Locate where Libero robot state is passed or omitted.
4. Summarize the current PI0 result JSON and failure symptoms.
5. Write findings into a new ignored handoff note under `handoff/`.

Constraints:

- Do not commit.
- Do not modify evaluation scripts unless Codex explicitly asks.
- Do not include credentials, SSH host passwords, or private environment values.
- Do not use `rm`.

## Review Rules

- Claude Code output is treated as draft until Codex reviews it.
- Any change that affects model training, controller behavior, reward logic, or result interpretation needs Codex review.
- Any command that spends GPU time or starts a long remote job needs user approval unless already authorized in a task card.
