# Task Board

Updated: 2026-05-29

## Roles

### User

- Owns final priorities, budget decisions, remote GPU rental, and external accounts.
- Provides missing credentials only through local environment variables or private files.
- Reviews major direction changes before expensive training runs.

### Active Agent (Claude Code or Codex)

- One agent owns the work continuously: analyze → implement → verify → record.
- Does not switch agents unless the user explicitly asks, or the agent hits a high-risk decision it cannot resolve.
- Keeps sensitive information out of tracked files.

## Task Board and Handoff Rules

- `TASK_BOARD.md` is a progress overview for the user, not a strict scheduling system.
- `handoff/*.md` files preserve context across restarts/disconnects/context loss.
- Every agent must write a handoff note before stopping.
- Before starting, read `TASK_BOARD.md` and the latest handoff note.

## Current Phase

OpenVLA baseline is verified (`0/3`, credible zero-shot failure). Moving to:

1. PI0 baseline smoke test → baseline eval
2. OpenVLA SFT on Libero data
3. Evaluate fine-tuned model

## Current Task Cards

| ID | Status | Task | Notes |
|----|--------|------|-------|
| T001 | Done | OpenVLA baseline eval path | Verified deterministic `0/3`, credible zero-shot result |
| T002 | Done | Agent workflow files | `TASK_BOARD.md` + `handoff/` in place |
| T003 | Ready | PI0 baseline smoke test | Run 1 task × 1 episode, verify client/server/tunnel work |
| T004 | Ready | PI0 small baseline | 3 tasks × 1 episode, compare with OpenVLA |
| T005 | Ready | Download Libero SFT dataset | `openvla/modified_libero_rlds` on remote |
| T006 | Ready | OpenVLA SFT on Libero | Modify finetune script, run debug training |
| T007 | Blocked | Full SFT + evaluation | Needs T005+T006 done first |

## OpenVLA Baseline Result

- Commit: `55770e8`
- Server decode: `do_sample_false`
- Init states: bundled Libero `.pruned_init`
- Image: camera `256` → resize `224`, `rotate180`
- Result: `0/3`, reward `0.0`
- Detailed note: `handoff/2026-05-29-openvla-baseline.md`

## Review Rules

- Default: active agent owns all decisions and runs continuously.
- User approval needed before: spending GPU money on long runs, changing action semantics / controller / reward logic, deleting important files.
- If results conflict with expectations, pause and discuss with user before changing direction.
- Agent should explicitly flag when it is uncertain about a high-risk judgment.
