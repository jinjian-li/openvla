# Task Board

Updated: 2026-05-28

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
| T003 | Claude Code | Ready | Summarize current OpenVLA baseline run | Result JSON/logs are located, key metrics are summarized, missing data is listed |
| T004 | Codex | Ready | Review PI0 action interface | Decide controller strategy and define the next implementation card |
| T005 | Claude Code | Blocked | Download or verify Libero SFT dataset | Waits for Codex/user to specify dataset source and storage path |
| T006 | User | Ready | Confirm next paid GPU window | Decide when to run SFT/RL jobs and expected budget ceiling |

## Next Claude Code Card

### T003: Summarize OpenVLA Baseline Run

Goal: produce a concise status note for the current OpenVLA Libero baseline evaluation.

Steps:

1. Inspect tracked and untracked result files without deleting anything.
2. Identify which script produced each result file when possible.
3. Summarize task count, episode count, success rate, reward, latency, and obvious failure modes.
4. List missing information needed for a clean baseline table.
5. Write the summary into a new ignored handoff note, not into `CONTEXT_FOR_CODEX.md`.

Constraints:

- Do not commit.
- Do not modify evaluation scripts.
- Do not include credentials, SSH host passwords, or private environment values.
- Do not use `rm`.

## Review Rules

- Claude Code output is treated as draft until Codex reviews it.
- Any change that affects model training, controller behavior, reward logic, or result interpretation needs Codex review.
- Any command that spends GPU time or starts a long remote job needs user approval unless already authorized in a task card.
