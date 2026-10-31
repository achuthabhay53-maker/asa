
# ACA/ASA Project Instructions

## Before changing code
- Inspect the existing implementation and relevant tests first.
- Explain the planned changes before making broad changes.
- Make the smallest targeted change that solves the task.
- Do not overwrite or discard existing user changes.
- Work only on the current feature branch. Do not switch branches.

## Agent architecture
- Preserve the existing agent responsibilities and workflow.
- Preserve the Agent 3 -> Agent 4 handoff contract.
- Do not change Agents 4, 5, or 6 unless necessary.
- If a schema change is necessary, explain why and test compatibility.

## Agent 3 verification rules
- Use only the customer's official website and official pages
  under its control as evidence for company-specific claims.
- Do not use third-party directories as verification evidence.
- Link evidence to a claim only when the quoted content directly
  supports that claim.
- Do not mark inconclusive claims as verified.
- Do not mark a subtopic fully supported when its central claims
  lack sufficient direct evidence.
- Keep unsupported claims and human-review requirements explicit.

## Testing and code quality
- Inspect the project's test configuration before running tests.
- Run relevant tests for every code change.
- Run formatting, linting, and type checks when configured.
- Never claim a test passed unless it actually ran successfully.
- Report changed files, tests run, failures, and remaining risks.

## Git and release safety
- Never commit API keys, passwords, tokens, or other secrets.
- Do not push directly to main.
- Do not merge pull requests.
- Do not force-push or rewrite Git history.
- Do not delete files or make unrelated refactors without approval.
- Create a summary of changes for human review.
- Keep changes limited to the task and current feature branch.
- ## GitHub Actions branch workflow
- When a task is triggered from a GitHub Issue, work on the
  dedicated Claude-created branch for that task.
- Use `setup-claude-github` as the base branch for new task branches.
- Never directly push changes to `main` or `setup-claude-github`.
- Never switch branches manually.
- Provide a Pull Request link targeting `setup-claude-github`
  when the task is complete.
- Run relevant tests before considering the task complete.
- Human review and approval are required before merging.

