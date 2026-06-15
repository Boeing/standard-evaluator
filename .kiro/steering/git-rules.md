---
inclusion: always
---

# Git Rules

## Branching Workflow

- Branches are created via GitLab: an issue is created first, then a merge request is opened named after the issue title.
- Merge requests target `dev`. Once merged, the issue is closed automatically.
- Do not create branches locally — always use the GitLab-driven process.

## Commit Message Format

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<optional scope>): <short summary>

<optional body>
```

- **type**: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `ci`, `build`, `perf`, `style`
- **scope** (optional): module or area affected (e.g., `kriging`, `opt-problem`, `evaluators`, `ci`)
- **summary**: imperative mood, lowercase, no period, max ~72 characters
- **body** (optional): bullet list of notable changes when the summary alone isn't sufficient

## Allowed Git Commands (no approval needed)

- Read-only: `git status`, `git log`, `git diff`, `git show`, `git branch` (listing only), `git tag` (listing)
- `git add`
- `git commit`
- `git push`, `git push -u`
- `git fetch`
- `git pull`
- `git checkout` / `git switch` (switching branches)
- `git stash`, `git stash pop`, `git stash list`
- `git merge` (non-force)
- `git rebase` (non-interactive, non-force)

## Requires Explicit Approval

- `git rm`
- `git push --force` / `git push --force-with-lease`
- `git reset --hard`
- `git clean -f`
- `git branch -D` (force delete)
- `git branch <name>` (creating branches — use GitLab instead)
- `git rebase -i` (interactive rebase)

## Command Execution

- Always run git commands as **separate individual calls** (e.g., `git add`, then `git commit`, then `git push`). Do not chain them with `;` or `&&` in a single command — chained commands don't match trusted command patterns and will trigger unnecessary approval prompts.

## Guidelines

- One logical change per commit — don't mix unrelated changes.
- Prefer staging specific files over `git add .` to avoid committing unrelated changes.
- Never commit secrets (`.env`, credentials, tokens).
- Prefer new commits over `--amend`. Only amend unpushed commits when explicitly asked.
- Never push directly to `main` or `dev` unless explicitly asked.
