# Contributing to mwCore

First off, thank you for considering contributing to `mwCore`! It's people like you that make `mwCore` a central, robust tool for TI mmWave radar research and robot navigation.

## 1. Local Development Environment

We use [`uv`](https://github.com/astral-sh/uv) to manage dependencies and virtual environments.

### Prerequisites
1. Install Python 3.10+.
2. Install `uv`.

### Setup
Clone the repository and install dependencies:

```bash
git clone https://github.com/barrydoooit/mwCore.git
cd mwCore
uv sync --all-extras
```

## 2. Recommended Git Workflow

We follow a strict Pull Request workflow to maintain code quality:

1. **Find or Create an Issue:** Before writing code, ensure an Issue exists for your proposed feature or bug fix. This allows the maintainers to discuss the approach with you beforehand.
2. **Branch Formatting:** Create a branch from `mwcore-dev` using the format `<type>/<issue-number-or-short-desc>`. (e.g., `feat/obstacle-avoidance`, `docs/add-contributing-guide`, `fix/12-tracker-crash`).
3. **Commit Messages:** Follow [Conventional Commits](https://www.conventionalcommits.org/).
   - `feat:` for new features (e.g., `feat: integrate raw radar point cloud processing`)
   - `fix:` for bug fixes
   - `docs:` for documentation changes
   - `test:` for adding or updating tests
   - `chore:` for maintenance tasks, dependency updates, etc.
   - Example: `feat: add Dynamic Window Approach module (Resolves #42)`
4. **Pull Request:** Push your branch to GitHub and open a Pull Request against `mwcore-dev`. Fill out the PR template completely.

## 3. Testing and Verification Requirements

**All new features and mathematical changes MUST include tests.**

For instance, if you port or modify core mathematical functions (like the ones in `mwcore/signal_processing`), you should provide a verification script similar to `tests/verify_mwcore_math.py`.

Before submitting your PR, ensure your local changes work. If tests fail, your PR will not be merged.

## 4. Code Quality

While we don't have automated pre-commit hooks currently running, please ensure your code is readable, documented with docstrings, and follows standard PEP8 conventions as much as possible.
