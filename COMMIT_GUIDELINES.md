# Commit Guidelines

This document outlines the requirements and best practices for writing commits in this project.

This document uses the key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" as defined in [RFC 2119](https://www.ietf.org/rfc/rfc2119.txt).

## Table of Contents

- [Conventional Commits](#conventional-commits)
- [Atomic Commits](#atomic-commits)
- [Commit Messages](#commit-messages)

---

## Conventional Commits

All commits MUST follow the [Conventional Commits](https://www.conventionalcommits.org/) specification.

---

## Atomic Commits

Commits MUST be atomic, meaning each commit SHOULD represent a single, indivisible unit of change.

### What Makes a Commit Atomic

An atomic commit:

- **MUST** contain all changes necessary for the feature, fix, or refactor to function
- **MUST** include the implementation, tests, and documentation together in a consistent state
- **MUST** leave the codebase in a working state (builds successfully, tests pass)
- **MUST** be focused on a single logical change
- **MUST NOT** mix unrelated changes
- **SHOULD** be revertable independently without breaking other functionality
- **SHOULD** be small enough to review in one sitting
- **SHOULD NOT** contain "while I'm here" changes

### Guidelines for Creating Atomic Commits

1. **One Logical Change**: Each commit addresses exactly one bug, feature, or refactor
   - If you use "and" in your commit message, you might need to split it

2. **Complete and Self-Contained**: All necessary components are included
   - Code implementation
   - Unit tests
   - Integration tests (if applicable)
   - Documentation updates
   - Configuration changes (if needed)

3. **Functional at Every Commit**: The build MUST succeed and tests MUST pass
   - Never commit broken code, even temporarily
   - Each commit in your history should be a valid checkpoint

4. **Independent and Revertable**: Commits can be cherry-picked or reverted without side effects
   - No hidden dependencies on other commits in your branch
   - Reverting a commit should cleanly undo that specific change

5. **Properly Scoped**: Not too large, not too granular
   - Too large: "refactor entire authentication system"
   - Too granular: separate commits for implementation and tests
   - Just right: "add password reset functionality"

### Benefits of Atomic Commits

Breaking down large features into atomic commits provides:
- **Easier code review**: Reviewers can understand each change independently
- **Clearer history**: `git log` tells a coherent story of how the code evolved
- **Better debugging**: `git bisect` can pinpoint exactly which change introduced a bug
- **Simpler reverts**: Roll back specific changes without affecting other work
- **Flexible integration**: Cherry-pick commits to different branches as needed

---

## Commit Messages

### Description Line

The description line (first line) of a commit message:

- **MUST** use the imperative mood ("add feature" not "added feature" or "adds feature")
- **SHOULD** be concise (50 characters or less is RECOMMENDED)
- **MUST NOT** end with a period
- **SHOULD** complete the sentence: "If applied, this commit will..."

### Body

The commit body:

- **MAY** be included to provide additional context
- **SHOULD** wrap at 72 characters
- **MUST** explain *why* the change was made, not just *what* changed
- **SHOULD NOT** describe *how* to use the feature (that belongs in docs)

#### Writing a Good "Why"

The body should answer: *What problem does this solve and why does it matter?*

**Structure for feature commits:**
1. State the problem or limitation being addressed
2. Explain the solution approach (briefly)
3. Describe the benefit or what this enables

**Good example:**
```
feat(auth): add session timeout configuration

Sessions previously had a hardcoded 24-hour timeout, which forced users
to re-authenticate daily regardless of activity. This was frustrating
for users in long-running workflows.

Add a configurable timeout with activity-based refresh. Administrators
can now balance security requirements against user experience for their
specific use case.
```

**Weak example** (focuses on what, not why):
```
feat(auth): add session timeout configuration

Add a new SESSION_TIMEOUT environment variable that accepts a duration
string. Users can set this to configure how long sessions last. The
default is 24 hours for backward compatibility.
```

The weak example describes the feature but doesn't explain why anyone would want it.

### Footer

The footer:

- **MAY** include references to issues, tickets, or breaking changes
- **MUST** use `BREAKING CHANGE:` to indicate breaking changes
- **MAY** include `Closes #123` or `Fixes #456` to link issues

