## What does this PR do?

<!-- One sentence summary -->

## Motivation

<!-- Why is this change needed? Link to the related issue if applicable. Fixes #NNN -->

## Changes

<!-- Bullet list of what changed -->

## Parity / correctness

- [ ] I ran `cargo test --workspace` and all tests pass
- [ ] If this touches inference logic: I verified numerical parity with the Python reference
      (or added a test to `crates/needle-infer/tests/` that covers it)

## Checklist

- [ ] `cargo fmt --all` applied
- [ ] `cargo clippy --all-targets --all-features -- -D warnings` passes
- [ ] No new panics in hot path (`unwrap`/`expect` only in tests or one-time init)
- [ ] Public API changes are doc-commented
