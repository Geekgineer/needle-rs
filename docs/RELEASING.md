# Releasing

One tag drives everything. Pushing `v*` runs
[`.github/workflows/release.yml`](../.github/workflows/release.yml), which builds
the artifacts and publishes to four registries. Hugging Face is the one manual
step.

## Channels

| Channel | Package | Built from | Job |
|---|---|---|---|
| crates.io | `needle-core`, `needle-infer`, `needle-c`, `needle-rs-cli` | the workspace | `crates-publish` |
| npm | `needle-rs` | `crates/needle-wasm` via wasm-pack | `npm-publish` |
| PyPI | `needle-rs` | `crates/needle-python` via maturin + cibuildwheel | `python-publish` |
| GitHub Releases | CLI + C libs + `.wasm`, per target | `build-native`, `build-wasm` | `github-release` |
| Cloudflare Pages | [needle-rs.pages.dev](https://needle-rs.pages.dev) | `examples/browser-demo` + `pkg/` | `deploy-demo` |
| Hugging Face | `Abdalrahman/needle-rs-safetensors` | **manual** | — |

Names that are not what you would guess, and why:

- **`needle-rs-cli`**, not `needle-cli`. Both `needle-cli` and `needle-rs` on
  crates.io belong to unrelated projects. The installed binary is still
  `needle-rs`. Never "fix" the crate name back — `cargo install needle-cli`
  fetches a stranger's tool.
- **`needle-wasm` and `needle-python` are `publish = false`.** They ship as the
  npm and PyPI packages both called `needle-rs`; neither is a crates.io artifact.
- The npm and PyPI packages **share the name `needle-rs`** and each carries its
  own README (`crates/needle-wasm/README.md`, `crates/needle-python/README.md`).
  Those files exist because the repo README's image links are repo-relative and
  render broken on a registry page. Keep every link in them absolute — CI fails
  the npm build if a relative one appears.

## Credentials

All five live as **GitHub Actions repository secrets** (Settings → Secrets and
variables → Actions). Nothing is stored in the repo, and no workflow echoes a
secret value.

| Secret | Used by | Scope needed |
|---|---|---|
| `CARGO_REGISTRY_TOKEN` | `crates-publish` | publish-update, plus publish-new for a crate's first release |
| `NPM_TOKEN` | `npm-publish` | automation token, publish on `needle-rs` |
| `PYPI_TOKEN` | `python-publish-upload` | project-scoped API token for `needle-rs` |
| `CF_API_TOKEN` | `deploy-demo`, `wasm-demo` | Cloudflare Pages: Edit |
| `CF_ACCOUNT_ID` | `deploy-demo`, `wasm-demo` | account identifier, not a secret as such |

All four crates now exist on crates.io, so publish-update is enough for an
ordinary release. **Adding a new crate to the workspace changes that**: the
token needs **publish-new** as well, and it needs it before the tag is pushed.
Crates are published leaf-first, so a token scoped to updates only will publish
every existing crate and then fail on the new one — leaving the release
half-done, with no way to unpublish the rest. This is what the 0.2.0 release had
to be sequenced around, when `needle-rs-cli` was first introduced.

Rotate the registry tokens on any maintainer change; all three registries treat a
published version as permanent, so a leaked token cannot be undone by yanking.

## Cutting a release

1. **Bump the version in both manifests.** `Cargo.toml`
   (`workspace.package.version`) and `pyproject.toml` (`project.version`). They
   must match the tag: the `version-guard` job fails the whole run before
   anything uploads if they don't. This guard exists because npm takes its
   version from the tag while crates.io and PyPI take theirs from the manifests —
   a tag that outran a bump would publish three different versions, none
   retractable.
2. **Update `CHANGELOG.md`** — move `Unreleased` to the new version, and refresh
   the link definitions at the bottom.
3. **Re-verify locally**, in both feature configurations:
   ```bash
   cargo test --workspace --release
   cargo test --workspace --release --features needle-core/parallel
   cargo clippy --workspace --all-targets --release -- -D warnings
   cargo clippy -p needle-core --no-default-features --release -- -D warnings
   cargo +1.87 check --workspace          # MSRV
   wasm-pack build crates/needle-wasm --target nodejs --release --out-dir ../../pkg-nodejs/
   node crates/needle-wasm/tests/node_e2e_v2.js
   ```
   The parity suites need `weights/needle2.cact`, `weights/needle.safetensors`
   and `weights/vocab.txt`; each skips with a printed notice if absent, so a run
   that is silently green may have tested nothing. Check the counts.
4. **Tag and push.**
   ```bash
   git tag -a v0.2.0 -m "needle-rs 0.2.0" && git push origin v0.2.0
   ```
5. **Watch the run.** `version-guard` gates everything; if a publish job fails
   partway, re-running is safe — every publish step is idempotent (npm checks
   `npm view` first, `cargo publish` treats "already exists" as success).
6. **Upload the Hugging Face weights** if the v1 conversion changed. Follow the
   checklist in [hf-model-card.md](hf-model-card.md) — banner before README, or
   the CDN caches a broken image. v2 needs nothing here: needle-rs reads
   upstream's own `.cact` container, so there is no conversion to host.

## Verifying a release landed

```bash
cargo search needle-infer                     # crates.io
npm view needle-rs version                    # npm
pip index versions needle-rs                  # PyPI
curl -sI https://needle-rs.pages.dev | head -1  # demo
```

crates.io's API rejects requests without a `User-Agent`; if you query it
directly, send one, or you will read a "not published" answer that is wrong.

## When a publish job fails

A rejected credential does not announce itself as one. Two rejections observed
during the 0.2.0 release:

| Job | What it printed | What it meant |
|---|---|---|
| Publish to npm | `npm error 404 Not Found - PUT https://registry.npmjs.org/needle-rs` | The token was rejected. npm answers unauthorized *writes* with 404 rather than 403 so it does not leak whether a package exists — so a 404 on a package you know is published points at the credential, not the package. |
| Deploy demo | `Authentication error [code: 10000]` on `/accounts/*/pages/projects/needle-rs` | The Cloudflare token was rejected. |

Before reading either as a code problem, check whether the step changed. If the
same step succeeded on an earlier run and nothing in the workflow moved, the
credential is the variable. `gh secret list` prints an update timestamp for each
secret, which is usually enough to spot the stale one.

Recovery, once the secret is rotated:

```bash
gh run rerun <run-id> --failed
```

Artifacts persist for the life of the run, so a partial re-run picks up the
existing `pkg-npm` and `dist-*` uploads instead of rebuilding them. Confirm with
`gh api repos/<owner>/<repo>/actions/runs/<run-id>/artifacts` if in doubt.

Two things make a partial failure survivable, and they are worth preserving if
this workflow is ever restructured:

- **Every publish step is idempotent.** `publish_crate` treats crates.io's
  "already exists" as success, so re-running after a mid-sequence failure will
  not try to re-publish what already landed. This matters because a crates.io
  release cannot be withdrawn — without the skip, one failed crate in a
  dependency chain would force a patch bump.
- **`deploy-demo` is a leaf.** Nothing depends on it, so a dead Cloudflare token
  cannot block crates.io, npm or PyPI.

## Known rough edges

- `wasm-opt` is not run by wasm-pack (the crate sets `wasm-opt = false`, since
  the binary wasm-pack downloads fails in some environments). Both release
  workflows run it as an explicit step. A local `wasm-pack build` therefore
  produces a 462 KB module where CI produces 529 KB.
- Python wheels are `abi3` (pyo3 `abi3-py38`), so each platform gets exactly one
  wheel that serves every CPython >= 3.8. 0.2.0 published 5 wheels, not 25, and
  they were verified to install and import on 3.13 and 3.14. `CIBW_BUILD` lists
  `cp38-*` through `cp312-*`, but cibuildwheel detects the abi3 wheel is
  compatible and skips the cp39–cp312 builds in under 0.1 s each, so the extra
  entries cost nothing — narrowing the list would be cosmetic, not a speed-up.
- Linux aarch64 gets no wheel: cibuildwheel would need QEMU emulation, which is
  too slow to run in the release job. aarch64 Linux users build from source.
- `needle-cli`'s directory name still differs from its published name. Renaming
  the directory would be churn for no gain, but it does surprise people.
