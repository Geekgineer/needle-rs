# Security Policy

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Report security issues by email to: **hi@geekgineer.com**

Include:
- A description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested fix (optional)

You will receive a response within 72 hours. If the issue is confirmed, a fix
will be released as soon as possible and you will be credited in the release
notes (unless you prefer otherwise).

## Scope

This policy covers the needle-rs runtime code in this repository. It does not
cover the Needle model weights or the upstream Python reference implementation
(report those to [cactus-compute/needle](https://github.com/cactus-compute/needle)).

## Known Non-Issues

- The C ABI (`libneedle_c`) exposes raw pointer APIs by design. Callers are
  responsible for following the documented safety contracts in `needle.h`.
- The WASM build runs in the browser sandbox; host security properties apply.
