---
paths:
  - "**/*.rs"
  - "ci/**"
---

# Building and testing locally

- bindgen needs libclang: install `libclang-dev` and set `LIBCLANG_PATH=/usr/lib/llvm-18/lib`
  (`ci/harness.sh` finds it itself).
- Test binaries need libwayland-server >= 1.23 (`wl_client_set_max_buffer_size`,
  `wl_display_set_default_max_buffer_size`). Ubuntu 24.04 ships 1.22, so `cargo test` fails to link.
  - Workaround: compile a C file defining both as no-ops and pass
    `RUSTFLAGS="-C link-arg=<stub>.o"`. `cargo check --tests` needs no stub.
  - CI uses the gstreamer 1.26.7 container, which has a new enough libwayland.
- The compositor creates `$XDG_RUNTIME_DIR/wayland-N`; give each parallel test run its own short
  `XDG_RUNTIME_DIR`.
- Plugin tests run the compositor with `render-node=software`; no GPU is needed.
- A test that builds its own `WaylandDisplay` must let `Drop` join the compositor thread (its
  `Quit` send has to succeed); a detached thread races process exit and crashes the test binary
  when it runs alone.
