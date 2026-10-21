---
paths:
  - "gst-plugin-wayland-display/src/waylandsrc/**"
  - "wayland-display-core/src/lib.rs"
  - "wayland-display-core/src/comp/mod.rs"
---

# waylanddisplaysrc unlock() and frame waits

- Never drop a `PendingFrame` the compositor still owes while the display lives: the compositor
  sets `should_quit` when a frame send fails, so the stream dies. Stash it and wait on it again.
- `request_frame` uses `sync_channel(1)`: with a rendezvous channel the compositor blocks
  delivering a stashed frame, and PAUSED->NULL hangs for the reply timeout.
- unlock() must ignore a live PLAYING->PAUSED: a create() cancelled by it can return FLUSHING
  after a quick resume restarted the task, which stops it for good (GstBaseSrc defers
  unlock_stop() to the next create()). Flush and stop still cancel.
- A pad-level FLUSH_START skips the state lock, so it can land inside that pause window: `event()`
  sets `flushing` itself rather than relying on unlock().
  - Set it before `parent_event()`, which blocks on LIVE_LOCK until the pause ends; no unit test
    catches the wrong order.
- Tests: `frames_flow_after_flushes`, `frames_flow_after_quick_pause_resume`,
  `unlock_ends_create_waiting_for_frame`, `pad_flush_during_pause_ends_create_waiting_for_frame`
  in waylandsrc/imp.rs.
