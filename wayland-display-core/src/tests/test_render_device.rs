//! `get_render_device` must answer without a compositor round trip: the
//! streaming thread calls it under the pad stream lock during renegotiation,
//! so a compositor that never answers must not be able to wedge it.

use crate::{Command, MaybeRecv, WaylandDisplay, channel};
use std::sync::mpsc;
use std::time::Duration;

#[test]
fn get_render_device_does_not_wait_on_compositor() {
    // A command channel nobody services: the compositor is alive (the
    // receiver exists) but never answers, as when its event loop is wedged.
    let (command_tx, _unserviced_rx) = channel::<Command>();
    let (_devices_tx, devices_rx) = mpsc::channel();
    let (_envs_tx, envs_rx) = mpsc::channel();
    let display = WaylandDisplay {
        thread_handle: Some(std::thread::spawn(|| {})),
        command_tx,
        tracer: None,
        devices: MaybeRecv::Rx(devices_rx),
        envs: MaybeRecv::Rx(envs_rx),
    };

    let (done_tx, done_rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = done_tx.send(display.get_render_device());
    });
    assert!(
        done_rx.recv_timeout(Duration::from_secs(5)).is_ok(),
        "get_render_device blocked on an unanswered compositor"
    );
}
