//! Calls that wait on a compositor reply must give up when the compositor never
//! answers (e.g. its event loop is wedged): the streaming thread makes them under
//! the pad stream lock, so an unbounded wait wedges the whole element.

use crate::{Command, MaybeRecv, WaylandDisplay, channel};
use std::sync::mpsc;
use std::time::Duration;

fn unanswered_display() -> WaylandDisplay {
    // A command channel nobody services: the compositor is alive (the receiver
    // exists) but never answers. Leak the receiver so it outlives the test thread.
    let (command_tx, unserviced_rx) = channel::<Command>();
    std::mem::forget(unserviced_rx);
    let (_devices_tx, devices_rx) = mpsc::channel();
    let (_envs_tx, envs_rx) = mpsc::channel();
    WaylandDisplay {
        thread_handle: Some(std::thread::spawn(|| {})),
        command_tx,
        reply_timeout: Duration::from_millis(200),
        tracer: None,
        devices: MaybeRecv::Rx(devices_rx),
        envs: MaybeRecv::Rx(envs_rx),
    }
}

fn returns_within<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> Option<T> {
    let (done_tx, done_rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = done_tx.send(f());
    });
    done_rx.recv_timeout(Duration::from_secs(5)).ok()
}

#[test]
fn get_supported_dma_formats_errors_on_unanswered_compositor() {
    let display = unanswered_display();
    let result = returns_within(move || display.get_supported_dma_formats().is_err())
        .expect("get_supported_dma_formats blocked on an unanswered compositor");
    assert!(
        result,
        "an unanswered query must be an error, not empty formats"
    );
}

#[test]
fn frame_errors_on_unanswered_compositor() {
    let display = unanswered_display();
    let result =
        returns_within(move || display.frame()).expect("frame blocked on an unanswered compositor");
    assert_eq!(result.unwrap_err(), gst::FlowError::Error);
}

#[test]
fn get_supported_dma_formats_errors_when_compositor_is_gone() {
    // The compositor thread died: its receiver is dropped. Must not panic.
    let mut display = unanswered_display();
    let (command_tx, rx) = channel::<Command>();
    drop(rx);
    display.command_tx = command_tx;
    let result = returns_within(move || display.get_supported_dma_formats().is_err())
        .expect("get_supported_dma_formats panicked or blocked on a dead compositor");
    assert!(result);
}
