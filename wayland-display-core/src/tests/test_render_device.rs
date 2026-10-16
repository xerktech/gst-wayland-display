//! `get_render_device` must answer without a compositor round trip: the
//! streaming thread calls it under the pad stream lock during renegotiation,
//! so a compositor that never answers must not be able to wedge it.

use crate::utils::device::gpu::GPUDevice;
use crate::{Command, MaybeRecv, WaylandDisplay, channel};
use smithay::backend::drm::DrmNode;
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
        render_device: None,
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

fn constructed_render_devices(render_node: &str) -> [Option<GPUDevice>; 2] {
    let display = WaylandDisplay::new(Some(render_node.into())).unwrap();
    let (command_tx, command_rx) = channel::<Command>();
    let display_with_channel =
        WaylandDisplay::new_with_channel(Some(render_node.into()), command_tx, command_rx)
            .unwrap();
    [display.get_render_device(), display_with_channel.get_render_device()]
}

#[test]
fn software_render_target_has_no_render_device() {
    assert_eq!(constructed_render_devices("software"), [None, None]);
}

#[test]
#[ignore = "needs a GPU render node; run via ci/harness.sh gpu"]
fn hardware_render_target_resolves_its_render_device() {
    let path = "/dev/dri/renderD128";
    let expected = GPUDevice::try_from(DrmNode::from_path(path).unwrap()).unwrap();
    assert_eq!(constructed_render_devices(path), [Some(expected.clone()), Some(expected)]);
}
