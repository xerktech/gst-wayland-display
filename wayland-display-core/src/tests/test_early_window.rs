//! A client that creates its toplevel before the output exists (i.e. before caps
//! negotiation runs `apply_video_info`) must still get configured and mapped once the
//! output appears, instead of having its window silently dropped.

use crate::comp::apply_video_info;
use crate::tests::fixture::Fixture;
use crate::utils::RenderTarget;
use crate::utils::video_info::GstVideoInfo;
use gst_video::{VideoFormat, VideoInfo};
use test_log::test;

fn video_info() -> GstVideoInfo {
    GstVideoInfo::RAW(
        VideoInfo::builder(VideoFormat::Rgba, 320, 240)
            .build()
            .unwrap(),
    )
}

#[test]
fn window_committed_before_output_is_mapped() {
    let mut f = Fixture::new_without_output();
    assert!(f.server.output.is_none());

    // Initial (bufferless) commit with no output: the compositor can't size the
    // configure yet, so it must hold the window rather than answer or drop it.
    f.client.create_window();
    f.round_trip();
    assert_eq!(f.client.configure_count(), 0);

    apply_video_info(&mut f.server, video_info(), &RenderTarget::Software, None);
    f.round_trip();
    f.round_trip();
    assert_eq!(
        f.client.configure_count(),
        1,
        "the held window must receive its initial configure once the output exists"
    );

    f.client.setup_window(320, 240);
    f.round_trip();
    f.round_trip();

    assert_eq!(f.server.space.elements().count(), 1);
    assert!(f.server.pending_windows.is_empty());
    assert!(
        f.server
            .seat
            .get_keyboard()
            .unwrap()
            .current_focus()
            .is_some(),
        "the mapped window must get keyboard focus"
    );
}

#[test]
fn window_destroyed_before_output_is_forgotten() {
    let mut f = Fixture::new_without_output();

    f.client.create_window();
    f.round_trip();
    f.client.destroy_window();
    f.round_trip();

    apply_video_info(&mut f.server, video_info(), &RenderTarget::Software, None);
    f.round_trip();

    assert!(f.server.pending_windows.is_empty());
    assert_eq!(f.server.space.elements().count(), 0);
}
