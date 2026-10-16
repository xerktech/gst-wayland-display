pub mod allocator;
pub mod device;
pub mod renderer;
pub mod video_info;

mod target;

pub use self::target::*;

pub mod tests {
    #[cfg(test)]
    use smithay::backend::drm::DrmNode;
    use std::sync::Once;
    pub static INIT: Once = Once::new();

    #[cfg(test)]
    pub fn test_init() -> () {
        INIT.call_once(|| {
            tracing_subscriber::fmt::try_init().ok();
            gst::init().expect("Failed to initialize GStreamer");
        });
    }

    /// A render node whose kernel driver is one of `drivers`, for hardware-gated
    /// tests -- so they target the right GPU on a multi-GPU host instead of a
    /// hardcoded `renderD12x` that may be a different vendor.
    #[cfg(test)]
    pub fn pick_render_node(drivers: &[&str]) -> Option<DrmNode> {
        for entry in std::fs::read_dir("/dev/dri").ok()?.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if !name.starts_with("renderD") {
                continue;
            }
            let drv = std::fs::read_to_string(format!("/sys/class/drm/{name}/device/uevent"))
                .ok()
                .and_then(|s| {
                    s.lines()
                        .find_map(|l| l.strip_prefix("DRIVER=").map(str::to_owned))
                })
                .unwrap_or_default();
            if drivers.iter().any(|d| *d == drv) {
                if let Ok(node) = DrmNode::from_path(format!("/dev/dri/{name}")) {
                    return Some(node);
                }
            }
        }
        None
    }
}
