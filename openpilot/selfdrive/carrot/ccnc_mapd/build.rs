use std::{fs, io, path::Path};

#[allow(dead_code)]
#[path = "../../../../opendbc_repo/opendbc/car/hyundai/ccnc_core/build.rs"]
mod core_build;

fn main() -> io::Result<()> {
    // One SCHED_FIFO step below plannerd and radard (Priority.CTRL_LOW) on their core 4: the display-only map
    // never delays the control loops, and still runs ahead of the normal threads there.
    let realtime = "../../../common/realtime.py";
    println!("cargo:rerun-if-changed={realtime}");
    let source = fs::read_to_string(realtime)?;
    let priority = source
        .lines()
        .find_map(|line| line.trim().strip_prefix("CTRL_LOW ="))
        .and_then(|value| value.split_whitespace().next())
        .and_then(|value| value.parse::<i32>().ok())
        .ok_or_else(|| io::Error::other("Priority.CTRL_LOW missing from realtime.py"))?;
    println!("cargo:rustc-env=MAPD_RT_PRIORITY={}", priority - 1);
    core_build::emit_source_hash(Path::new("../../../../opendbc_repo/opendbc/car/hyundai/ccnc_core"))
}
