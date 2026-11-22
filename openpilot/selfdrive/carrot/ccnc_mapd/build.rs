use std::{fs, io};

fn main() -> io::Result<()> {
    // Keep the daemon at the same SCHED_FIFO priority as radarcan (Priority.CTRL_LOW).
    let realtime = "../../../common/realtime.py";
    println!("cargo:rerun-if-changed={realtime}");
    let source = fs::read_to_string(realtime)?;
    let priority = source
        .lines()
        .find_map(|line| line.trim().strip_prefix("CTRL_LOW ="))
        .and_then(|value| value.split_whitespace().next())
        .and_then(|value| value.parse::<i32>().ok())
        .ok_or_else(|| io::Error::other("Priority.CTRL_LOW missing from realtime.py"))?;
    println!("cargo:rustc-env=MAPD_RT_PRIORITY={priority}");
    Ok(())
}
