use sha2::{Digest, Sha256};
use std::{fs, io, path::Path};

fn collect_sources(dir: &Path, paths: &mut Vec<String>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            collect_sources(&path, paths)?;
        } else if path.is_file() {
            paths.push(
                path.to_str()
                    .expect("source paths must be UTF-8")
                    .replace(std::path::MAIN_SEPARATOR, "/"),
            );
        }
    }
    Ok(())
}

fn main() -> io::Result<()> {
    let mut paths = vec![
        "Cargo.toml".into(),
        "Cargo.lock".into(),
        "build.rs".into(),
        "ccnc_map/Cargo.toml".into(),
    ];
    for watched in paths.iter().map(String::as_str).chain(["src", "ccnc_map/src"]) {
        println!("cargo:rerun-if-changed={watched}");
    }
    collect_sources(Path::new("src"), &mut paths)?;
    collect_sources(Path::new("ccnc_map/src"), &mut paths)?;
    paths.sort_unstable();

    // Keep in sync with _source_hash in ccnc_core_loader.py.
    let mut hash = Sha256::new();
    for path in paths {
        let contents = fs::read(&path)?;
        for field in [path.as_bytes(), contents.as_slice()] {
            hash.update((field.len() as u64).to_be_bytes());
            hash.update(field);
        }
    }
    println!("cargo:rustc-env=CCNC_CORE_SOURCE_HASH={:x}", hash.finalize());
    Ok(())
}
