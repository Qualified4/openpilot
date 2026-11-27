use sha2::{Digest, Sha256};
use std::{fs, io, path::Path};

fn collect_sources(root: &Path, dir: &Path, paths: &mut Vec<String>) -> io::Result<()> {
    for entry in fs::read_dir(root.join(dir))? {
        let entry = entry?;
        let path = dir.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            collect_sources(root, &path, paths)?;
        } else if entry.path().is_file() {
            paths.push(
                path.to_str()
                    .expect("source paths must be UTF-8")
                    .replace(std::path::MAIN_SEPARATOR, "/"),
            );
        }
    }
    Ok(())
}

pub fn emit_source_hash(root: &Path) -> io::Result<()> {
    let mut paths = vec![
        "Cargo.toml".into(),
        "Cargo.lock".into(),
        "build.rs".into(),
        "ccnc_map/Cargo.toml".into(),
        "det_rules/Cargo.toml".into(),
    ];
    for watched in paths.iter().map(String::as_str).chain(["src", "ccnc_map/src", "det_rules/src"]) {
        println!("cargo:rerun-if-changed={}", root.join(watched).display());
    }
    collect_sources(root, Path::new("src"), &mut paths)?;
    collect_sources(root, Path::new("ccnc_map/src"), &mut paths)?;
    collect_sources(root, Path::new("det_rules/src"), &mut paths)?;
    paths.sort_unstable();

    // Keep in sync with _source_hash in ccnc_core_loader.py.
    let mut hash = Sha256::new();
    for path in paths {
        let contents = fs::read(root.join(&path))?;
        for field in [path.as_bytes(), contents.as_slice()] {
            hash.update((field.len() as u64).to_be_bytes());
            hash.update(field);
        }
    }
    println!("cargo:rustc-env=CCNC_CORE_SOURCE_HASH={:x}", hash.finalize());
    Ok(())
}

fn main() -> io::Result<()> {
    emit_source_hash(Path::new("."))
}
