fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(feature = "ipc")]
    {
        // msgq + VisionIPC from msgq_repo (the sources openpilot's SConscript builds) and the C shim.
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../../msgq_repo");
        let vipc_buf = if std::path::Path::new("/dev/ion").exists() { "visionbuf_ion.cc" } else { "visionbuf.cc" };
        let mut b = cc::Build::new();
        b.cpp(true).std("c++17").include(&root).warnings(false).file("ipc/shim.cc");
        for f in ["msgq/ipc.cc", "msgq/event.cc", "msgq/impl_msgq.cc", "msgq/impl_fake.cc", "msgq/msgq.cc"] {
            b.file(root.join(f));
        }
        for f in ["visionipc.cc", "visionipc_server.cc", "visionipc_client.cc", vipc_buf] {
            b.file(root.join("msgq/visionipc").join(f));
        }
        b.compile("stripnet_ipc");
        println!("cargo:rerun-if-changed=ipc/shim.cc");
    }
}
