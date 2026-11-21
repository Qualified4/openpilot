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
    #[cfg(any(feature = "kleidicv", feature = "opencv"))]
    {
        // The static libraries and headers tools/build_native.sh makes (aarch64 only).
        println!("cargo:rerun-if-env-changed=STRIPNET_NATIVE_DIR");
        let dir = std::path::PathBuf::from(std::env::var("STRIPNET_NATIVE_DIR").expect("STRIPNET_NATIVE_DIR: run tools/build_native.sh"));
        let lib = dir.join("lib");
        let mut b = cc::Build::new();
        b.cpp(true).std("c++17").warnings(false).include(dir.join("include")).include(dir.join("include/opencv4"));
        let mut libs = Vec::new();
        if cfg!(feature = "kleidicv") {
            b.file("native/kleidicv.cc");
            println!("cargo:rerun-if-changed=native/kleidicv.cc");
            libs.push("kleidicv");
        }
        if cfg!(feature = "opencv") {
            b.file("native/opencv.cc");
            println!("cargo:rerun-if-changed=native/opencv.cc");
            libs = vec!["opencv_imgproc", "opencv_core", "kleidicv_hal", "kleidicv_thread", "kleidicv", "zlib"];
        }
        b.compile("stripnet_native");
        // OpenCV core and the KleidiCV HAL refer to each other: one group.
        println!("cargo:rustc-link-arg=-Wl,--start-group");
        for l in libs {
            println!("cargo:rustc-link-arg={}", lib.join(format!("lib{l}.a")).display());
        }
        println!("cargo:rustc-link-arg=-Wl,--end-group");
    }
}
