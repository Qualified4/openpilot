fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(feature = "ipc")]
    {
        // msgq + VisionIPC from msgq_repo (the sources openpilot's SConscript builds) and the C shim.
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let root = manifest.join("../../../../../msgq_repo");
        let services = manifest.join("../../../../cereal/services.py");
        let out_dir = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
        println!("cargo:rerun-if-changed={}", services.display());
        let header = std::process::Command::new("python3").arg(services).output().expect("run cereal services.py");
        assert!(header.status.success(), "cereal services.py: {}", String::from_utf8_lossy(&header.stderr));
        std::fs::write(out_dir.join("services.h"), header.stdout).expect("write services.h");
        // msgq_repo/SConscript picks the ion buffer when the build machine has /dev/ion; a cross build names it
        // (STRIPNET_VISIONBUF=ion|plain, the device's choice).
        println!("cargo:rerun-if-env-changed=STRIPNET_VISIONBUF");
        let ion = match std::env::var("STRIPNET_VISIONBUF").as_deref() {
            Ok("ion") => true,
            Ok("plain") => false,
            Ok(v) => panic!("STRIPNET_VISIONBUF={v}: expected ion or plain"),
            Err(_) => std::path::Path::new("/dev/ion").exists(),
        };
        let vipc_buf = if ion { "visionbuf_ion.cc" } else { "visionbuf.cc" };
        let mut b = cc::Build::new();
        b.cpp(true).std("c++17").include(&root).include(&out_dir).warnings(false).file("ipc/shim.cc");
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
