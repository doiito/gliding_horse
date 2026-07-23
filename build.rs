use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?);

    tonic_build::configure()
        .build_server(true)
        .build_client(false)
        .compile(&["proto/pdca_core.proto"], &["proto"])?;

    let se_app_proto = manifest_dir.join("apps/software_engineering_single/proto/se_app.proto");
    let se_app_proto_dir = manifest_dir.join("apps/software_engineering_single/proto");

    tonic_build::configure()
        .build_server(true)
        .build_client(false)
        .compile(
            &[se_app_proto.to_str().unwrap()],
            &[se_app_proto_dir.to_str().unwrap()],
        )?;

    println!("cargo:rerun-if-changed=proto/pdca_core.proto");
    println!("cargo:rerun-if-changed=apps/software_engineering_single/proto/se_app.proto");

    Ok(())
}
