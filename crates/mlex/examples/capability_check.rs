//! Prints `supports_images`/`supports_audio` for a model dir, plus the
//! `config.json` capability declarations, for auditing detection accuracy:
//! `cargo run --release --example capability_check -- <model_dir>`

use std::path::PathBuf;

use mlex::generate::Session;

fn main() {
    let model_dir = PathBuf::from(std::env::args().nth(1).expect("usage: <model_dir>"));
    let config_path = model_dir.join("config.json");
    let raw = std::fs::read_to_string(&config_path).expect("read config.json");
    let json: serde_json::Value = serde_json::from_str(&raw).expect("parse config.json");
    let declares_vision = json.get("vision_config").is_some();
    let declares_audio = json.get("audio_config").is_some();
    let model_type = json
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("?");

    match Session::load(&model_dir) {
        Ok(session) => {
            println!(
                "{}\tmodel_type={}\tdeclares_vision={}\tdeclares_audio={}\tsupports_images={}\tsupports_audio={}",
                model_dir.display(),
                model_type,
                declares_vision,
                declares_audio,
                session.supports_images(),
                session.supports_audio(),
            );
        }
        Err(e) => {
            println!(
                "{}\tmodel_type={}\tdeclares_vision={}\tdeclares_audio={}\tLOAD FAILED: {e}",
                model_dir.display(),
                model_type,
                declares_vision,
                declares_audio,
            );
        }
    }
}
