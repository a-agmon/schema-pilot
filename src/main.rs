mod llm;
mod vecdb;
use yaml_rust2::{YamlEmitter, YamlLoader};

fn main() -> anyhow::Result<()> {
    init_tracing();
    read_tables("assets/schemas.yaml")?;
    Ok(())
}

fn read_tables(path: &str) -> anyhow::Result<()> {
    // read the file from the assets folder
    let file = std::fs::read_to_string(path)?;
    let docs = YamlLoader::load_from_str(&file)?;
    let schema = docs[0].clone();
    println!("{:?}", schema);
    Ok(())
}

fn init_tracing() {
    tracing_subscriber::fmt::init();
}
