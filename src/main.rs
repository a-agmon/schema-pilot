use futures::{future, stream, StreamExt};
use std::sync::Arc;
mod llm;
mod vecdb;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();
    let tables = read_tables_definitions("assets/schemas.yaml")?;
    let table_taggings = generate_table_tagging(tables).await?;
    table_taggings.iter().for_each(|tag| {
        println!("\n---\n{}", tag);
    });
    Ok(())
}

fn read_tables_definitions(path: &str) -> anyhow::Result<Vec<String>> {
    // read the file from the assets folder
    let file = std::fs::read_to_string(path)?;
    // split the file into lines
    let tables = file
        .split("\n\n")
        .map(|s| s.to_string())
        .collect::<Vec<String>>();
    Ok(tables)
}

// the function receives a vector of table definitions and generates for each table
// a string that is made of tags and strings that are used to tag the table and represent the table schema
async fn generate_table_tagging(tables: Vec<String>) -> anyhow::Result<Vec<String>> {
    let prompt_str = r#"
    The following is a database table definition:
    {table_definition}
    Instructions:
    Generate a string that is made of tags and strings that are used to tag the table and represent the table schema.
    "#;
    let local_llm = Arc::new(llm::LlmClient::new("http://localhost", 11434));

    let results = stream::iter(tables)
        .then(|table| {
            let llm = local_llm.clone();
            let prompt = prompt_str.replace("{table_definition}", &table);
            async move {
                llm.generate_text(llm::LlmModel::Llama323b, &prompt)
                    .await
                    .unwrap()
            }
        })
        .collect::<Vec<_>>()
        .await;
    Ok(results)
}

fn init_tracing() {
    tracing_subscriber::fmt::init();
}
