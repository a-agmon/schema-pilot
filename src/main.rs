use futures::{future, stream, StreamExt};
use std::{collections::HashMap, rc::Rc, sync::Arc};
use vecdb::VecDB;
use vectors::VectorsOps;
mod llm;
mod vecdb;
mod vectors;
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();
    let vecdb = VecDB::create_or_open("runtime_assets/vecdb", "tables", Some(768)).await?;
    let tables = read_tables_definitions("assets/schemas.yaml", "\n\n")?;
    generate_vecdb_from_table_definitions(tables, &vecdb).await?;
    Ok(())
}

async fn generate_vecdb_from_table_definitions(
    tables: Vec<String>,
    vecdb: &VecDB,
) -> anyhow::Result<()> {
    let table_vectors = generate_vectors_from_strs(tables.clone()).await?;
    let tables_names: Rc<Vec<String>> = Rc::new(
        tables
            .iter()
            .map(|t| t.split("\n").next().unwrap().to_string())
            .collect(),
    );

    let table_names_vectors = generate_vectors_from_strs(tables_names.as_ref().clone()).await?;
    let joined_vectors = table_vectors
        .iter()
        .zip(table_names_vectors.iter())
        .map(|(tbl_def, tbl_names)| {
            VectorsOps::weighted_average(tbl_def, 1.0, tbl_names, 2.0, true)
        })
        .collect::<Vec<Vec<f32>>>();
    let empty_vec = vec![""; tables_names.len()];
    let table_refs: Vec<&str> = tables_names.iter().map(|s| s.as_str()).collect();
    let table_details: Vec<&str> = tables.iter().map(|s| s.as_str()).collect();
    vecdb
        .add_vector(&table_refs, &empty_vec, &table_details, joined_vectors, 768)
        .await?;
    Ok(())
}

fn read_tables_definitions(path: &str, split_by: &str) -> anyhow::Result<Vec<String>> {
    // read the file from the assets folder
    let file = std::fs::read_to_string(path)?;
    // split the file into lines
    let tables = file
        .split(split_by)
        .map(|s| s.to_string())
        .collect::<Vec<String>>();
    Ok(tables)
}

async fn generate_vectors_from_strs(tables: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
    let local_llm = llm::LlmClient::new("http://localhost", 11434);
    local_llm
        .generate_embedding(llm::EmbeddingModel::NomicText, tables)
        .await
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
