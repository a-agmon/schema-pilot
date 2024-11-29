use anyhow::Ok;
use arrow_array::StringArray;
use console::{style, Term};
use futures::{future, stream, StreamExt};
use lancedb::{table, Table};
use rayon::{string, vec};
use tokio::sync::mpsc;
use tracing::info;
use std::{collections::HashMap, rc::Rc, sync::Arc};
use vecdb::VecDB;
use vectors::VectorsOps;
mod llm;
mod vecdb;
mod vectors;
mod embedder;
mod models;


#[derive(Debug, serde::Deserialize)]
pub struct TableContent {
    pub content: String,
}
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();
    let vecdb = VecDB::create_or_open("runtime_assets/vecdb", "idm_health100", Some(384)).await?;
    //create_embedding(&vecdb).await?;
    let mut llama_model = models::Llama321B::load().unwrap();
    let embedding_model = embedder::get_model_reference().unwrap();
    let term = Arc::new(Term::stdout());
    let (tx, mut rx) = mpsc::channel::<String>(100);
    let term_clone = term.clone();
    tokio::spawn(async move {
        while let Some(c) = rx.recv().await {
            if let Err(e) = term_clone.write_str(c.as_str()) {
                eprintln!("Error writing to terminal: {}", e);
                break;
            }
        }
    });
    term.clear_screen()?;
    term.write_line(&format!("{}", style("Hello, I'm SchemaPilot, ask my anything about your data model)").green().bright()))?;
     loop {
        term.write_line(&format!("{}", style("What's on your mind?").bold().yellow()))?;
        let query = term.read_line()?;
        // embed the prompt
        let prompt_embedding = embedding_model.embed_multiple(vec![query.clone()])?;
        //println!("embedding shape: {:?} length: {}", prompt_embedding[0].len(), prompt_embedding.len());
        let prompt_embedding = prompt_embedding[0].clone();
        // search relevant tables
        let tables = vecdb.find_similar_x(prompt_embedding, 10 ).await?;
        let table_contents:Vec<TableContent> = serde_arrow::from_record_batch(&tables)?;
        let context_str = table_contents.iter()
        .map(|t| t.content.clone())
        .collect::<Vec<String>>().join("\n");
        let prompt = models::GetSchemaPrompt(&context_str, &query);
        //println!("prompt: {}", prompt);
        //println!("\n ########### \n");
        llama_model.generate_with_default(&prompt, 0.5, 250, Some(tx.clone())).await?;
        // let _ = llama_model.generate(&prompt,
        //      0.25, 
        //      Some(5), 
        //      Some(0.9),
        //       42,
        //     400,
        //     1.2,
        //     64).unwrap();
    term.write_line("\n ----- \n")?;
    }
    Ok(())
}

async fn create_embedding(vecdb: &VecDB) -> anyhow::Result<()> {
    //embedding ---
    //embed_csv_tables_data("assets/table_definition.csv".to_string()).await?;
    let tables = read_tables_definitions("/home/alonagmon/rust/idm_reader/tables.md", "\n\n")?;
    generate_vecdb_from_table_definitions(tables, &vecdb).await?;
    Ok(())
}

#[derive(Debug, serde::Deserialize)]
struct ColumnRecord {
    #[serde(rename = "TableName")]
    table_name: String,
    #[serde(rename = "TableDesc")]
    table_desc: String,
    #[serde(rename = "ColumnName")]
    column_name: String,
    #[serde(rename = "DataType")]
    data_type: String,
    #[serde(rename = "Description")]
    description: String,
}
impl ColumnRecord {
    pub fn to_string(self) -> String {
        format!(
            "table_name: \"{}\", table_desc: \"{}\", column_name: \"{}\", data_type: \"{}\", description: \"{}\"",
            self.table_name,
            self.table_desc, 
            self.column_name,
            self.data_type,
            self.description
        )
    }
}


async fn embed_csv_tables_data(path: String) -> anyhow::Result<()> {
    let mut def_file = csv::Reader::from_path(path)?;
    let records: Vec<String> = def_file
        .deserialize::<ColumnRecord>()
        .filter_map(Result::ok)
        .map(ColumnRecord::to_string)
        .collect();
    let table_vectors = generate_vectors_from_strs(records.clone(), 25)?;
    let table_defs: Vec<&str> = records.iter().map(String::as_str).collect();
    let empty_vec = vec![""; table_vectors.len()];
    let vecdb = VecDB::create_or_open("runtime_assets/vecdb", "columns", Some(768)).await?;
    vecdb.add_vector(&empty_vec, &empty_vec, &table_defs, table_vectors).await?;
    Ok(())
}

async fn generate_vecdb_from_table_definitions(
    tables: Vec<String>,
    vecdb: &VecDB,
) -> anyhow::Result<()> {
    let table_vectors = generate_vectors_from_strs(tables.clone(), 25)?;
    let tables_names: Rc<Vec<String>> = Rc::new(
        tables
            .iter()
            .map(|t| t.split("\n").next().unwrap().to_string())
            .collect(),
    );

    let table_names_vectors = generate_vectors_from_strs(tables_names.as_ref().clone(), 25)?;
    let joined_vectors = table_vectors
        .iter()
        .zip(table_names_vectors.iter())
        .map(|(tbl_def, tbl_names)| {
            VectorsOps::weighted_average(tbl_def, 1.0, tbl_names, 3.0, true)
        })
        .collect::<Vec<Vec<f32>>>();
    let empty_vec = vec![""; tables_names.len()];
    let table_refs: Vec<&str> = tables_names.iter().map(|s| s.as_str()).collect();
    let table_details: Vec<&str> = tables.iter().map(|s| s.as_str()).collect();
    vecdb
        .add_vector(&table_refs, &empty_vec, &table_details, joined_vectors)
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

use indicatif::ProgressBar;
fn generate_vectors_from_strs(tables: Vec<String>, chunk_size: usize) -> anyhow::Result<Vec<Vec<f32>>> {
    let bar = Arc::new(ProgressBar::new(tables.len() as u64));
    let tables_chunks:Vec<Vec<String>> = tables.chunks(chunk_size)
    .map(|chunk| chunk.to_vec())
    .collect();
    use rayon::prelude::*;
    // use rayon to parallelize the embedding process
    let vectors:Vec<Vec<f32>> = tables_chunks.par_iter().map(|chunk| {
        let result = embedder::get_model_reference()
        .unwrap()
        .embed_multiple(chunk.clone());
        bar.inc(chunk_size as u64);
        result.unwrap()
    }).flatten().collect();    
    bar.finish();
    Ok(vectors)
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
     tracing::subscriber::set_global_default(
        tracing_subscriber::FmtSubscriber::new()
    ).expect("setting default subscriber failed");
}
