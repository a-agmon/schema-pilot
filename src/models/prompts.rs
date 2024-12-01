pub fn GetSchemaPrompt(table_definition: &str, user_query: &str) -> String {
    format!(
        r#" 
   <|start_header_id|>system<|end_header_id|>
    You are an AI assistant that answers questions about database schemas and tables. 
    Your answer should include relevant information about the relevant tables, columns and their purpose.
    Here are the tables in our database:
    {table_definition}
     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Based on the tables in our database, please answer the following question concisely and directly:
    {user_query}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    "#
    )
}
