pub fn GetSchemaPrompt(table_definition: &str, user_query: &str) -> String {
    format!(
        r#" 
   <|start_header_id|>system<|end_header_id|>
    You are an AI assistant that answers questions about database schemas and tables. 
   <|eot_id|><|start_header_id|>user<|end_header_id|>
    These are the tables in our database:
    {table_definition}
    ----------
    Based on the tables provided above, answer the user's question concisely and directly. 
    Always include relevant information about the relevant tables and their purpose. And, if appropriate, provide an SQL query to retrieve the data. 
    
    User question: {user_query}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>"#
    )
}
