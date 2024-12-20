use candle_transformers::models::mimi::encodec::ResampleMethod;
use futures::StreamExt;
use ollama_rs::{
    generation::{
        chat::ChatMessage,
        completion::request::GenerationRequest,
        embeddings::request::{EmbeddingsInput, GenerateEmbeddingsRequest},
        options::GenerationOptions,
    },
    Ollama,
};
use tokio::sync::mpsc;
pub struct LlmClient {
    ollama_client: Ollama,
}
pub enum LlmModel {
    Llama323b,
}
pub enum EmbeddingModel {
    NomicText,
}
impl LlmModel {
    pub fn to_string(&self) -> String {
        match self {
            LlmModel::Llama323b => String::from("llama3.2"),
        }
    }
}
impl EmbeddingModel {
    pub fn to_string(&self) -> String {
        match self {
            EmbeddingModel::NomicText => String::from("nomic-embed-text"),
        }
    }
}

impl LlmClient {
    pub fn new(host: &str, port: u16) -> Self {
        let ollama_client = Ollama::new(host, port);
        Self { ollama_client }
    }

    pub async fn generate_text_stream(
        &self,
        model: LlmModel,
        prompt: &str,
        stream_channel: Option<mpsc::Sender<String>>,
    ) -> anyhow::Result<()> {
        let options = GenerationOptions::default()
            .temperature(0.2)
            .repeat_penalty(1.5)
            .top_k(25)
            .top_p(0.25);
        let mut stream = self
            .ollama_client
            .generate_stream(
                GenerationRequest::new(model.to_string(), prompt.to_string()).options(options),
            )
            .await
            .unwrap();

        while let Some(res) = stream.next().await {
            let responses = res?;
            for response in responses {
                let response_text = response.response;
                if let Some(ref stream_channel) = stream_channel {
                    stream_channel.send(response_text).await?;
                }
            }
        }
        Ok(())
    }

    pub async fn generate_text(&self, model: LlmModel, prompt: &str) -> anyhow::Result<String> {
        let gen_res = self
            .ollama_client
            .generate(GenerationRequest::new(
                model.to_string(),
                prompt.to_string(),
            ))
            .await?;
        Ok(gen_res.response)
    }

    pub async fn generate_chat(
        &self,
        model: LlmModel,
        messages: Vec<ChatMessage>,
    ) -> anyhow::Result<String> {
        todo!()
    }

    pub async fn generate_embedding(
        &self,
        model: EmbeddingModel,
        text: Vec<String>,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        let embeddings_input = EmbeddingsInput::Multiple(text);
        let embed_res = self
            .ollama_client
            .generate_embeddings(GenerateEmbeddingsRequest::new(
                model.to_string(),
                embeddings_input,
            ))
            .await?;
        Ok(embed_res.embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_generate_text() {
        let llm_client = LlmClient::new("http://localhost", 11434);
        let start = std::time::Instant::now();
        let res = llm_client
            .generate_text(LlmModel::Llama323b, "Whats the capital of france?")
            .await;
        let pred1 = std::time::Instant::now();
        let res = res.unwrap();
        println!("{}", res);
        println!("Time taken: {:?}", pred1.duration_since(start));
        assert!(res.contains("Paris"));
    }
    #[tokio::test]
    async fn test_generate_embedding() {
        let llm_client = LlmClient::new("http://localhost", 11434);
        let texts = vec![
            "This is the first test sentence".to_string(),
            "This is another test sentence".to_string(),
            "And here is a third one".to_string(),
        ];

        let start = std::time::Instant::now();
        let embeddings = llm_client
            .generate_embedding(EmbeddingModel::NomicText, texts)
            .await
            .unwrap();
        let pred1 = std::time::Instant::now();

        println!("Time taken: {:?}", pred1.duration_since(start));
        println!("Number of embeddings: {}", embeddings.len());
        println!("Embedding dimension: {}", embeddings[0].len());

        // Basic validation
        assert_eq!(embeddings.len(), 3); // Should match number of input texts
        assert!(embeddings[0].len() > 0); // Embeddings should not be empty
        assert!(embeddings[0].len() == embeddings[1].len()); // All embeddings should have same dimension
    }
}
