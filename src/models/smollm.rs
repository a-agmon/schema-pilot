use std::io::Write;
use tokenizers::Tokenizer;

use crate::models::token_output_stream::TokenOutputStream;
use candle_core::Tensor;
use candle_core::{quantized::gguf_file, Device};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_llama as model;
use model::ModelWeights;

pub struct SmolLM2 {
    model: ModelWeights,
    //tokenizer: Tokenizer,
    tos: TokenOutputStream,
}

const MODEL_REPO: &str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"; //tokenizer
const WEIGHTS_REPO: &str = "bartowski/SmolLM2-1.7B-Instruct-GGUF";
const WEIGHTS_FILE: &str = "SmolLM2-1.7B-Instruct-Q6_K_L.gguf";
impl SmolLM2 {
    fn get_tokenizer(repo: String) -> Result<Tokenizer, anyhow::Error> {
        let api = hf_hub::api::sync::Api::new()?;
        let api = api.model(repo);
        let tokenizer_path = api.get("tokenizer.json")?;
        Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)
    }
    fn get_model_path(
        repo: &str,
        revision: &str,
        filename: &str,
    ) -> anyhow::Result<std::path::PathBuf> {
        let api = hf_hub::api::sync::Api::new()?;
        let path = api
            .repo(hf_hub::Repo::with_revision(
                repo.to_string(),
                hf_hub::RepoType::Model,
                revision.to_string(),
            ))
            .get(filename)?;
        Ok(path)
    }

    pub fn load() -> Result<Self, anyhow::Error> {
        let model_path = Self::get_model_path(WEIGHTS_REPO, "main", WEIGHTS_FILE)?;
        let mut file = std::fs::File::open(&model_path)?;
        let mut model = {
            let model = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(model_path))?;
            ModelWeights::from_gguf(model, &mut file, &Device::Cpu)?
        };
        let tokenizer = Self::get_tokenizer(MODEL_REPO.to_string())?;
        let mut tos = TokenOutputStream::new(tokenizer);
        Ok(Self { model, tos })
    }

    pub fn generate(
        &mut self,
        prompt_str: &str,
        temperature: f64,
        top_k: Option<usize>,
        top_p: Option<f64>,
        seed: u64,
        sample_len: usize,
        repeat_penalty: f32,
        repeat_last_n: usize,
    ) -> Result<String, anyhow::Error> {
        let tokens = self
            .tos
            .tokenizer()
            .encode(prompt_str, true)
            .map_err(anyhow::Error::msg)?;
        let tokens = tokens.get_ids();
        let to_sample = sample_len.saturating_sub(1);
        let mut all_tokens = vec![];
        let mut logits_processor = {
            let temperature = temperature;
            let sampling = if temperature <= 0. {
                Sampling::ArgMax
            } else {
                match (top_k, top_p) {
                    (None, None) => Sampling::All { temperature },
                    (Some(k), None) => Sampling::TopK { k, temperature },
                    (None, Some(p)) => Sampling::TopP { p, temperature },
                    (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
                }
            };
            LogitsProcessor::from_sampling(seed, sampling)
        };
        let mut next_token = {
            let input = Tensor::new(tokens, &Device::Cpu)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, 0)?;
            let logits = logits.squeeze(0)?;
            logits_processor.sample(&logits)?
        };
        let mut generated_buffer: Vec<String> = Vec::new();
        all_tokens.push(next_token);
        if let Some(t) = self.tos.next_token(next_token)? {
            print!("{t}");
            generated_buffer.push(t);
        }
        let eos_token = *self
            .tos
            .tokenizer()
            .get_vocab(true)
            .get("<|endoftext|>")
            .unwrap();
        for index in 0..to_sample {
            let input = Tensor::new(&[next_token], &Device::Cpu)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, tokens.len() + index)?;
            let logits = logits.squeeze(0)?;
            let logits = if repeat_penalty == 1. {
                logits
            } else {
                let start_at = all_tokens.len().saturating_sub(repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    repeat_penalty,
                    &all_tokens[start_at..],
                )?
            };
            next_token = logits_processor.sample(&logits)?;
            all_tokens.push(next_token);
            if let Some(t) = self.tos.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
                generated_buffer.push(t);
            }
            if next_token == eos_token {
                break;
            };
        }
        if let Some(rest) = self.tos.decode_rest().map_err(candle_core::Error::msg)? {
            print!("{rest}");
        }
        Ok(generated_buffer.join(""))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_generate() {
        let mut model = SmolLM2::load().unwrap();
        let result = model
            .generate(
                "What is the capital of France?",
                0.7,
                Some(5),
                Some(0.9),
                299792458,
                80,
                1.2,
                64,
            )
            .unwrap();
        println!("{result}");
    }
}
