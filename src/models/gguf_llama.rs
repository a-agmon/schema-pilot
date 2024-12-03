use std::io::Write;

use anyhow::{Error, Result};

use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::{
        self,
        llama::{self as llama_model, Config, LlamaEosToks},
        quantized_llama::ModelWeights,
    },
};

use candle_core::{quantized::gguf_file, DType, Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{
    api::sync::{Api, ApiBuilder},
    Repo, RepoType,
};
use llama_model::{Llama, LlamaConfig};
use tokenizers::Tokenizer;
use tokio::sync::mpsc;

use crate::models::token_output_stream::TokenOutputStream;

pub struct GGUFLLama {
    model: ModelWeights,
    tokenizer: Tokenizer,
    device: Device,
}

impl GGUFLLama {
    pub fn load(model_file: &str, tokenizer_file: &str, device: Device) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(anyhow::Error::msg)?;
        let model_path = std::path::PathBuf::from(model_file);
        let mut model_file = std::fs::File::open(&model_path)?;
        let model =
            gguf_file::Content::read(&mut model_file).map_err(|e| e.with_path(model_path))?;
        let model_weights = ModelWeights::from_gguf(model, &mut model_file, &device)?;
        Ok(Self {
            model: model_weights,
            tokenizer,
            device,
        })
    }

    pub fn generate_with_defaults(&mut self, prompt_str: &str, sample_len: usize) -> Result<()> {
        self.generate(
            prompt_str,
            sample_len,
            1.1,
            Some(50),
            Some(0.95),
            42,
            1.0,
            64,
        )
    }

    pub fn generate(
        &mut self,
        prompt_str: &str,
        sample_len: usize,
        temperature: f64,
        top_k: Option<usize>,
        top_p: Option<f64>,
        seed: u64,
        repeat_penalty: f32,
        repeat_last_n: usize,
    ) -> Result<()> {
        let mut tos = TokenOutputStream::new(self.tokenizer.clone());
        let mut pre_prompt_tokens = vec![];
        let tokens = tos
            .tokenizer()
            .encode(prompt_str, true)
            .map_err(anyhow::Error::msg)?;
        let prompt_tokens = [&pre_prompt_tokens, tokens.get_ids()].concat();
        let to_sample = sample_len.saturating_sub(1);
        let max_seq_len = models::quantized_llama::MAX_SEQ_LEN;
        let prompt_tokens = if prompt_tokens.len() + to_sample > max_seq_len - 10 {
            let to_remove = prompt_tokens.len() + to_sample + 10 - max_seq_len;
            prompt_tokens[prompt_tokens.len().saturating_sub(to_remove)..].to_vec()
        } else {
            prompt_tokens
        };
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
            let input = Tensor::new(prompt_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, 0)?;
            let logits = logits.squeeze(0)?;
            logits_processor.sample(&logits)?
        };
        all_tokens.push(next_token);
        if let Some(t) = tos.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }
        let eos_token = tos.tokenizer().token_to_id("<|end_of_text|>").unwrap();
        let mut sampled = 0;
        for index in 0..to_sample {
            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, prompt_tokens.len() + index)?;
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
            if let Some(t) = tos.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
            sampled += 1;
            if next_token == eos_token {
                break;
            };
        }
        if let Some(rest) = tos.decode_rest().map_err(Error::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_with_defaults() -> Result<()> {
        let mut model = GGUFLLama::load(
            "/home/alonagmon/gguf/Llama-3.2-3B-Instruct.Q4_K_M.gguf",
            "/home/alonagmon/gguf/tokenizer.json",
            Device::Cpu,
        )?;
        model.generate_with_defaults("Hello, world!", 100)?;
        Ok(())
    }
}
