use std::io::Write;

use anyhow::{Error, Result};

use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::llama::{self as llama_model, Config, LlamaEosToks},
};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{
    api::sync::{Api, ApiBuilder},
    Repo, RepoType,
};
use llama_model::{Llama, LlamaConfig};
use tokenizers::Tokenizer;
use tokio::sync::mpsc;

use crate::models::token_output_stream::TokenOutputStream;

pub struct Llama321B {
    model: Llama,
    tokenizer: Tokenizer,
    cache: llama_model::Cache,
    config: Config,
}

impl Llama321B {
    pub fn load() -> Result<Self> {
        let token = std::env::var("HUGGINGFACE_TOKEN").ok();
        let api = ApiBuilder::new().with_token(token).build()?;
        let model_id = "meta-llama/Llama-3.2-1B-Instruct".to_string();
        let rev = "main".to_string();
        let api = api.repo(Repo::with_revision(model_id, RepoType::Model, rev));
        let model_filename = vec![api.get("model.safetensors")?];
        let tokenizer_filename = api.get("tokenizer.json")?;
        let config_filename = api.get("config.json")?;
        let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;
        let config = config.into_config(false);
        let cache = llama_model::Cache::new(true, DType::F16, &config, &Device::Cpu)?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&model_filename, DType::F16, &Device::Cpu)?
        };
        let llama = Llama::load(vb, &config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(Error::msg)?;
        Ok(Self {
            model: llama,
            tokenizer,
            cache,
            config,
        })
    }
 pub async fn generate_with_default(
        &mut self,
        prompt: &str,
        temperature: f64,
        sample_len: usize,
        stream_channel: Option<mpsc::Sender<String>>,
    ) -> anyhow::Result<String> {
        self.generate(
            prompt,
            temperature,
            Some(50),
            Some(0.95),
            299792458,
            sample_len,
            1.0,
            64,
            stream_channel,
        )
        .await
    }

    pub async fn generate(
        &mut self,
        prompt: &str,
        temperature: f64,
        top_k: Option<usize>,
        top_p: Option<f64>,
        seed: u64,
        sample_len: usize,
        repeat_penalty: f32,
        repeat_last_n: usize,
        stream_channel: Option<mpsc::Sender<String>>,
    ) -> anyhow::Result<String> {
        let mut generated_buffer: Vec<String> = Vec::new();
        let mut tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!(e))?
            .get_ids()
            .to_vec();
        let mut tokenizer = TokenOutputStream::new(self.tokenizer.clone());
        let mut logits_processor = {
            let temperature = temperature;
            let sampling = if temperature <= 0. {
                Sampling::ArgMax
            } else {
                match (top_k, top_p) {
                    (None, None) => Sampling::All { temperature },
                    (Some(k), None) => Sampling::TopK { k, temperature },
                    (None, Some(p)) => Sampling::TopP { p, temperature },
                    (Some(k), Some(p)) => {
                        Sampling::TopKThenTopP { k, p, temperature }
                    }
                }
            };
            LogitsProcessor::from_sampling(seed, sampling)
        };
        let mut index_pos = 0;
        self.cache = llama_model::Cache::new(
            true,
            DType::F16,
            &self.config,
            &Device::Cpu,
        )?;

        for index in 0..sample_len {
            let (context_size, context_index) =
                if self.cache.use_kv_cache && index > 0 {
                    (1, index_pos)
                } else {
                    (tokens.len(), 0)
                };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &Device::Cpu)?.unsqueeze(0)?;
            let logits =
                self.model.forward(&input, context_index, &mut self.cache)?;
            let logits = logits.squeeze(0)?;
            let logits = if repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    repeat_penalty,
                    &tokens[start_at..],
                )?
            };
            index_pos += ctxt.len();

            let next_token = logits_processor.sample(&logits)?;
            tokens.push(next_token);
            let eos_token = self.eos_token();

            match eos_token {
                Some(llama_model::LlamaEosToks::Single(eos_tok_id))
                    if next_token == eos_tok_id =>
                {
                    break;
                }
                Some(llama_model::LlamaEosToks::Multiple(ref eos_ids))
                    if eos_ids.contains(&next_token) =>
                {
                    break;
                }
                _ => (),
            }
            if let Some(t) = tokenizer.next_token(next_token)? {
                //generated_buffer.push(t);
                Self::add_token(
                    t,
                    &mut generated_buffer,
                    stream_channel.clone(),
                )
                .await?;
            }
        }
        if let Some(rest) =
            tokenizer.decode_rest().map_err(|e| anyhow::anyhow!(e))?
        {
            Self::add_token(
                rest,
                &mut generated_buffer,
                stream_channel.clone(),
            )
            .await?;
        }
        let generated_text = generated_buffer.join("");
        tokenizer.clear();
        Ok(generated_text)
    }

    fn eos_token(&self) -> Option<LlamaEosToks> {
        self.config.eos_token_id.clone().or_else(|| {
            self.tokenizer
                .token_to_id("</s>")
                .map(llama_model::LlamaEosToks::Single)
        })
    }

    async fn add_token(
        token: String,
        buffer: &mut Vec<String>,
        stream_channel: Option<mpsc::Sender<String>>,
    ) -> anyhow::Result<()> {
        buffer.push(token.clone());
        if let Some(ref stream_channel) = stream_channel {
            stream_channel.send(token).await?;
        }
        Ok(())
    }
}