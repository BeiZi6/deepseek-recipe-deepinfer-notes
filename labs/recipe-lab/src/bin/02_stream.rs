use deepseek_recipe::openai::ChatCompletionRequest;
use deepseek_recipe::request::{ConversionOptions, ProtocolRequest};
use deepseek_recipe::stream::{InferenceChunk, InferenceFinishReason, PromptUsage, StreamProcessor};
use serde_json::json;
use tokio_stream::{iter, StreamExt};

const D: &str = "\u{FF5C}DSML\u{FF5C}";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let body = json!({
        "model": "DeepSeek-V4.1-Flash",
        "messages": [{"role": "user", "content": "weather in Beijing?"}],
        "thinking": {"type": "disabled"},
        "tools": [{"type": "function", "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
        }}],
    });
    let req: ChatCompletionRequest = serde_json::from_value(body)?;
    let converted = req.convert(ConversionOptions::default())?;
    let generator = ChatCompletionRequest::chunk_generator(&converted, "id-2".into(), "DeepSeek-V4.1-Flash".into());
    let processor = StreamProcessor::new(generator, converted.parsing_options);

    let text = format!("Checking.\n\n<{D} calls>\n<{D} invoke name=\"get_weather\">\n<{D} parameter name=\"city\" string=\"true\">Beijing</{D} parameter>\n</{D} invoke>\n</{D} calls>");
    let inference = iter(vec![
        InferenceChunk::Ready { system_fingerprint: None, prompt_usage: PromptUsage { prompt_tokens: 9, prompt_cache_hit_tokens: 0 } },
        InferenceChunk::Text { content: text, token_count: 40 },
        InferenceChunk::Finish { finish_reason: InferenceFinishReason::Stop },
    ]);
    let mut chunks = std::pin::pin!(processor.into_protocol_stream(inference));
    while let Some(chunk) = chunks.next().await {
        println!("{}", serde_json::to_string(&chunk?)?);
    }
    Ok(())
}
