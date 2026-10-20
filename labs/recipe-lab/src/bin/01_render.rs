use deepseek_recipe::openai::ChatCompletionRequest;
use deepseek_recipe::request::{ConversionOptions, ProtocolRequest};
use deepseek_recipe_encoding::v4::dsv41::DeepseekV41Encoding;
use deepseek_recipe_encoding::{PromptEncoding, PromptTokenizer};
use deepseek_recipe_lab::load_tokenizer;
use serde_json::json;
use tokenizers::Tokenizer;

fn show(label: &str, body: serde_json::Value, tok: &Tokenizer) {
    let req: ChatCompletionRequest = serde_json::from_value(body).unwrap();
    let converted = req.convert(ConversionOptions::default()).unwrap();
    let enc = DeepseekV41Encoding::new().with_tokenizer(PromptTokenizer::new(tok));
    let rendered = enc.render_conversation(&converted.conversation);
    let ids = enc.encode(&converted.conversation).unwrap();
    println!("===== {label} =====");
    println!("thinking_mode = {}", converted.conversation.thinking_mode);
    println!("prompt_tokens = {}, special_ranges = {}", ids.len(), rendered.special_ranges.len());
    println!("{}", rendered.prompt.replace('｜', "|"));
    println!();
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let tok = load_tokenizer();
    show("1. plain chat", json!({
        "model": "DeepSeek-V4.1-Flash",
        "messages": [{"role": "user", "content": "Hello"}],
    }), &tok);
    show("2. thinking off", json!({
        "model": "DeepSeek-V4.1-Flash",
        "messages": [{"role": "user", "content": "Hello"}],
        "thinking": {"type": "disabled"},
    }), &tok);
    show("3. tools + json_schema", json!({
        "model": "DeepSeek-V4.1-Flash",
        "messages": [{"role": "system", "content": "You are terse."},
                     {"role": "user", "content": "weather in Beijing?"}],
        "tools": [{"type": "function", "function": {
            "name": "get_weather",
            "description": "Look up weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        }}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "w", "schema": {"type": "object", "properties": {"t": {"type": "number"}}}
        }},
    }), &tok);
    Ok(())
}
