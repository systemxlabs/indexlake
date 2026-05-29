use std::sync::LazyLock;

use jieba_rs::Jieba;

static JIEBA: LazyLock<Jieba> = LazyLock::new(Jieba::new);

#[derive(Debug)]
pub struct JiebaTokenizer;

impl bm25::Tokenizer for JiebaTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        JIEBA
            .cut_for_search(text, true)
            .into_iter()
            .map(|token| token.word.to_string())
            .collect()
    }
}

impl Default for JiebaTokenizer {
    fn default() -> Self {
        Self
    }
}
