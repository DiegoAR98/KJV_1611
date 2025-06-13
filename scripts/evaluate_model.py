try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
except ImportError as exc:
    raise ImportError(
        "The 'transformers' package is required. Install dependencies with \"pip install -r requirements.txt\"."
    ) from exc

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained('models/kjv_language_model')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def evaluate_model(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt')
    loss = model(**inputs, labels=inputs['input_ids']).loss
    return loss.item()

# Example evaluation
test_text = "In the beginning God created the heaven and the earth."
print(f"Model Loss: {evaluate_model(test_text, model, tokenizer)}")
