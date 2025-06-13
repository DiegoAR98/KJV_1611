try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from transformers.optimization import AdamW
except ImportError as exc:
    raise ImportError(
        "The 'transformers' package is required. Install dependencies with \"pip install -r requirements.txt\"."
    ) from exc

import torch

# Load pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Load preprocessed text
with open('data/kjv_1611.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Tokenize the Bible text
inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)

# Simple fine-tuning loop to avoid distributed training issues
optimizer = AdamW(model.parameters(), lr=5e-5)
model.train()
for _ in range(1):  # single epoch for demonstration
    outputs = model(inputs["input_ids"], labels=inputs["input_ids"])
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Save the model
save_path = "models/kjv_language_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
