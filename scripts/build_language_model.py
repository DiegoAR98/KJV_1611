from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Load preprocessed text
with open('../data/kjv_1611.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Tokenize the Bible text
inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)

# Fine-tune the model
training_args = TrainingArguments(
    output_dir='../models/',
    logging_dir='../logs/',
    logging_steps=100,
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=inputs['input_ids'],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)



trainer.train()
