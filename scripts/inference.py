from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the trained model
model = GPT2LMHeadModel.from_pretrained('models/kjv_language_model')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def generate_text(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(inputs['input_ids'], max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
prompt = "In the beginning"
print(generate_text(prompt))
