from transformers import GPT2LMHeadModel

# Save the model
def save_model(model, path='../models/kjv_language_model'):
    model.save_pretrained(path)

# Load the model
def load_model(path='../models/kjv_language_model'):
    return GPT2LMHeadModel.from_pretrained(path)
