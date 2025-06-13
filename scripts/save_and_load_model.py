try:
    from transformers import GPT2LMHeadModel
except ImportError as exc:
    raise ImportError(
        "The 'transformers' package is required. Install dependencies with \"pip install -r requirements.txt\"."
    ) from exc

# Save the model
def save_model(model, path='models/kjv_language_model'):
    model.save_pretrained(path)

# Load the model
def load_model(path='models/kjv_language_model'):
    return GPT2LMHeadModel.from_pretrained(path)
