"""
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained("model_save")
model = T5ForConditionalGeneration.from_pretrained("model_save").to(device)
model.eval()

def restore_punctuation(text):
    text = "restore punctuation: " + text
    enc = tokenizer(text, return_tensors="pt").to(device)
    output = model.generate(**enc, max_length=64)
    return tokenizer.decode(output[0], skip_special_tokens=True)

print(restore_punctuation("how are you doing today"))
print(restore_punctuation("lets go to the park tomorrow"))
print(restore_punctuation("we will visit the museum on sunday"))
"""

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained("model_save")
model = T5ForConditionalGeneration.from_pretrained("model_save").to(device)
model.eval()

def restore_punctuation(text):
    """Restore punctuation in a given sentence or paragraph."""
    text = "restore punctuation: " + text.lower()
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128).to(device)
    output = model.generate(**enc, max_length=128)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Take user input (paragraph or sentence)
user_input = input("Enter a sentence or paragraph without punctuation:\n")
result = restore_punctuation(user_input)
print("\nPunctuated Output:\n", result)
