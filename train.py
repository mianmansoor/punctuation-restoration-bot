import re
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import AdamW
from tqdm import tqdm
import os

print("Loading AG News dataset...")
dataset = load_dataset("ag_news")
train_data = dataset['train'].select(range(5000))

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

input_texts = [ "restore punctuation: " + remove_punctuation(item['text'].lower()) for item in train_data]
target_texts = [item['text'] for item in train_data]

df = pd.DataFrame({"input_text": input_texts, "target_text": target_texts})
df.to_csv("punctuation_dataset_small.csv", index=False)
print(f"Dataset size: {len(df)}")

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")

class PunctuationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=64):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text = self.data.iloc[idx]['input_text']
        target_text = self.data.iloc[idx]['target_text']

        input_enc = tokenizer(
            input_text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        target_enc = tokenizer(
            target_text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
        )

        return {
            "input_ids": input_enc.input_ids.flatten(),
            "attention_mask": input_enc.attention_mask.flatten(),
            "labels": target_enc.input_ids.flatten()
        }

train_dataset = PunctuationDataset(df, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small").to(device)

optimizer = AdamW(model.parameters(), lr=3e-5)
epochs = 3

model.train()
for epoch in range(epochs):
    loop = tqdm(train_loader, leave=True)
    total_loss = 0
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader)}")

os.makedirs("model_save", exist_ok=True)
model.save_pretrained("model_save")
tokenizer.save_pretrained("model_save")
print("Model saved in 'model_save' folder.")
