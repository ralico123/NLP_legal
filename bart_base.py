# ✅ INSTALL DEPENDENCIES
import time
start = time.time()
!rm -rf ~/.cache/huggingface/transformers
!rm -rf ~/.cache/torch
!pip uninstall -y transformers torch accelerate
!pip install torch==2.2.1 transformers==4.39.3 accelerate==0.28.0 -q
!pip install roman datasets textstat evaluate rouge-score bert_score --quiet
!pip install git+https://github.com/PrimerAI/blanc.git --quiet
!pip install symspellpy nltk wordfreq
end = time.time()
time = end-start
print(f"\n All dependencies installed in {time:.2f} seconds!")

import torch
import numpy as np
import textstat 
import evaluate
import rouge_score
from blanc import BlancHelp
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
import pandas as pd
from tqdm import tqdm
import os
import random
from torch.optim import AdamW

def set_reproducibility(seed: int = 42):
    # Python RNG
    random.seed(seed)
    
    # NumPy RNG
    np.random.seed(seed)
    
    # PyTorch RNGs
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # CuDNN settings for full determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set PYTHONHASHSEED env var
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"✅ Reproducibility seed set to {seed}")

# Call it at the very start of your script or notebook
set_reproducibility(42)

# ✅ Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("CUDA cache cleared!")

# ✅ Load dataset from parquet files (Hugging Face Hub)
splits = {
    'train': 'data/train-00000-of-00001.parquet',
    'test': 'data/test-00000-of-00001.parquet',
}
train_df = pd.read_parquet("hf://datasets/FiscalNote/billsum/" + splits["train"])[['text', 'summary']]
val_df = pd.read_parquet("hf://datasets/FiscalNote/billsum/" + splits["test"])[['text', 'summary']]
# ✅ Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# ✅ Load tokenizer and model
model_name = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
model.gradient_checkpointing_enable()
model.config.use_cache = False


# ✅ Preprocessing with padding/truncation
max_input_length = 512
max_target_length = 128

def preprocess_function(examples):
    inputs = tokenizer(
        examples["text"],
        max_length=max_input_length,
        padding="max_length",
        truncation=True,
    )
    targets = tokenizer(
        examples["summary"],
        max_length=max_target_length,
        padding="max_length",
        truncation=True,
    )
    inputs["labels"] = targets["input_ids"]
    return inputs

# ✅ Map preprocessing
train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=["text", "summary"])
val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=["text", "summary"])

# ✅ Dataloaders
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=data_collator) # change batch size according to available compute power
val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=data_collator)

# ✅ Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# ✅ Training loop
epochs = 3
model.train()

for epoch in range(epochs):
    loop = tqdm(train_loader, leave=True)
    total_loss = 0

    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        loop.set_description(f"Epoch {epoch + 1}")
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1} finished. Avg loss: {avg_loss:.4f}")



# ===== GENERATION FUNCTION =====
model.eval()

from tqdm import tqdm

def generate_in_batches(texts, batch_size=1, limit=300):
    preds = []
    texts = texts[:limit]
    for i in tqdm(range(0, len(texts), batch_size), desc="🔄 Generating summaries"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_target_length,
                num_beams=4,
                early_stopping=True
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        preds.extend(decoded)
    return preds

# Limit for faster runs
GEN_LIMIT = len(val_df)
raw_texts = val_df["text"].tolist()[:GEN_LIMIT]
refs = val_df["summary"].tolist()[:GEN_LIMIT]
preds = generate_in_batches(raw_texts, limit=GEN_LIMIT)

# ===== EVALUATION =====
import re
import numpy as np
from rouge_score import rouge_scorer
import bert_score
from blanc import BlancHelp
import textstat
import roman
import string
from symspellpy.symspellpy import SymSpell, Verbosity
import pkg_resources
import nltk
nltk.download("punkt_tab")


# Initialize SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# Function to clean/standardize text
def clean_text_for_readability(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove digits for readability check
    allowed_chars = string.ascii_lowercase + " "
    cleaned = "".join(ch if ch in allowed_chars else " " for ch in text)
    return " ".join(cleaned.split())

# Convert Roman numerals (e.g., II -> 2)
def convert_roman_numerals(text):
    words = text.split()
    converted = []
    for word in words:
        clean_word = word.strip(string.punctuation)
        try:
            num = roman.fromRoman(clean_word.upper())
            converted_word = word.replace(clean_word, str(num))
            converted.append(converted_word)
        except roman.InvalidRomanNumeralError:
            converted.append(word)
    return " ".join(converted)

# Use SymSpell to correct spelling/grammar
def correct_with_symspell(text):
    corrected_words = []
    for word in text.split():
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        if suggestions:
            corrected_words.append(suggestions[0].term)
        else:
            corrected_words.append(word)
    return " ".join(corrected_words)

# Optional acronym expansion
def replace_acronyms(text):
    known_acronyms = {
        "fda": "food and drug administration",
        "epa": "environmental protection agency",
        "pdmp": "prescription drug monitoring program",
        "medpac": "medicare payment advisory commission"
        # add more if needed
    }
    for acronym, full in known_acronyms.items():
        text = re.sub(rf"\b{acronym}\b", full, text, flags=re.IGNORECASE)
    return text

def is_valid_text(text):
    return text and len(text.split()) > 1

# 🚀 Evaluation
fk_scores = []
dc_scores = []
readability_failures = 0  
cleaned_preds = []  

for i, pred in enumerate(preds):
    try:
        text = convert_roman_numerals(pred)
        text = correct_with_symspell(text)
        text = replace_acronyms(text)
        cleaned = clean_text_for_readability(text)

        if is_valid_text(cleaned):
            fk = textstat.flesch_kincaid_grade(cleaned)
            dc = textstat.dale_chall_readability_score(cleaned)
            fk_scores.append(fk)
            dc_scores.append(dc)
            cleaned_preds.append(text)
        else:
            readability_failures += 1
            cleaned_preds.append("")
    except Exception:
        readability_failures += 1
        cleaned_preds.append("")


# ================== BLANC Help ==================
blanc = BlancHelp(device='cuda', inference_batch_size=1)
blanc_scores = blanc.eval_pairs(refs, cleaned_preds)
blanc_mean = np.mean(blanc_scores)


# ================== ROUGE ==================
rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
r1, r2, rl = [], [], []

for ref, pred in zip(refs, cleaned_preds):
    if pred.strip():
        scores = rouge.score(ref, pred)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rl.append(scores["rougeL"].fmeasure)

# ================== BERTScore ==================

P, R, F1 = bert_score.score(cleaned_preds, refs, model_type="distilbert-base-uncased", batch_size=1, device="cuda", lang="en", verbose=False)

# ================== REPORT ==================
print(f"❌ Total samples that failed readability: {readability_failures}")
print("\n📘 Readability (Mean):")
print(f"Flesch-Kincaid Grade: {np.nanmean(fk_scores):.2f}")
print(f"Dale-Chall Score: {np.nanmean(dc_scores):.2f}")

print("\n📝 ROUGE Scores (Mean):")
print(f"ROUGE-1: {np.mean(r1):.4f}")
print(f"ROUGE-2: {np.mean(r2):.4f}")
print(f"ROUGE-L: {np.mean(rl):.4f}")

print("\n🧠 BERTScore F1 (Mean):")
print(f"F1: {F1.mean().item():.4f}")

print("\n💡 BLANC Help (Mean):")
print(f"BLANC Score: {blanc_mean:.4f}")

# ===== SAVE MODEL + TOKENIZER =====
from pathlib import Path

save_dir = Path("saved_models") / f"{model_name}_in{max_input_length}_out{max_target_length}"
save_dir.mkdir(parents=True, exist_ok=True)

model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"\n💾 Model + tokenizer saved to: {save_dir}")
