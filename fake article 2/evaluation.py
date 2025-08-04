import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import re

# 1. Load your saved model
model_path = "fake article 2/bert_fake_news_model/"  # Path containing your model files
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
print("Model loaded successfully")

# 2. Prepare your data (assuming df has 'cleaned_text' and 'label' columns)
# If you need to recreate your validation set:
# Load CSVs - make sure these files are in your working directory
real1 = pd.read_csv('fake article 2/newdatasets/PolitiFact_real_news_content.csv')
fake1 = pd.read_csv('fake article 2/newdatasets/PolitiFact_fake_news_content.csv')
real2 = pd.read_csv('fake article 2/newdatasets/True.csv')
fake2 = pd.read_csv('fake article 2/newdatasets/Fake.csv')

# Assign binary labels
real1['label'] = 0
real2['label'] = 0
fake1['label'] = 1
fake2['label'] = 1

# Select and rename text column
real1['cleaned_text'] = real1['text']
fake1['cleaned_text'] = fake1['text']
real2['cleaned_text'] = real2['text']
fake2['cleaned_text'] = fake2['text']

# Combine relevant columns
df = pd.concat([
    real1[['cleaned_text', 'label']],
    fake1[['cleaned_text', 'label']],
    real2[['cleaned_text', 'label']],
    fake2[['cleaned_text', 'label']]
], ignore_index=True)

# Clean the text
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)  # Remove non-alphanum chars
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace
    return text.strip().lower()

df['cleaned_text'] = df['cleaned_text'].astype(str).apply(clean_text)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['cleaned_text'],
    df['label'],
    test_size=0.2,  # Same as your original 80-20 split
    random_state=42,
    stratify=df['label']  # Maintains class balance
)
print("Data cleaned")

# 3. Tokenize validation data (no need to retokenize training data)
def tokenize_for_eval(texts, labels):
    encodings = tokenizer(texts.tolist(), 
                         truncation=True,
                         padding=True,
                         max_length=64,
                         return_tensors="pt")
    return TensorDataset(encodings['input_ids'], 
                       encodings['attention_mask'],
                       torch.tensor(labels.values))

val_dataset = tokenize_for_eval(val_texts, val_labels)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
print("Tokenization started")
# 4. Enhanced evaluation function
def evaluate_with_metrics(dataloader):
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    
    predictions = []
    true_labels = []
    
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1]
        }
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(preds)
        true_labels.extend(batch[2].cpu().numpy())
    
    # Generate metrics
    print("\nClass Distribution in Validation Set:")
    print(pd.Series(true_labels).value_counts().to_frame('Count'))
    
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, 
                              target_names=['Fake (0)', 'Real (1)'],
                              digits=4))
    
    # Confusion Matrix with percentages
    cm = confusion_matrix(true_labels, predictions)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(10,5))
    
    plt.subplot(1,2,1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Fake (0)', 'Real (1)'],
               yticklabels=['Fake (0)', 'Real (1)'])
    plt.title('Counts')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    plt.subplot(1,2,2)
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
               xticklabels=['Fake (0)', 'Real (1)'],
               yticklabels=['Fake (0)', 'Real (1)'])
    plt.title('Percentages')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    plt.tight_layout()
    plt.show()
print("Tokenization done")

# 5. Run evaluation
evaluate_with_metrics(val_dataloader)