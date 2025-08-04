# -*- coding: utf-8 -*-
"""
This script analyzes a news article from a URL using a two-pronged approach:
1. A locally-run, fine-tuned BERT model to classify the article's text style as 'Real' or 'Fake'.
2. DeepSeek API to analyze the article's claims for fact-checking.

Setup:
1. Ensure the 'bert_fake_news_model' directory is in the same folder as this script.
2. Install required packages:
   pip install torch transformers requests beautifulsoup4 numpy
3. Get a DeepSeek API Key from their platform.
4. Set the API key as an environment variable before running:
   - Linux/macOS: export DEEPSEEK_API_KEY="your_api_key"
   - Windows:     set DEEPSEEK_API_KEY="your_api_key"
"""

import torch
import re
import requests
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import os
import textwrap
from urllib.parse import urlparse
from openai import OpenAI

# --- 1. Configuration and Model Loading ---

# --- BERT Model Configuration ---
# Get the directory where the current script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Define the path to the directory containing the fine-tuned model
MODEL_DIR = os.path.join(SCRIPT_DIR, "bert_fake_news_model")
MAX_LEN = 128
class_names = ['Real', 'Fake'] # 0: Real, 1: Fake
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    if not os.path.isdir(MODEL_DIR):
        raise OSError(f"Directory not found: '{MODEL_DIR}'")
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    bert_model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    bert_model.to(device)
    bert_model.eval()
    print("✅ BERT model and tokenizer loaded successfully.")
except Exception as e:
    print(f"❌ Error loading local BERT model: {e}")
    print("Please ensure your project structure is correct and all model files are in place.")
    exit()

# --- DeepSeek API Configuration ---
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
if not DEEPSEEK_API_KEY:
    print("❌ DEEPSEEK_API_KEY environment variable not found.")
    print("Please make sure you have set the DEEPSEEK_API_KEY environment variable.")
    fact_check_enabled = False
else:
    fact_check_enabled = True
    print("✅ DeepSeek API configured successfully.")

# --- 2. Helper Functions ---

def fetch_article_text(url):
    """Fetches and parses the main text content from a given article URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try to get the article title for fact-checking
        title = soup.find('h1')
        article_title = title.get_text(strip=True) if title else ""
        
        paragraphs = soup.find_all('p')
        if not paragraphs:
            return None, ""
        article_text = ' '.join([p.get_text(strip=True) for p in paragraphs])
        return article_text, article_title
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching URL: {e}")
        return None, ""

def clean_text_for_bert(text):
    """Cleans text specifically for the BERT model's input."""
    if not isinstance(text, str): return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def predict_with_bert(text):
    """Takes cleaned text and returns the BERT model's prediction and probabilities."""
    if not text:
        return "No text provided", {class_names[0]: 0.5, class_names[1]: 0.5}

    encoded_text = tokenizer.encode_plus(
        text,
        max_length=MAX_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_mask)
    
    probabilities = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    predicted_index = np.argmax(probabilities)
    predicted_class = class_names[predicted_index]
    
    probs_dict = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    
    return predicted_class, probs_dict

def extract_key_claims(text, title):
    """Extracts potential claims from the article text for fact-checking."""
    # Simple approach - use the title and first few sentences as claims
    claims = []
    if title:
        claims.append(title)
    
    # Take the first 3 sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    claims.extend(sentences[:3])
    
    return claims[:5]  # Return up to 5 claims

def deepseek_fact_check(query):
    """Analyzes claims using DeepSeek's API via OpenAI client."""
    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "You are a fact-checking assistant. Analyze the following claim and provide a rating (True/False/Misleading/Unverified), confidence percentage, and brief explanation."
                },
                {
                    "role": "user",
                    "content": f"Fact-check this claim: {query}"
                }
            ],
            temperature=0.3,
            max_tokens=300,
            stream=False
        )
        
        # Extract the response content
        if response.choices and response.choices[0].message:
            return {
                'text': query,
                'rating': '',  # These would need to be parsed from the response
                'explanation': response.choices[0].message.content,
                'sources': [],
                'confidence': 0
            }
        return None
        
    except Exception as e:
        print(f"❌ Error calling DeepSeek API: {e}")
        return None

# Update the analyze_with_fact_check function to handle the new response format:
def analyze_with_fact_check(article_text, article_title):
    """Analyzes the article using DeepSeek's API."""
    if not fact_check_enabled or not article_text:
        return None, "Fact-check analysis skipped (API not configured or no text)."

    claims = extract_key_claims(article_text, article_title)
    if not claims:
        return None, "No significant claims found to fact-check."

    results = []
    for claim in claims:
        fact_check = deepseek_fact_check(claim)
        if fact_check:
            result = {
                'claim': fact_check.get('text', claim),
                'rating': fact_check.get('rating', 'Unknown'),
                'explanation': fact_check.get('explanation', 'No explanation provided'),
                'sources': fact_check.get('sources', []),
                'confidence': fact_check.get('confidence', 0)
            }
            results.append(result)
    
    if not results:
        return None, "No existing fact-checks found for the article's claims."
    
    # Format the results
    summary = f"Article makes {len(claims)} key claims. Found {len(results)} relevant fact-checks."
    analysis = "\n".join([
        f"Claim: {r['claim']}\n"
        f"Rating: {r['rating']} (confidence: {r['confidence']:.0%})\n"
        f"Explanation: {r['explanation']}\n"
        f"Sources: {', '.join(r['sources']) if r['sources'] else 'None'}\n"
        for r in results
    ])
    
    return summary, analysis

def get_overall_assessment(bert_pred, fact_check_results):
    """Provides a synthesized conclusion based on both models' outputs."""
    bert_is_fake = bert_pred.lower() == 'fake'
    
    # Analyze fact check results
    if "No existing fact-checks" in fact_check_results:
        fc_verdict = "unverified"
    elif any(rating.lower() in fact_check_results.lower() for rating in ['false', 'misleading', 'incorrect', 'debunked']):
        fc_verdict = "false"
    elif any(rating.lower() in fact_check_results.lower() for rating in ['true', 'correct', 'accurate', 'verified']):
        fc_verdict = "true"
    else:
        fc_verdict = "mixed"

    if bert_is_fake and fc_verdict == "false":
        return "High likelihood of being unreliable. Both the text style and existing fact-checks raise concerns."
    elif not bert_is_fake and fc_verdict == "true":
        return "High likelihood of being credible. The article's style appears standard and its claims are supported by fact-checkers."
    elif bert_is_fake and fc_verdict == "true":
        return "Mixed signals. The article's writing style resembles patterns of fake news, but its core claims appear to be factually correct. Caution is advised."
    elif not bert_is_fake and fc_verdict == "false":
        return "Mixed signals. The article is written in a conventional style, but contains claims that have been debunked by fact-checkers."
    elif fc_verdict == "unverified":
        return "Assessment inconclusive. The article's claims haven't been fact-checked yet. Please verify with additional sources."
    else:
        return "Assessment inconclusive. Please review both analyses carefully to form your own judgment."

# --- 3. Main Execution Block ---

if __name__ == "__main__":
    while True:
        article_url = input("Paste the URL of the article to check (or type 'exit' to quit): ")
        if article_url.lower() == 'exit':
            break
        
        print("\nFetching article content...")
        article_content, article_title = fetch_article_text(article_url)
        
        if not article_content:
            print("Could not retrieve article content. Please check the URL and try again.\n")
            continue
        
        # --- BERT Analysis ---
        print("Analyzing with BERT model (style & patterns)...")
        cleaned_content = clean_text_for_bert(article_content)
        
        if len(cleaned_content) < 50:
            print("Could not find enough text on the page to analyze.\n")
            continue

        bert_prediction, bert_probs = predict_with_bert(cleaned_content)
        
        # --- Fact Check Analysis ---
        print("Searching for existing fact-checks...")
        fact_check_summary, fact_check_analysis = analyze_with_fact_check(article_content, article_title)

        # --- Display Combined Report ---
        print("\n" + "="*40)
        print("      DUAL-LAYER ANALYSIS REPORT")
        print("="*40)
        print(f"URL Analyzed: {article_url}\n")
        
        print("--- BERT Model Analysis (Text-Based) ---")
        print(f"Prediction: This article's writing style is likely {bert_prediction.upper()}")
        print("Confidence:")
        print(f"  - Real Style: {bert_probs['Real']:.2%}")
        print(f"  - Fake Style: {bert_probs['Fake']:.2%}")
        print("-" * 40)
        
        print("\n--- Fact Check Analysis ---")
        if fact_check_summary:
            print("Summary:")
            print(textwrap.fill(fact_check_summary, width=80))
            print("\nDetailed Results:")
            print(textwrap.fill(fact_check_analysis, width=80))
        else:
            print(fact_check_analysis)
        print("-" * 40)

        print("\n--- Overall Assessment ---")
        if fact_check_analysis and "error" not in fact_check_analysis.lower():
            overall_conclusion = get_overall_assessment(bert_prediction, fact_check_analysis)
            print(textwrap.fill(overall_conclusion, width=80))
        else:
            print("Could not generate an overall assessment as the fact-check failed.")

        print("="*40 + "\n")