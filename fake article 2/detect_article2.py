# -*- coding: utf-8 -*-
"""
This script analyzes a news article from a URL or pasted text using a two-pronged approach:
1. A locally-run, fine-tuned BERT model to classify the article's text style as 'Real' or 'Fake'.
2. Google's Gemini API to perform an AI-driven fact-check on the article's claims.

Setup:
1. Ensure the 'bert_fake_news_model' directory is in the same folder as this script.
2. Install required packages:
   pip install torch transformers requests beautifulsoup4 numpy google-generativeai
3. Get a Google API Key from the Google AI Studio.
4. Set the API key as an environment variable before running:
   - Linux/macOS: export GOOGLE_API_KEY="your_api_key"
   - Windows:     set GOOGLE_API_KEY="your_api_key"
"""

import torch
import re
import requests
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import os
import textwrap
import google.generativeai as genai

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

# --- Gemini API Configuration ---
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    print("❌ GOOGLE_API_KEY environment variable not found.")
    print("Please make sure you have set the GOOGLE_API_KEY environment variable.")
    fact_check_enabled = False
else:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        fact_check_enabled = True
        print("✅ Google Gemini API configured successfully.")
    except Exception as e:
        print(f"❌ Error configuring Gemini API: {e}")
        fact_check_enabled = False


# --- 2. Helper Functions ---

def fetch_article_text(url):
    """Fetches and parses the main text content from a given article URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
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

def analyze_with_gemini(article_text, article_title):
    """Analyzes the article using the Gemini API for fact-checking."""
    if not fact_check_enabled or not article_text:
        return None, "Fact-check analysis skipped (API not configured or no text)."

    # Truncate content to avoid exceeding token limits
    truncated_text = article_text[:8000]

    prompt = f"""
    You are an expert fact-checker. Your task is to analyze the following news article for factual accuracy.

    Please perform the following steps:
    1.  Read the provided article title and content carefully.
    2.  Identify the main factual claims made in the article.
    3.  Independently verify each claim using reliable, neutral sources (e.g., established news agencies, scientific bodies, official reports). Do not use the source of the article itself for verification.
    4.  Structure your analysis clearly as follows. Do not include any other preamble before the "OVERALL SUMMARY".

    OVERALL SUMMARY: [Provide a one-sentence conclusion about the article's general reliability.]
    ---
    CLAIM 1: [State the first major claim from the article.]
    VERDICT: [Choose one: True, False, Misleading, Unverifiable]
    EVIDENCE: [Explain your reasoning and cite the URL of the primary source you used for verification.]
    ---
    CLAIM 2: [State the second major claim from the article.]
    VERDICT: [Choose one: True, False, Misleading, Unverifiable]
    EVIDENCE: [Explain your reasoning and cite the URL of the primary source you used for verification.]
    ---
    (Continue for all major claims found)

    Here is the article to analyze:
    ARTICLE TITLE: "{article_title}"
    ARTICLE CONTENT: "{truncated_text}"
    """
    
    try:
        # Lower the safety settings to reduce blocking for controversial news topics
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        response = gemini_model.generate_content(prompt, safety_settings=safety_settings)
        
        # Check if the response was blocked or empty
        if not response.parts:
            return None, "Analysis blocked. The content may have violated safety policies or was unanswerable."

        full_analysis = response.text
        
        # Split summary from detailed analysis
        summary_match = re.search(r"OVERALL SUMMARY:\s*(.*)", full_analysis)
        summary = summary_match.group(1).strip() if summary_match else "No overall summary provided by AI."
        
        details_match = re.search(r"---(.*)", full_analysis, re.DOTALL)
        details = details_match.group(1).strip() if details_match else "No detailed claim analysis provided by AI."
        
        return summary, details

    except Exception as e:
        return None, f"❌ An error occurred while calling the Gemini API: {e}"

def get_overall_assessment(bert_pred, fact_check_analysis):
    """Provides a synthesized conclusion based on both models' outputs."""
    bert_is_fake = bert_pred.lower() == 'fake'
    
    # Analyze Gemini's verdict
    analysis_lower = fact_check_analysis.lower()
    has_false = "verdict: false" in analysis_lower or "verdict: misleading" in analysis_lower
    has_true = "verdict: true" in analysis_lower
    
    fc_verdict = "mixed" # Default
    if has_false and not has_true:
        fc_verdict = "false"
    elif has_true and not has_false:
        fc_verdict = "true"
    elif not has_true and not has_false:
        fc_verdict = "unverified"
    
    if bert_is_fake and fc_verdict == "false":
        return "High likelihood of being unreliable. Both the text style and the factual claims raise significant concerns. Extreme caution is advised."
    elif not bert_is_fake and fc_verdict == "true":
        return "High likelihood of being credible. The article's style appears standard and its key claims are verified as true."
    elif bert_is_fake and (fc_verdict == "true" or fc_verdict == "mixed"):
        return "Mixed signals. The article's writing style shows patterns associated with disinformation, but some or all of its core claims appear factually correct. This could be 'truthful' content framed in a misleading or propagandistic way."
    elif not bert_is_fake and (fc_verdict == "false" or fc_verdict == "mixed"):
        return "Mixed signals. The article is written in a conventional style, but contains claims that have been identified as false or misleading. This may be a professionally written piece containing disinformation."
    elif fc_verdict == "unverified":
        return "Assessment inconclusive. The AI could not verify the article's claims. Based on text style alone, the BERT model predicts it is likely '{}'. Please verify with additional trusted sources.".format(bert_pred)
    else: # Fallback for other mixed cases
        return "Assessment inconclusive. The analyses produced conflicting or mixed results. Please review both reports carefully to form your own judgment."

# --- 3. Main Execution Block ---

if __name__ == "__main__":
    while True:
        # --- MODIFIED: Updated input prompt for URL or text ---
        print("Paste the article URL or the full article text to check.")
        user_input = input("Enter input (or type 'exit' to quit): ")
        
        if user_input.lower() == 'exit':
            break

        article_content = None
        article_title = ""
        source_identifier = ""

        # --- MODIFIED: Logic to differentiate between URL and pasted text ---
        # Heuristic: if it starts like a web address, treat it as a URL.
        cleaned_input = user_input.strip()
        if cleaned_input.startswith(('http://', 'https://', 'www.')):
            print("\nInput detected as a URL. Fetching article content...")
            source_identifier = cleaned_input
            article_content, article_title = fetch_article_text(source_identifier)
        else:
            # Assume it's pasted text
            print("\nInput detected as pasted text. Proceeding with analysis...")
            source_identifier = "Pasted Text"
            article_content = cleaned_input
            # Use a placeholder title as it's not available from pasted text
            article_title = "Pasted Article"

        if not article_content:
            print("Could not retrieve or identify article content. Please check the input and try again.\n")
            continue
        
        # --- BERT Analysis ---
        print("Analyzing with BERT model (style & patterns)...")
        cleaned_content = clean_text_for_bert(article_content)
        
        if len(cleaned_content) < 50:
            print("Could not find enough text in the input to analyze meaningfully.\n")
            continue

        bert_prediction, bert_probs = predict_with_bert(cleaned_content)
        
        # --- Gemini Fact Check Analysis ---
        print("Analyzing claims with Gemini API...")
        fact_check_summary, fact_check_analysis = analyze_with_gemini(article_content, article_title)

        # --- Display Combined Report ---
        print("\n" + "="*40)
        print("      DUAL-LAYER ANALYSIS REPORT")
        print("="*40)
        # --- MODIFIED: Use the generic source identifier ---
        print(f"Source Analyzed: {source_identifier}\n")
        
        print("--- BERT Model Analysis (Text Style) ---")
        print(f"Prediction: This article's writing style is likely {bert_prediction.upper()}")
        print("Confidence:")
        print(f"  - Real Style: {bert_probs['Real']:.2%}")
        print(f"  - Fake Style: {bert_probs['Fake']:.2%}")
        print("-" * 40)
        
        print("\n--- Gemini AI Fact-Check Analysis ---")
        if fact_check_summary:
            print("AI Summary:")
            print(textwrap.fill(fact_check_summary, width=80))
            print("\nDetailed Claim Analysis:")
            print(textwrap.fill(fact_check_analysis, width=80, subsequent_indent='  '))
        else:
            # This branch handles errors or API disabled cases
            print(fact_check_analysis)
        print("-" * 40)

        print("\n--- Overall Assessment ---")
        if fact_check_analysis and "error" not in fact_check_analysis.lower() and "blocked" not in fact_check_analysis.lower():
            overall_conclusion = get_overall_assessment(bert_prediction, fact_check_analysis)
            print(textwrap.fill(overall_conclusion, width=80))
        else:
            print("Could not generate an overall assessment as the AI fact-check failed or was disabled.")

        print("="*40 + "\n")