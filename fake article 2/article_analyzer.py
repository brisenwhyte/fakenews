# -*- coding: utf-8 -*-
"""
This script contains the core logic for analyzing a news article.
It has been refactored from detect_article2.py to be importable and used by a web server.
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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "bert_fake_news_model")
MAX_LEN = 128
class_names = ['Real', 'Fake']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

bert_model = None
tokenizer = None
bert_loaded = False
try:
    if not os.path.isdir(MODEL_DIR):
        raise OSError(f"Directory not found: '{MODEL_DIR}'")
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    bert_model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    bert_model.to(device)
    bert_model.eval()
    bert_loaded = True
    print("✅ BERT model and tokenizer loaded successfully.")
except Exception as e:
    print(f"❌ Error loading local BERT model: {e}")
    print("Please ensure your project structure is correct and all model files are in place.")

# --- Gemini API Configuration ---
gemini_model = None
fact_check_enabled = False
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    print("❌ GOOGLE_API_KEY environment variable not found. Fact-checking will be disabled.")
else:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        fact_check_enabled = True
        print("✅ Google Gemini API configured successfully.")
    except Exception as e:
        print(f"❌ Error configuring Gemini API: {e}")

# --- 2. Helper Functions ---

def fetch_article_text(url):
    """Fetches and parses the main text content from a given article URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        title = soup.find('h1')
        article_title = title.get_text(strip=True) if title else "Title Not Found"
        
        paragraphs = soup.find_all('p')
        if not paragraphs:
            return None, ""
        article_text = ' '.join([p.get_text(strip=True) for p in paragraphs])
        return article_text, article_title
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching URL: {e}")
        return None, ""

def clean_text_for_bert(text):
    if not isinstance(text, str): return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def predict_with_bert(text):
    if not bert_loaded:
        return "BERT Model Not Loaded", {'Real': 0.0, 'Fake': 0.0}
    if not text:
        return "No text provided", {'Real': 0.5, 'Fake': 0.5}

    encoded_text = tokenizer.encode_plus(
        text, max_length=MAX_LEN, add_special_tokens=True, return_token_type_ids=False,
        padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt',
    )
    
    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_mask)
    
    probabilities = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    predicted_index = np.argmax(probabilities)
    predicted_class = class_names[predicted_index]
    
    return predicted_class, {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}

def analyze_with_gemini(article_text, article_title):
    if not fact_check_enabled or not article_text:
        return "Analysis Skipped", "Fact-check analysis skipped (API not configured or no text)."

    truncated_text = article_text[:8000]
    prompt = f"""
    You are an expert fact-checker. Your task is to analyze the following news article for factual accuracy.
    Structure your analysis clearly as follows. Do not include any other preamble before the "OVERALL SUMMARY".

    OVERALL SUMMARY: [Provide a one-sentence conclusion about the article's general reliability.]
    ---
    CLAIM 1: [State the first major claim from the article.]
    VERDICT: [Choose one: True, False, Misleading, Unverifiable]
    EVIDENCE: [Explain your reasoning and cite the URL of the primary source you used for verification.]
    ---
    (Continue for all major claims found, up to a maximum of 3)

    ARTICLE TITLE: "{article_title}"
    ARTICLE CONTENT: "{truncated_text}"
    """
    
    try:
        safety_settings = [{"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        response = gemini_model.generate_content(prompt, safety_settings=safety_settings)
        
        if not response.parts:
            return "Analysis Blocked", "The content may have violated safety policies or was unanswerable."

        full_analysis = response.text
        summary_match = re.search(r"OVERALL SUMMARY:\s*(.*)", full_analysis)
        summary = summary_match.group(1).strip() if summary_match else "No overall summary provided by AI."
        details_match = re.search(r"---(.*)", full_analysis, re.DOTALL)
        details = details_match.group(1).strip() if details_match else "No detailed claim analysis provided by AI."
        
        return summary, details
    except Exception as e:
        return "API Error", f"An error occurred while calling the Gemini API: {e}"

def get_overall_assessment(bert_pred, fact_check_details):
    bert_is_fake = bert_pred.lower() == 'fake'
    analysis_lower = fact_check_details.lower()
    has_false = "verdict: false" in analysis_lower or "verdict: misleading" in analysis_lower
    has_true = "verdict: true" in analysis_lower
    
    fc_verdict = "mixed"
    if has_false and not has_true: fc_verdict = "false"
    elif has_true and not has_false: fc_verdict = "true"
    elif not has_true and not has_false: fc_verdict = "unverified"
    
    if bert_is_fake and fc_verdict == "false":
        return "High likelihood of being unreliable. Both text style and factual claims raise significant concerns. Extreme caution is advised."
    elif not bert_is_fake and fc_verdict == "true":
        return "High likelihood of being credible. The article's style appears standard and its key claims are verified as true."
    elif bert_is_fake and (fc_verdict == "true" or fc_verdict == "mixed"):
        return "Mixed signals. The article's writing style shows patterns associated with disinformation, but some or all of its core claims appear factually correct. This could be 'truthful' content framed in a misleading or propagandistic way."
    elif not bert_is_fake and (fc_verdict == "false" or fc_verdict == "mixed"):
        return "Mixed signals. The article is written in a conventional style, but contains claims that have been identified as false or misleading. This may be a professionally written piece containing disinformation."
    elif fc_verdict == "unverified":
        return f"Assessment inconclusive. The AI could not verify the article's claims. Based on text style alone, the BERT model predicts it is likely '{bert_pred}'."
    else:
        return "Assessment inconclusive. Analyses produced conflicting or mixed results. Review both reports carefully."

def run_analysis(input_data):
    """
    Main analysis function called by the API.
    `input_data` is a dict with either a 'url' or 'text' key.
    """
    article_content = None
    article_title = ""
    source_identifier = ""

    if 'url' in input_data and input_data['url']:
        source_identifier = input_data['url']
        article_content, article_title = fetch_article_text(source_identifier)
    elif 'text' in input_data and input_data['text']:
        source_identifier = "Pasted Text"
        article_content = input_data['text']
        article_title = "Pasted Article"
    
    if not article_content or len(article_content) < 50:
        return {"error": "Could not retrieve enough text to analyze."}

    # BERT Analysis
    cleaned_content = clean_text_for_bert(article_content)
    bert_prediction, bert_probs = predict_with_bert(cleaned_content)
    
    # Gemini Fact Check Analysis
    fact_check_summary, fact_check_details = analyze_with_gemini(article_content, article_title)

    # Overall Assessment
    overall_assessment = "Could not generate overall assessment."
    if "error" not in fact_check_details.lower() and "blocked" not in fact_check_details.lower():
        overall_assessment = get_overall_assessment(bert_prediction, fact_check_details)

    return {
        "source": source_identifier,
        "title": article_title,
        "bert_prediction": bert_prediction,
        "bert_probabilities": bert_probs,
        "gemini_summary": fact_check_summary,
        "gemini_details": fact_check_details,
        "overall_assessment": overall_assessment
    }