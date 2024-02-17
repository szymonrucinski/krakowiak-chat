from googlesearch import search
from transformers import AutoTokenizer
import requests
from bs4 import BeautifulSoup
import justext
import re
import os
TOKENIZER_NAME = os.getenv('TOKENIZER_NAME', 'mistralai/Mistral-7B-Instruct-v0.1')


def fetch_html(url: str) -> str:
    """Fetches and returns the HTML content of a given URL."""
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching HTML: {e}")
        return ""


def clean_html(html_content: str) -> str:
    """Removes unwanted HTML tags and boilerplate content, returning clean text."""
    try:
        paragraphs = justext.justext(html_content, justext.get_stoplist("English"))
        text = "\n".join(p.text for p in paragraphs if not p.is_boilerplate)
        return condense_whitespace(text)
    except Exception as e:
        print(f"Error cleaning HTML: {e}")
        return ""


def condense_whitespace(text: str) -> str:
    """Reduces multiple whitespaces and newlines to a single space or newline."""
    return re.sub(r'\s+', ' ', text).strip()


def filter_paragraphs(text: str, blacklist: list) -> str:
    """Removes paragraphs containing any blacklisted substring."""
    return "\n".join(p for p in text.split("\n") if not any(sub in p for sub in blacklist))


def fetch_and_clean_data(query: str) -> str:
    """Fetches HTML content for a query, cleans it, and returns a concatenated string of clean data."""
    results_urls = [url for url in search(query, num=10, stop=10, pause=2)
                    if "example.com" not in url]

    html_contents = [fetch_html(url) for url in results_urls]
    cleaned_texts = [clean_html(html) for html in html_contents if html]

    # Optionally, sort by length or apply further filtering
    cleaned_texts = sorted(cleaned_texts, key=len)

    final_text = " ".join(cleaned_texts)[:4000]  # Limit to first 4000 characters
    final_text = final_text.lower()
    unwanted_phrases = ["comments", "login", "register", "advertisement"]
    final_text = filter_paragraphs(final_text, unwanted_phrases)

    return final_text


def create_prompt_with_source(user_query: str) -> str:
    """Constructs a prompt with cleaned source data for a given query."""
    source_data = fetch_and_clean_data(user_query)
    messages = [
    {"role": "user", "content": f"Please provide a concise answer to the following question: {user_query}, using only this source material: {source_data}"}
    ]
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer.apply_chat_template(messages, tokenize=False)
