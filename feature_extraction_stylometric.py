import numpy as np
import re
import nltk
import textstat
from nltk.corpus import stopwords
from nltk import pos_tag

nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag

FUNCTION_WORDS = [
    "the", "and", "to", "of", "in", "that", "it", "is",
    "was", "he", "for", "on", "are", "as", "with",
    "his", "they", "i", "at", "be", "this", "have"
]
STOPWORDS = set(stopwords.words('english'))

def function_word_features(words):
    """
    Proportion of function words in text.
   """
    total = len(words)
    return [words.count(w)/total for w in FUNCTION_WORDS] if total > 0 else [0]*len(FUNCTION_WORDS)

def pos_features(words):
    """
    Major POS tag frequencies.
    """
    tags = pos_tag(words)
    tag_counts = {}
    for _, tag in tags:
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
    total = len(words)
    selected_tags = ['NN', 'VB', 'JJ', 'RB', 'PRP']
    return [tag_counts.get(tag, 0)/total if total>0 else 0 for tag in selected_tags]

def readability_features(text):
    """
    Readability metrics: Flesch Reading Ease + Flesch-Kincaid Grade.
    """
    return [textstat.flesch_reading_ease(text), textstat.flesch_kincaid_grade(text)]

def extract_features(text):
    """
    Features include:
    - Sentence structure: mean/variance of sentence length, total words
    - Lexical style: average word length, type-token ratio, uppercase ratio
    - Rhetoric: punctuation counts (!, ?)
    - Function words: proportion of 24 function words
    - POS tag distribution: NN, VB, JJ, RB, PRP
    - Readability: Flesch Reading Ease, Flesch-Kincaid Grade
    """
    # Sentence & word tokenization
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    words_raw = re.findall(r'\b\w+\b', text)
    words = [w.lower() for w in words_raw]
    
    if len(words) == 0:
        return [0]* (8 + len(FUNCTION_WORDS) + 5 + 2)  # base + function + POS + readability
    
    # base features for model 1 referenced in blurb below
    sentence_lengths = [len(s.split()) for s in sentences if len(s.split())>0]
    base_features = [
        np.mean(sentence_lengths) if sentence_lengths else 0,
        np.std(sentence_lengths) if sentence_lengths else 0,
        np.mean([len(w) for w in words]),
        len(set(words))/len(words),
        sum(1 for w in words_raw if w.isupper())/len(words),
        text.count('!')/len(words),
        text.count('?')/len(words),
        len(words)
    ]

    # additional features for model 3
    punct_counts = sum(text.count(p) for p in [';', ':', '"', "'"])
    punct_diversity = punct_counts / len(words) if len(words) > 0 else 0
    
    stopword_ratio = sum(1 for w in words if w in STOPWORDS) / len(words) if len(words) > 0 else 0
    
    long_sent_ratio = sum(1 for l in sentence_lengths if l>20)/len(sentence_lengths) if sentence_lengths else 0
    
    features = base_features \
               + function_word_features(words) \
               + pos_features(words) \
               + readability_features(text) \
               + [punct_diversity, stopword_ratio, long_sent_ratio]
    
    return features