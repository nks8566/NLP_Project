import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
import re


# Initialize VADER
vader = SentimentIntensityAnalyzer()

# Feature extraction function
def extract_emotionality_features1(article_text):
    features = {}
    
    # Sentence-level analysis
    sentences = sent_tokenize(article_text)
    sent_scores = [vader.polarity_scores(s) for s in sentences]
    
    # Aggregate features
    features['compound_mean'] = np.mean([s['compound'] for s in sent_scores])
    features['compound_std'] = np.std([s['compound'] for s in sent_scores])
    features['pos_proportion'] = np.mean([s['pos'] for s in sent_scores])
    features['neg_proportion'] = np.mean([s['neg'] for s in sent_scores])
    features['neu_proportion'] = np.mean([s['neu'] for s in sent_scores])
    
    # Sentiment trajectory
    features['sentiment_change'] = sent_scores[-1]['compound'] - sent_scores[0]['compound']
    
    # Intensity features
    features['max_positive'] = max([s['pos'] for s in sent_scores])
    features['max_negative'] = max([s['neg'] for s in sent_scores])
    
    return features

def extract_emotionality_features(text):
    """Extract emotionality features from one article"""
    features = []
    
    # Basic VADER scores
    scores = vader.polarity_scores(text)
    features.extend([scores['compound'], scores['pos'], scores['neg'], scores['neu']])
    
    # Sentence-level features
    sentences = sent_tokenize(text)
    if len(sentences) > 0:
        sent_compounds = [vader.polarity_scores(s)['compound'] for s in sentences]
        features.extend([
            np.mean(sent_compounds),
            np.std(sent_compounds),
            max(sent_compounds),
            min(sent_compounds)
        ])
    else:
        features.extend([0, 0, 0, 0])
    
    # Linguistic markers
    word_count = len(text.split())
    features.extend([
        text.count('!') / word_count if word_count > 0 else 0,
        text.count('?') / word_count if word_count > 0 else 0,
        len(re.findall(r'\b[A-Z]{2,}\b', text)) / word_count if word_count > 0 else 0
    ])
    
    return features


