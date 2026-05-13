import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.tokenize import sent_tokenize
from textblob import TextBlob
from nrclex import NRCLex
import pandas as pd
import re

from joblib import Parallel, delayed


# Initialize VADER
vader = SentimentIntensityAnalyzer()

NRC_EMOTIONS = ['fear', 'anger', 'trust', 'surprise', 'sadness',
                'disgust', 'joy', 'anticipation', 'positive', 'negative']

def extract_emotionality_features(text: str) -> dict:
    """Extract emotionality features from one article"""
    features = {}
    words = text.split()
    word_count = len(words)
    
    # Basic VADER scores -- at the article level
    scores = vader.polarity_scores(text)
    features['vader_compound'] = scores['compound']
    features['vader_pos'] = scores['pos']
    features['vader_neg'] = scores['neg']
    features['vader_neu'] = scores['neu']
    
    # VADER scores -- sentence level aggregates. Includes standard dev, max, min for each article
    sentences = sent_tokenize(text)
    if len(sentences) > 0:
        sent_compounds = [vader.polarity_scores(s)['compound'] for s in sentences]
        features['vader_sent_mean'] = np.mean(sent_compounds)
        features['vader_sent_std'] = np.std(sent_compounds)
        features['vader_sent_max'] = max(sent_compounds)
        features['vader_sent_min'] = min(sent_compounds)
        
    else:
        features['vader_sent_mean'] = 0
        features['vader_sent_std'] = 0
        features['vader_sent_max'] = 0
        features['vader_sent_min'] = 0
    
    # Linguistic markers
    word_count = len(text.split())
    features['exclamation_ratio'] = text.count('!') / word_count if word_count > 0 else 0
    features['question_ratio'] = text.count('?') / word_count if word_count > 0 else 0
    features['all_caps_ratio'] = len(re.findall(r'\b[A-Z]{2,}\b', text)) / word_count if word_count > 0 else 0
    
    # Subjectivity score -- from textblob
    features['subjectivity'] = TextBlob(text).sentiment.subjectivity

    # Plutchik emotion scores -- using NRCLex
    nrc = NRCLex()
    nrc.load_raw_text(text[:2000])
    raw_freq = nrc.raw_emotion_scores
    total_nrc = sum(raw_freq.values()) or 1
    for emotion in NRC_EMOTIONS:
        features[f'nrc_{emotion}'] = raw_freq.get(emotion, 0) / total_nrc


    return features




