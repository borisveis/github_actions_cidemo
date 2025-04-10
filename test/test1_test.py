import pytest
from sklearn.metrics.pairwise import cosine_similarity
from tfIdfInheritVectorizer.feature_extraction.vectorizer import TFIDFVectorizer


def add(a, b):
    return a + b

def cosine_similarity_score(sentence1, sentence2):
    vectorizer = TFIDFVectorizer()
    vectors = vectorizer.fit_transform([sentence1, sentence2])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return similarity


def test_add():
    assert 2 == add(1, 1)
def test_cs():
    assert cosine_similarity_score("This is a sentence","This is another sentence")>0.5
