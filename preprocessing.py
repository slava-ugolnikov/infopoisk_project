"""
preprocessing.py — предобработка текста корпуса.

Включает:
- очистку от пунктуации и цифр
- приведение к нижнему регистру
- лемматизацию (через pymorphy3)
- удаление стоп-слов (через nltk + расширенный список)
"""

import re
import nltk
import pymorphy3
from nltk.corpus import stopwords

morph = pymorphy3.MorphAnalyzer()
STOPWORDS = set(stopwords.words("russian"))

def lemmatize_word(word: str) -> str:
    """Лемматизирует слово через pymorphy3."""
    return morph.parse(word)[0].normal_form


def clean_text(text: str) -> str:
    """Убирает всё, кроме кириллицы и пробелов, приводит к нижнему регистру."""
    text = text.lower()
    text = re.sub(r"[^а-яёa-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Разбивает текст на токены по пробелам."""
    return text.split()


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Удалить стоп-слова и слишком короткие токены (< 3 символов)."""
    return [t for t in tokens if t not in STOPWORDS and len(t) >= 3]


def preprocess(text: str) -> list[str]:
    """
    Полный пайплайн предобработки одного текста.
    Возвращает список лемм.
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    lemmas = [lemmatize_word(token) for token in tokens]
    return lemmas
