"""
index_dict.py — обратные индексы через словари (частотный и BM-25).

FrequencyIndexDict — TF-IDF через словарь
BM25IndexDict — BM-25 через словарь
"""

import math
from collections import defaultdict


class FrequencyIndexDict:
    """
    TF-IDF через словарь.
    """

    def __init__(self):
        self.inverted_index = defaultdict(dict)
        self.idf = {}
        self.num_docs = 0

    def build(self, processed_docs: list[list[str]]) -> None:
        """
        Строит индекс по корпусу предобработанных документов.

        Параметры:
        processed_docs (list[list[str]]): Список документов, каждый из которых является списком лемм.
        """
        self.num_docs = len(processed_docs)

        # TF
        for doc_id, tokens in enumerate(processed_docs):
            if not tokens:
                continue
            term_count = defaultdict(int)
            for token in tokens:
                term_count[token] += 1
            doc_len = len(tokens)
            for term, count in term_count.items():
                tf = count / doc_len
                self.inverted_index[term][doc_id] = tf

        # IDF
        for term, doc_map in self.inverted_index.items():
            df = len(doc_map)
            self.idf[term] = math.log((self.num_docs + 1) / (df + 1)) + 1

        print(f"FrequencyIndexDict Построен. Термины: {len(self.inverted_index)}")

    def score(self, query_tokens: list[str], doc_id: int) -> float:
        """Считает TF-IDF score документа по токенам запроса."""
        total = 0.0
        for term in query_tokens:
            if term in self.inverted_index:
                tf = self.inverted_index[term].get(doc_id, 0.0)
                idf = self.idf.get(term, 0.0)
                total += tf * idf
        return total

    def search(self, query_tokens: list[str], top_k: int = 10) -> list[tuple[int, float]]:
        """
        Находит топ-k документов по запросу.

        Возвращает список пар (doc_id, score), отсортированных по убыванию.
        """
        # Собираем кандидатов — документы, содержащие хотя бы одно слово
        candidate_ids = set()
        for term in query_tokens:
            if term in self.inverted_index:
                candidate_ids.update(self.inverted_index[term].keys())

        scores = [
            (doc_id, self.score(query_tokens, doc_id))
            for doc_id in candidate_ids
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class BM25IndexDict:
    """
    Обратный индекс на основе BM-25, реализованный через словари.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        # term -> {doc_id: raw_frequency}
        self.inverted_index = defaultdict(dict)
        self.doc_lengths= {}
        self.avgdl = 0.0
        self.num_docs = 0
        self.idf = {}

    def build(self, processed_docs: List[List[str]]) -> None:
        """Строит BM-25 индекс по предобработанному корпусу."""
        self.num_docs = len(processed_docs)

        for doc_id, tokens in enumerate(processed_docs):
            self.doc_lengths[doc_id] = len(tokens)
            term_count = defaultdict(int)
            for token in tokens:
                term_count[token] += 1
            for term, count in term_count.items():
                self.inverted_index[term][doc_id] = count

        total_len = sum(self.doc_lengths.values())
        self.avgdl = total_len / self.num_docs if self.num_docs > 0 else 1.0

        for term, doc_map in self.inverted_index.items():
            df = len(doc_map)
            self.idf[term] = math.log(
                (self.num_docs - df + 0.5) / (df + 0.5) + 1
            )

        print(f"BM25IndexDict Построен. Термины: {len(self.inverted_index)}, "
              f"avgdl={self.avgdl:.1f}")

    def _score_doc(self, query_tokens: List[str], doc_id: int) -> float:
        """Вычисляет BM-25 score для одного документа."""
        doc_len = self.doc_lengths.get(doc_id, 0)
        total = 0.0
        for term in query_tokens:
            if term not in self.inverted_index:
                continue
            freq = self.inverted_index[term].get(doc_id, 0)
            if freq == 0:
                continue
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (
                1 - self.b + self.b * doc_len / max(self.avgdl, 1)
            )
            total += self.idf[term] * numerator / denominator
        return total

    def search(self, query_tokens: list[str],
               top_k: int = 10) -> list[tuple[int, float]]:
        """Находит топ-k документов по BM-25. Возвращает [(doc_id, score)]."""
        candidates = set()
        for term in query_tokens:
            if term in self.inverted_index:
                candidates.update(self.inverted_index[term].keys())

        scores = [
            (doc_id, self._score_doc(query_tokens, doc_id))
            for doc_id in candidates
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]