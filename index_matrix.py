"""
index_matrix.py — обратные индексы через матрицы (частотный и BM-25).

FrequencyIndexMatrix — TF-IDF через матрицы
BM25IndexMatrix — BM-25 через матрицы
"""

import numpy as np
from collections import defaultdict

class FrequencyIndexMatrix:
    """
    TF-IDF индекс на основе разреженной матрицы терм-документ.

    Матрица shape: (num_terms, num_docs)
    Каждая ячейка: tf_idf(term, doc)
    """
    def __init__(self):
        self.vocab = {}
        self.idf = None
        self.tfidf_matrix = None
        self.num_docs = 0

    def build(self, processed_docs: list[list[str]]) -> None:
        """
        Строит TF-IDF матрицу по корпусу.

        Параметры:
        processed_docs (list[list[str]]): Список документов, каждый документ — список лемм.
        """
        self.num_docs = len(processed_docs)

        for doc in processed_docs:
            for token in doc:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

        num_terms = len(self.vocab)

        # TF
        rows, cols, data = [], [], []
        for doc_id, tokens in enumerate(processed_docs):
            if not tokens:
                continue
            term_count = defaultdict(int)
            for token in tokens:
                term_count[token] += 1
            doc_len = len(tokens)
            for term, count in term_count.items():
                term_id = self.vocab[term]
                tf = count / doc_len
                rows.append(term_id)
                cols.append(doc_id)
                data.append(tf)

        tf_matrix = np.zeros((num_terms, self.num_docs))
        for r, c, d in zip(rows, cols, data):
            tf_matrix[r, c] = d

        # IDF: log((N+1)/(df+1)) + 1
        df = np.count_nonzero(tf_matrix, axis=1)

        self.idf = np.log((self.num_docs + 1) / (df + 1)) + 1

        # TF-IDF
        self.tfidf_matrix = tf_matrix * self.idf[:, np.newaxis]

        print(f"FrequencyIndexMatrix построен. "
              f"Матрица: {num_terms}x{self.num_docs}")

    def search(self, query_tokens: list[str], top_k: int = 10) -> list[tuple[int, float]]:
        """
        Найти топ-k документов.
        Запрос преобразуется в вектор TF-IDF, затем вычисляется косинусное сходство с матрицей.

        Возвращает список пар (doc_id, score).
        """
        num_terms = len(self.vocab)
        query_vec = np.zeros(num_terms)

        # TF запроса
        term_count = defaultdict(int)
        for token in query_tokens:
            term_count[token] += 1
        q_len = max(len(query_tokens), 1)

        for term, count in term_count.items():
            if term in self.vocab:
                term_id = self.vocab[term]
                query_vec[term_id] = (count / q_len) * self.idf[term_id]

        # сходство
        scores = self.tfidf_matrix.T.dot(query_vec)

        # Топ-k по убыванию
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]


class BM25IndexMatrix:
    """
    BM-25 индекс на основе разреженной матрицы терм-документ.

    Параметры BM-25:
        k1 = 1.5
        b = 0.75
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vocab = {}
        self.idf = None
        self.bm25_matrix = None
        self.num_docs = 0

    def build(self, processed_docs: list[list[str]]) -> None:
        """
        Строит BM-25 матрицу по корпусу.

        Параметры:
        processed_docs (list[list[str]]): Список документов, каждый документ — список лемм.
        """
        self.num_docs = len(processed_docs)

        # Словарь
        for doc in processed_docs:
            for token in doc:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

        num_terms = len(self.vocab)

        # Длины документов и средняя длина
        doc_lengths = np.array([len(doc) for doc in processed_docs], dtype=float)
        avgdl = doc_lengths.mean() if self.num_docs > 0 else 1.0

        # Строим матрицу сырых частот (term freq)
        rows, cols, freq_data = [], [], []
        for doc_id, tokens in enumerate(processed_docs):
            if not tokens:
                continue
            term_count = defaultdict(int)
            for token in tokens:
                term_count[token] += 1
            for term, count in term_count.items():
                rows.append(self.vocab[term])
                cols.append(doc_id)
                freq_data.append(float(count))

        freq_matrix = np.zeros((num_terms, self.num_docs))
        for r, c, d in zip(rows, cols, freq_data):
            freq_matrix[r, c] = d

        # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        df = np.count_nonzero(freq_matrix, axis=1)

        self.idf = np.log((self.num_docs - df + 0.5) / (df + 0.5) + 1)

        # Нормализующий коэффициент для каждого документа
        # norm[doc] = k1 * (1 - b + b * doc_len / avgdl)
        norm = self.k1 * (1 - self.b + self.b * doc_lengths / avgdl)

        # BM-25 score для каждой ячейки (term, doc):
        # bm25 = freq * (k1 + 1) / (freq + norm[doc])
        # Применяем поэлементно
        bm25_raw = freq_matrix * (self.k1 + 1) / (freq_matrix + norm[np.newaxis, :])
        self.bm25_matrix = bm25_raw * self.idf[:, np.newaxis]

        print(f"BM25IndexMatrix построен. "
              f"Матрица: {num_terms}x{self.num_docs}, avgdl={avgdl:.1f}")

    def search(self, query_tokens: list[str], top_k: int = 10) -> list[tuple[int, float]]:
        """
        Находит топ-k документов.
        Возвращает список пар (doc_id, score).
        """
        num_terms = len(self.vocab)
        query_vec = np.zeros(num_terms)

        for token in set(query_tokens):
            if token in self.vocab:
                query_vec[self.vocab[token]] = 1.0

        # Суммируем BM-25 веса по терминам запроса
        scores = self.bm25_matrix.T.dot(query_vec)

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
