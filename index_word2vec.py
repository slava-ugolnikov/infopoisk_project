"""
index_word2vec.py — поиск на основе Word2Vec эмбеддингов.

Модель: Word2Vec (CBOW), реализация через gensim.
Обучается непосредственно на загруженном корпусе при первом вызове build().

Сходство: косинусное расстояние между усреднёнными векторами слов.

Установка gensim: pip install gensim
"""

from typing import Optional
import numpy as np

try:
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    Word2Vec = None


def _cosine_scores(matrix: np.ndarray, query_vec: np.ndarray) -> np.ndarray:
    """Векторизованное косинусное сходство query_vec со всеми строками matrix."""
    norms = np.linalg.norm(matrix, axis=1)
    norms = np.where(norms == 0, 1e-10, norms)
    q_norm = np.linalg.norm(query_vec)
    if q_norm == 0:
        return np.zeros(len(matrix))
    return matrix.dot(query_vec) / (norms * q_norm)


class Word2VecIndex:
    """
    Поисковый индекс на основе Word2Vec.

    Каждый документ — среднее векторов его слов.
    Запрос — среднее векторов слов запроса.
    Ранжирование — косинусное сходство.
    """

    def __init__(self, vector_size: int = 100, window: int = 5,
                 min_count: int = 2, workers: int = 4, epochs: int = 10):
        """
        Параметры:
        vector_size (int): Размерность векторов (по умолчанию 100).
        window (int): Размер контекстного окна (по умолчанию 5).
        min_count (int): Минимальная частота слова для включения в словарь.
        workers (int): Число потоков обучения.
        epochs (int): Число эпох обучения.
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs

        self.model = None
        # Матрица документных векторов: shape (num_docs, vector_size)
        self.doc_vectors = None

    def _tokens_to_vector(self, tokens: list[str]) -> np.ndarray:
        """Усредняет векторы слов; слова не из словаря пропускаются."""
        vecs = [self.model.wv[t] for t in tokens if t in self.model.wv]
        if not vecs:
            return np.zeros(self.vector_size)
        return np.mean(vecs, axis=0)

    def build(self, processed_docs: list[list[str]]) -> None:
        """
        Обучает Word2Vec на корпусе и построить матрицу документных векторов.

        Параметры:
        processed_docs (list[list[str]]): Предобработанные документы.
        """
        if not GENSIM_AVAILABLE:
            raise ImportError(
                "gensim не установлен. Установите: pip install gensim"
            )

        print("Word2VecIndex Обучаем Word2Vec...")
        self.model = Word2Vec(
            sentences=processed_docs,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
        )

        print("Word2VecIndex Индексируем документы...")
        self.doc_vectors = np.vstack(
            [self._tokens_to_vector(doc) for doc in processed_docs]
        )

        print(f"Word2VecIndex Построен. "
              f"Словарь: {len(self.model.wv)} слов, "
              f"матрица: {self.doc_vectors.shape}")

    def search(self, query_tokens: list[str],
               top_k: int = 10) -> list[tuple[int, float]]:
        """
        Находит топ-k документов по косинусному сходству.

        Возвращает [(doc_id, score)].
        """
        query_vec = self._tokens_to_vector(query_tokens)
        scores = _cosine_scores(self.doc_vectors, query_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]
