"""
index_navec.py — поиск на основе эмбеддингов Navec.

Модель: Navec от проекта Natasha (https://github.com/natasha/navec)
Обучена на литературе и новостях
~500k слов, 300-мерные квантизованные векторы, ~50 MB
Специально для русского языка

Скачать модель: wget https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar

Установка: pip install navec

Сходство: косинусное расстояние между усреднёнными векторами слов.
"""

from typing import Optional
import numpy as np
from navec import Navec
NAVEC_AVAILABLE = True


# Имя файла модели по умолчанию (ищется в текущей директории)
DEFAULT_MODEL_PATH = "navec_hudlit_v1_12B_500K_300d_100q.tar"


def _cosine_scores(matrix: np.ndarray, query_vec: np.ndarray) -> np.ndarray:
    """Векторизованное косинусное сходство query_vec со всеми строками matrix."""
    norms = np.linalg.norm(matrix, axis=1)
    norms = np.where(norms == 0, 1e-10, norms)
    q_norm = np.linalg.norm(query_vec)
    if q_norm == 0:
        return np.zeros(len(matrix))
    return matrix.dot(query_vec) / (norms * q_norm)


class NavecIndex:
    """
    Поисковый индекс на основе предобученных векторов Navec.

    Каждый документ — среднее векторов его слов из Navec.
    Слова, отсутствующие в словаре Navec, пропускаются.
    Ранжирование — косинусное сходство.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Параметры:
        model_path (str, optional): Путь к .tar файлу Navec.
        По умолчанию: navec_hudlit_v1_12B_500K_300d_100q.tar
        """
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.navec = None
        self.vector_size: int = 300
        self.doc_vectors: Optional[np.ndarray] = None

    def _load_model(self) -> None:
        """Загрузить модель Navec из .tar файла."""
        if not NAVEC_AVAILABLE:
            raise ImportError
        print(f"NavecIndex Загружаем модель из '{self.model_path}'...")
        self.navec = Navec.load(self.model_path)
        self.vector_size = self.navec.pq.dim

    def _tokens_to_vector(self, tokens: list[str]) -> np.ndarray:
        """Усреднить векторы слов; слова не из словаря Navec пропускаются."""
        vecs = [self.navec[t] for t in tokens if t in self.navec]
        if not vecs:
            return np.zeros(self.vector_size)
        return np.mean(vecs, axis=0)

    def build(self, processed_docs: list[list[str]]) -> None:
        """
        Загрузить Navec и построить матрицу документных векторов.

        Параметры:
        processed_docs (list[list[str]]): Предобработанные документы.
        """
        self._load_model()

        print("NavecIndex Индексируем документы...")
        self.doc_vectors = np.vstack(
            [self._tokens_to_vector(doc) for doc in processed_docs]
        )

        print(f"NavecIndex Построен. "
              f"Матрица: {self.doc_vectors.shape}")

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
