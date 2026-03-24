"""
search_engine.py — ядро поискового движка.

Объединяет все индексы из ДЗ №1 и ДЗ №2:
    freq_dict — TF-IDF через словари
    bm25_dict — BM-25 через словари
    freq_matrix — TF-IDF через матрицы scipy
    bm25_matrix — BM-25 через матрицы scipy
    word2vec — Word2Vec (gensim)
    navec — Navec предобученные векторы

Единая точка входа из Python:

    from search_engine import SearchEngine

    engine = SearchEngine("corpus.csv")
    engine.build_indexes(["bm25_dict", "word2vec"])
    results, ms = engine.search("любовь осень", index_type="bm25_dict")
    engine.print_results(results, ms)

Или через функцию-обёртку:

    from search_engine import search
    results, ms = search("corpus.csv", "любовь осень", index_type="bm25_dict")
"""

import time
from typing import Dict, List, Literal, Optional, Tuple

import pandas as pd

from corpus import load_corpus
from index_dict import BM25IndexDict, FrequencyIndexDict
from index_matrix import BM25IndexMatrix, FrequencyIndexMatrix
from index_navec import NavecIndex
from index_word2vec import Word2VecIndex
from preprocessing import preprocess


# Все допустимые типы индексов
IndexType = Literal[
    "freq_dict", "bm25_dict",
    "freq_matrix", "bm25_matrix",
    "word2vec", "navec",
]

# Читаемые названия для интерфейса
INDEX_LABELS: Dict[str, str] = {
    "freq_dict":   "TF-IDF (словари)",
    "bm25_dict":   "BM-25 (словари)",
    "freq_matrix": "TF-IDF (матрицы)",
    "bm25_matrix": "BM-25 (матрицы)",
    "word2vec":    "Word2Vec",
    "navec":       "Navec",
}

# Индексы, доступные без внешних моделей
LIGHTWEIGHT_INDEXES = ["freq_dict", "bm25_dict", "freq_matrix", "bm25_matrix"]


class SearchResult:
    """Один результат поиска с метаданными документа."""

    def __init__(self, rank: int, doc_id: int, original_id: int,
                 score: float, text: str,
                 author: str = "", title: str = ""):
        self.rank = rank
        self.doc_id = doc_id
        self.original_id = original_id
        self.score = score
        self.text = text
        self.author = author
        self.title = title

    def to_dict(self) -> dict:
        """Сериализовать в словарь (для JSON-ответов Flask)."""
        return {
            "rank":        self.rank,
            "doc_id":      self.doc_id,
            "original_id": self.original_id,
            "score":       round(self.score, 4),
            "text":        self.text,
            "author":      self.author,
            "title":       self.title,
        }


class SearchEngine:
    """
    Поисковый движок по корпусу текстов.

    Поддерживает шесть типов индексов (см. INDEX_LABELS).
    Замеряет время поиска в миллисекундах.
    """

    def __init__(self, csv_path: str, text_column: str = "text",
                 sample_size: int = 1500,
                 navec_model_path: Optional[str] = None):
        """
        Параметры:
        csv_path (str):  Путь к CSV-файлу с корпусом.
        text_column (str): Название столбца с текстами.
        sample_size (int): Максимальное число документов для индексации.
        navec_model_path (str, optional): Путь к .tar файлу модели Navec.
        """
        self.csv_path = csv_path
        self.text_column = text_column
        self.sample_size = sample_size
        self.navec_model_path = navec_model_path

        self.raw_docs = []
        self.processed_docs = []
        self.original_ids = []
        self.df = None

        # Хранилище построенных индексов: name -> объект индекса
        self.indexes: Dict[str, object] = {}


    def load(self) -> None:
        """Загружает и предобрабатывает корпус из CSV."""
        self.raw_docs, self.processed_docs, self.df, self.original_ids = \
            load_corpus(self.csv_path, self.text_column, self.sample_size)


    def build_indexes(self,
                      which: Optional[List[str]] = None) -> None:
        """
        Построить указанные индексы.

        Параметры:
        which (list[str], optional): Список имён индексов. По умолчанию — все четыре «лёгких»:
            ["freq_dict", "bm25_dict", "freq_matrix", "bm25_matrix"].
        """
        if not self.raw_docs:
            self.load()

        if which is None:
            which = LIGHTWEIGHT_INDEXES

        print(f"\n Построение индексов: {which}")

        builders = {
            "freq_dict":   lambda: FrequencyIndexDict(),
            "bm25_dict":   lambda: BM25IndexDict(),
            "freq_matrix": lambda: FrequencyIndexMatrix(),
            "bm25_matrix": lambda: BM25IndexMatrix(),
            "word2vec":    lambda: Word2VecIndex(),
            "navec":       lambda: NavecIndex(model_path=self.navec_model_path),
        }

        for name in which:
            if name not in builders:
                print(f"Неизвестный индекс '{name}', пропускаем.")
                continue
            try:
                idx = builders[name]()
                idx.build(self.processed_docs)
                self.indexes[name] = idx
            except ImportError as e:
                print(f"Индекс '{name}' пропущен: {e}")
            except Exception as e:
                print(f"Ошибка при построении '{name}': {e}")

        print("Все индексы построены\n")

    def search(self, query: str,
               index_type: str = "bm25_dict",
               top_k: int = 10) -> Tuple[List[SearchResult], float]:
        """
        Выполнить поиск по запросу и вернуть результаты со временем.

        Параметры:
        query (str): Текст запроса на русском языке.
        index_type (str): Тип индекса (ключ из INDEX_LABELS).
        top_k (int): Количество результатов.

        Возвращает:
        results (list[SearchResult])
        elapsed_ms (float): Время поиска в миллисекундах.
        """
        if index_type not in self.indexes:
            raise ValueError(
                f"Индекс '{index_type}' не построен. "
                f"Доступны: {list(self.indexes.keys())}"
            )

        query_tokens = preprocess(query)
        if not query_tokens:
            print("Запрос пуст после предобработки.")
            return [], 0.0

        print(f"'{query}' | токены: {query_tokens} | "
              f"индекс: {INDEX_LABELS.get(index_type, index_type)}")

        # Замеряем только время самого поиска по индексу
        t0 = time.perf_counter()
        raw_results = self.indexes[index_type].search(query_tokens, top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        results = []
        for rank, (doc_id, score) in enumerate(raw_results, start=1):
            original_id = self.original_ids[doc_id]
            text = self.raw_docs[doc_id]
            author, title = "", ""
            if self.df is not None and doc_id < len(self.df):
                row = self.df.iloc[doc_id]
                a = row.get("author", "")
                n = row.get("name", "")
                author = str(a) if pd.notna(a) else ""
                title = str(n) if pd.notna(n) else ""
            results.append(
                SearchResult(rank, doc_id, original_id, score, text, author, title)
            )

        print(f"Найдено: {len(results)}, время: {elapsed_ms:.2f} мс")
        return results, elapsed_ms

    def print_results(self, results: List[SearchResult],
                      elapsed_ms: float = 0.0,
                      preview_len: int = 220) -> None:
        """Красиво вывести результаты поиска в терминал."""
        if elapsed_ms:
            print(f"  Время поиска: {elapsed_ms:.2f} мс  |  "
                  f"Результатов: {len(results)}")

        if not results:
            print("  Ничего не найдено.")
            return

        for r in results:
            preview = r.text[:preview_len].replace("\n", " ")
            if len(r.text) > preview_len:
                preview += "..."
            meta = ""
            if r.author:
                meta += r.author
            if r.title:
                meta += f" «{r.title}»"
            print(f"\n  #{r.rank}  score={r.score:.4f}  "
                  f"[orig_id={r.original_id}]")
            if meta:
                print(f"  {meta}")
            print(f"  {preview}")



def search(csv_path: str,
           query: str,
           index_type: str = "bm25_dict",
           top_k: int = 10,
           text_column: str = "text",
           sample_size: int = 1500,
           navec_model_path: Optional[str] = None
           ) -> Tuple[List[SearchResult], float]:
    """
    Создает движок, построить нужный индекс и вернуть результаты.

    Параметры:
    csv_path (str): Путь к CSV-файлу корпуса.
    query (str): Текст запроса.
    index_type (str): Тип индекса: "freq_dict", "bm25_dict", "freq_matrix",
        "bm25_matrix", "word2vec", "navec".
    top_k (int): Количество результатов.
    text_column (str): Название столбца с текстами в CSV.
    sample_size (int):Максимальное число документов.
    navec_model_path (str, optional) Путь к .tar файлу Navec (нужен только для index_type="navec").

    Пример
    >>> results, ms = search("corpus.csv", "любовь осень", index_type="bm25_dict")
    """
    engine = SearchEngine(csv_path, text_column, sample_size, navec_model_path)
    engine.build_indexes(which=[index_type])
    results, elapsed_ms = engine.search(query, index_type, top_k)
    engine.print_results(results, elapsed_ms)
    return results, elapsed_ms
