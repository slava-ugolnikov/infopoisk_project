"""
corpus.py — загрузка и подготовка корпуса .CSV.

Корпус: стихи/тексты из открытого датасета (author, date_from, text, name, date_to, themes/item/0..3).
Источник: https://www.kaggle.com/datasets/greencools/russianpoetry
"""

import pandas as pd
from preprocessing import preprocess


def load_corpus(
    csv_path: str,
    text_column: str = "text",
    sample_size: int = 1500,
) -> tuple[list[str], list[list[str]], pd.DataFrame, list[int]]:
    """
    Загружает корпус из csv-файла.

    Параметры:
    csv_path (str): Путь к csv-файлу.
    text_column (str): Название столбца с текстами.
    sample_size (int): Количество документов (для ограничения объёма).

    Возвращает:
    raw_docs (list[str]): Сырые тексты документов.
    processed_docs (list[list[str]]): Предобработанные документы (списки лемм).
    df (pd.DataFrame): Исходный датафрейм.
    original_ids (list[int]): Оригинальные индексы строк из CSV до сэмплирования.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[text_column]).reset_index(drop=False)

    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    raw_docs = df[text_column].astype(str).tolist()
    original_ids = df["index"].tolist()

    print(f"Загружено {len(raw_docs)} документов. Начинаем предобработку.")

    processed_docs = []
    for i, text in enumerate(raw_docs):
        processed_docs.append(preprocess(text))

    print("Предобработка завершена.")
    return raw_docs, processed_docs, df, original_ids
