"""
main.py — точка входа командной строки (CLI).

Запуск:
    python main.py --corpus corpus.csv --query "любовь осень"
    python main.py --corpus corpus.csv --query "война" --index bm25_matrix --top 5
    python main.py --corpus corpus.csv --query "грусть" --index word2vec
    python main.py --corpus corpus.csv --query "осень" --index navec \\
                   --navec-model navec_hudlit_v1_12B_500K_300d_100q.tar
"""

import argparse
import os
import sys

from search_engine import INDEX_LABELS, SearchEngine


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Поиск по корпусу текстов: TF-IDF / BM-25 / Word2Vec / Navec",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python main.py --corpus corpus.csv --query "любовь природа"
  python main.py --corpus corpus.csv --query "война" --index bm25_matrix --top 5
  python main.py --corpus corpus.csv --query "осень" --index word2vec
  python main.py --corpus corpus.csv --query "грусть" --index navec \\
                 --navec-model navec_hudlit_v1_12B_500K_300d_100q.tar

Типы индексов:
  freq_dict    — TF-IDF через словари 
  bm25_dict    — BM-25 через словари
  freq_matrix  — TF-IDF через матрицы scipy
  bm25_matrix  — BM-25 через матрицы scipy
  word2vec     — Word2Vec, обученный на корпусе
  navec        — Navec предобученные векторы
        """,
    )
    parser.add_argument("--corpus", required=True,
                        help="Путь к CSV-файлу корпуса")
    parser.add_argument("--query", required=True,
                        help="Текст поискового запроса")
    parser.add_argument(
        "--index", default="bm25_dict",
        choices=list(INDEX_LABELS.keys()),
        help="Тип индекса (по умолчанию: bm25_dict)",
    )
    parser.add_argument("--top", type=int, default=10,
                        help="Количество результатов (по умолчанию: 10)")
    parser.add_argument("--column", default="text",
                        help="Столбец с текстами в CSV (по умолчанию: text)")
    parser.add_argument("--sample", type=int, default=1500,
                        help="Максимум документов (по умолчанию: 1500)")
    parser.add_argument("--navec-model", default=None, dest="navec_model",
                        help="Путь к .tar файлу Navec (для --index navec)")
    return parser.parse_args()


def main() -> None:
    """Главная функция CLI."""
    args = parse_args()

    if not os.path.isfile(args.corpus):
        print(f"Ошибка Файл не найден: {args.corpus}")
        sys.exit(1)

    print(f"  Поисковая система по корпусу текстов")
    print(f"  Корпус : {args.corpus}")
    print(f"  Запрос : {args.query}")
    print(f"  Индекс : {INDEX_LABELS[args.index]}  |  Топ-{args.top}")

    engine = SearchEngine(
        csv_path=args.corpus,
        text_column=args.column,
        sample_size=args.sample,
        navec_model_path=args.navec_model,
    )
    engine.build_indexes(which=[args.index])
    results, elapsed_ms = engine.search(args.query, args.index, args.top)
    engine.print_results(results, elapsed_ms)


if __name__ == "__main__":
    main()
