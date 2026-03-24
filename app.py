"""
app.py — веб-интерфейс на Flask.

Страницы:
    /          — главная с описанием проекта и кнопкой на поиск
    /search    — форма поиска (запрос, тип индекса, топ-N)
    /results   — результаты со списком документов и временем поиска

Запуск:
    python app.py --corpus corpus.csv
    python app.py --corpus corpus.csv --indexes bm25_dict word2vec --port 5000
    python app.py --corpus corpus.csv --indexes bm25_dict navec \\
                  --navec-model navec_hudlit_v1_12B_500K_300d_100q.tar
"""

import argparse
import os
import sys

from flask import Flask, redirect, render_template, request, url_for

from search_engine import INDEX_LABELS, SearchEngine

app = Flask(__name__)

# Глобальный движок создаётся один раз при старте сервера
_engine: SearchEngine = None


def get_engine() -> SearchEngine:
    return _engine


# Маршруты
@app.route("/")
def index():
    """Главная страница."""
    return render_template("index.html")


@app.route("/search")
def search_form():
    """Страница с формой поиска."""
    engine = get_engine()
    available = list(engine.indexes.keys())
    # Передаём все метки, чтобы показать недоступные как disabled
    index_options = list(INDEX_LABELS.items())
    return render_template(
        "search.html",
        index_options=index_options,
        available=available,
    )


@app.route("/results")
def results():
    """Страница результатов поиска."""
    query = request.args.get("query", "").strip()
    index_type = request.args.get("index_type", "bm25_dict")
    top_k = int(request.args.get("top_k", 10))

    if not query:
        return redirect(url_for("search_form"))

    engine = get_engine()

    if index_type not in engine.indexes:
        label = INDEX_LABELS.get(index_type, index_type)
        error = (f"Индекс «{label}» не загружен на сервере. "
                 f"Перезапустите с параметром --indexes {index_type}.")
        return render_template("results.html", error=error, query=query,
                               results=[], elapsed_ms=0, index_label="")

    search_results, elapsed_ms = engine.search(query, index_type, top_k)

    return render_template(
        "results.html",
        query=query,
        index_label=INDEX_LABELS[index_type],
        results=[r.to_dict() for r in search_results],
        elapsed_ms=round(elapsed_ms, 2),
        top_k=top_k,
        error=None,
    )


# Запуск

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Веб-интерфейс поиска")
    parser.add_argument("--corpus", required=True,
                        help="Путь к CSV-файлу корпуса")
    parser.add_argument("--sample", type=int, default=1500,
                        help="Максимум документов (по умолчанию: 1500)")
    parser.add_argument("--port", type=int, default=5000,
                        help="Порт сервера (по умолчанию: 5000)")
    parser.add_argument(
        "--indexes", nargs="+",
        default=["bm25_dict", "word2vec"],
        choices=list(INDEX_LABELS.keys()),
        help="Индексы для загрузки (по умолчанию: bm25_dict word2vec)",
    )
    parser.add_argument("--navec-model", default=None, dest="navec_model",
                        help="Путь к .tar файлу Navec")
    parser.add_argument("--debug", action="store_true",
                        help="Режим отладки Flask")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.isfile(args.corpus):
        print(f"Файл не найден: {args.corpus}")
        sys.exit(1)

    print(f"\nЗапускаем сервер, строим индексы: {args.indexes} ...")
    _engine = SearchEngine(
        csv_path=args.corpus,
        sample_size=args.sample,
        navec_model_path=args.navec_model,
    )
    _engine.build_indexes(which=args.indexes)

    print(f"\n  Сервер запущен: http://localhost:{args.port}\n")
    app.run(debug=args.debug, port=args.port)
