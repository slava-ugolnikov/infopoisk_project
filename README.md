# Проект — Поисковая система по корпусу текстов

## Корпус

Русскоязычные поэтические тексты (~16 000 записей, используется выборка 1 500).  
Столбцы: `author`, `name`, `text`, `date_from`, `date_to`, `themes/item/0..3`.

**Почему этот корпус?**  
Поэтические тексты могут быть интересны для информационного поиска тем, что они художественно, лексически насыщены.

## Структура проекта

```
project/
├── main.py             # CLI-точка входа
├── app.py              # Flask веб-сервер
├── search_engine.py    # SearchEngine + функция search()
├── corpus.py           # Загрузка CSV и вызов предобработки
├── preprocessing.py    # Препроцессинг
│
├── index_dict.py       # TF-IDF и BM-25 через словари Python     
├── index_matrix.py     # TF-IDF и BM-25 через матрицы scipy      
├── index_word2vec.py   # Word2Vec (gensim, обучается на корпусе) 
├── index_navec.py      # Navec (предобученные русские векторы)  
│
└── templates/
    ├── base.html       # Базовый шаблон
    ├── index.html      # Главная страница
    ├── search.html     # Форма поиска
    └── results.html    # Результаты поиска
```


## Установка зависимостей

```bash
pip install pandas numpy scipy flask gensim pymorphy2 nltk navec
```

Скачать модель Navec (~50 MB, нужна только для индекса `navec`):
```bash
wget https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar
```


## Модели

### TF-IDF (freq_dict, freq_matrix)
Реализованы вручную. 
`freq_dict` — через словари Python; `freq_matrix` — через `scipy.sparse`.

### BM-25 (bm25_dict, bm25_matrix)
Реализованы вручную. 
`bm25_dict` — через словари Python; `bm25_matrix` — через `scipy.sparse`.

### Word2Vec (word2vec)
- Word2Vec (CBOW), реализация через библиотеку `gensim`
- обучается непосредственно на загруженном корпусе при запуске
- vector_size=100, window=5, min_count=2, epochs=10
- косинусное расстояние между усреднёнными векторами слов

### Navec (navec)
- предобученные русскоязычные word-векторы от проекта Natasha
- `navec_hudlit_v1_12B_500K_300d_100q.tar`, Yandex Cloud
- 500k слов, 300 измерений, квантизованные веса, ~50 MB
- косинусное расстояние между усреднёнными векторами слов


## Предобработка текста

1. Очистка
2. Токенизация
3. Удаление стоп-слов
4. Удаление коротких токенов
5. Лемматизация
6. Повторная фильтрация стоп-слов среди лемм


## CLI

```bash
# TF-IDF, словари
python main.py --corpus corpus.csv --query "любовь осень"

# BM-25, матрицы
python main.py --corpus corpus.csv --query "война герои" --index bm25_matrix --top 5

# Word2Vec
python main.py --corpus corpus.csv --query "грусть разлука" --index word2vec

# Navec (нужен .tar файл)
python main.py --corpus corpus.csv --query "осень листья" --index navec --navec-model navec_hudlit_v1_12B_500K_300d_100q.tar
```

### Параметры CLI

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--corpus` | обязательный | Путь к CSV-файлу |
| `--query` | обязательный | Текст запроса |
| `--index` | `bm25_dict` | Тип индекса (см. ниже) |
| `--top` | `10` | Количество результатов |
| `--column` | `text` | Столбец с текстами в CSV |
| `--sample` | `1500` | Максимум документов |
| `--navec-model` | — | Путь к .tar файлу Navec |

Доступные значения `--index`:
`freq_dict`, `bm25_dict`, `freq_matrix`, `bm25_matrix`, `word2vec`, `navec`

## Веб-интерфейс (Flask)

```bash
# BM-25 + Word2Vec (по умолчанию)
python app.py --corpus corpus.csv

# Все шесть индексов
python app.py --corpus corpus.csv --indexes freq_dict bm25_dict freq_matrix bm25_matrix word2vec navec --navec-model navec_hudlit_v1_12B_500K_300d_100q.tar

# Другой порт и режим отладки
python app.py --corpus corpus.csv --port 8080 --debug
```

Открыть в браузере: **http://localhost:5000**

### Параметры веб-сервера

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--corpus` | обязательный | Путь к CSV-файлу |
| `--sample` | `1500` | Максимум документов |
| `--port` | `5000` | Порт сервера |
| `--indexes` | `bm25_dict word2vec` | Индексы для загрузки |
| `--navec-model` | — | Путь к .tar файлу Navec |
| `--debug` | выключен | Режим отладки Flask |

### Страницы сайта

| URL | Описание |
|---|---|
| `/` | Главная: описание проекта и моделей, кнопка на поиск |
| `/search` | Форма: запрос, тип индекса, топ-N |
| `/results` | Результаты: список документов, оценка сходства, время поиска |


## Использование из Python

```python
from search_engine import SearchEngine, search

# Один вызов — всё автоматически
results, elapsed_ms = search("corpus.csv", "любовь осень", index_type="bm25_dict")

# Или полный контроль
engine = SearchEngine("corpus.csv", sample_size=1500)
engine.build_indexes(which=["bm25_dict", "freq_matrix", "word2vec"])

results, ms = engine.search("осень листья", index_type="word2vec", top_k=5)
engine.print_results(results, ms)
```
