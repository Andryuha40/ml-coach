# Работа с текстовыми данными и эмбеддинги (базовый уровень)

Конспект собран целиком из двух семинаров курса («Эмбеддинги для текста» и
«Преобразование текстовых данных») — у репозитория Евгения Паточенко
отдельного материала по этой теме нет, поэтому здесь нет слоя «репозиторий
как основа»: всё взято из локальных ноутбуков.

## Зачем текст превращать в числа

Алгоритмы машинного обучения (линейная регрессия, логистическая регрессия и
почти всё остальное) умеют работать только с числами — точнее, с векторами
чисел. Текст в исходном виде (строка символов) для них не подходит, поэтому
первый шаг любой работы с текстом — **векторизация**: превращение текста в
числовой вектор фиксированной длины.

## Bag of Words (мешок слов)

Самый простой способ векторного представления текста. Название — метафора:
слова из текста как бы складываются в мешок и перемешиваются — **порядок
слов совсем не учитывается**.

Чтобы векторизовать набор документов (текстов) мешком слов, нужно:
1. составить словарь всех уникальных слов, встречающихся во всех документах;
2. зафиксировать порядок слов в словаре и присвоить каждому слову
   порядковый индекс;
3. для каждого документа составить вектор размерности, равной размеру
   словаря, где на позиции каждого слова стоит число вхождений этого слова
   в данный документ (или просто 1/0, если считать только присутствие).

Итоговый вектор документа — это, по сути, счётчик того, сколько раз каждое
слово словаря встретилось в этом документе.

```python
from sklearn.feature_extraction.text import CountVectorizer

bow_vectorizer = CountVectorizer(min_df=1)
bow_vectorizer.fit(data_train['text'])

bow_vector = bow_vectorizer.transform(data_train['text'][:1])
bow_vector.shape
```

Словарь при работе с реальными текстами получается огромным, а большинство
слов встречается очень редко или, наоборот, слишком часто (предлоги,
союзы) и почти не несёт полезной информации. Поэтому размер словаря обычно
сокращают, отбрасывая слишком редкие и слишком частые слова:

```python
bow_vectorizer = CountVectorizer(min_df=4, max_df=0.95)
bow_vectorizer.fit(data_train['text'])
```

`min_df` — минимальное число документов, в которых должно встретиться
слово, чтобы попасть в словарь; `max_df` — максимальная доля документов
(здесь 95%), в которых слово может встречаться, иначе оно считается
слишком общеупотребимым (шумом).

Обучение классификатора на BoW-векторах — обычный sklearn-пайплайн:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_eval_model(train_X, test_X, train_y, test_y):
    model = LogisticRegression(max_iter=500)
    model.fit(train_X, train_y)

    train_pred = model.predict(train_X)
    test_pred = model.predict(test_X)

    train_acc = accuracy_score(train_y, train_pred)
    test_acc = accuracy_score(test_y, test_pred)

    print('Train accuracy:', round(train_acc, 3))
    print('Test accuracy: ', round(test_acc, 3))

bow_train = bow_vectorizer.transform(data_train['text'])
bow_test = bow_vectorizer.transform(data_test['text'])

train_eval_model(bow_train, bow_test, labels_train, labels_test)
```

## TF-IDF

Главный недостаток BoW: слово, которое встречается очень часто во всех без
исключения документах («и», «в», «на»), получает такой же вес, как и
слово, характерное именно для этого документа и потому более
информативное. TF-IDF (Term Frequency — Inverse Document Frequency)
исправляет это, взвешивая слова по двум составляющим:

- **TF (term frequency)** — частота слова внутри конкретного документа:
  число вхождений слова в документ, делённое на общее число слов в этом
  документе. Чем чаще слово встречается в документе — тем выше TF.
- **IDF (inverse document frequency)** — обратная частота слова по всей
  коллекции документов: логарифм от отношения общего числа документов к
  числу документов, где это слово встречается хотя бы раз. Чем реже слово
  встречается в коллекции в целом (то есть чем более оно «уникально» для
  отдельных документов) — тем выше IDF; слова, которые есть почти в каждом
  документе, получают IDF, близкий к нулю.
- **TF-IDF** — произведение TF и IDF. Итоговый вес слова в документе высок
  тогда и только тогда, когда слово часто встречается именно в этом
  документе, но редко — в остальных документах коллекции.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(min_df=4, max_df=0.95)
tfidf_vectorizer.fit(data_train['text'])

tfidf_train = tfidf_vectorizer.transform(data_train['text'])
tfidf_test = tfidf_vectorizer.transform(data_test['text'])

train_eval_model(tfidf_train, tfidf_test, labels_train, labels_test)
```

На практике TF-IDF почти всегда даёт качество не хуже, а часто и лучше
простого BoW, за счёт того, что automatически принижает вес неинформативных
частых слов.

## Базовая предобработка текста

Прежде чем векторизовать текст, его обычно предобрабатывают — это заметно
влияет на итоговое качество модели.

### Токенизация

**Токенизация** — деление текста на токены (в простейшем случае — на
отдельные слова). Может быть реализована через регулярные выражения или
без них:

```python
import re
from string import punctuation

# два способа сделать одно и то же

def tokenize(text):
    for p in punctuation:
        text = text.replace(p, ' ')

    text = text.strip().split()
    return text

def tokenize(text):
    reg = re.compile(r'\w+')
    return reg.findall(text)
```

Библиотека `nltk` — большая библиотека для работы с текстом (по значимости
для NLP сравнима с `numpy` для матриц); в ней уже реализованы токенизация,
лемматизация, стемминг и многое другое, например:

```python
from nltk.tokenize import wordpunct_tokenize

print(wordpunct_tokenize(data_train['text'][50]))
```

Текст перед токенизацией обычно приводят к нижнему регистру, чтобы «Dog» и
«dog» считались одним и тем же словом:

```python
data_tok_train = [tokenize(t.lower()) for t in data_train['text']]
data_tok_test = [tokenize(t.lower()) for t in data_test['text']]
```

### Удаление стоп-слов

**Стоп-слова** — очень частые служебные слова (предлоги, союзы, местоимения
и т.п.), которые почти не несут смысловой нагрузки для задач вроде
классификации текста и обычно убираются перед векторизацией:

```python
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

stop_words = stopwords.words('english')
stop_words += ['ap', 'us', 'u', '39']  # можно дополнить своими служебными токенами

def remove_stopwords(tokenized_texts):
    clear_texts = []
    for words in tokenized_texts:
        clear_texts.append([word for word in words if word not in stop_words])
    return clear_texts

data_tok_train = remove_stopwords(data_tok_train)
data_tok_test = remove_stopwords(data_tok_test)
```

### Стемминг

**Стемминг** — нахождение основы (стема) слова: в результате однокоренные
слова преобразуются к одинаковому виду (пример на русском: «вагон»,
«вагона», «вагоне», «вагонов», «вагоном», «вагоны» → все сводятся к
«вагон»). Для английского языка используют `nltk.stem.PorterStemmer` или
`nltk.stem.SnowballStemmer`, для русского — `nltk.stem.SnowballStemmer(language="russian")`.

```python
from nltk.stem import PorterStemmer

def stem_text(tokenized_texts):
    stemmed_data = []
    stemmer = PorterStemmer()

    for words in tqdm(tokenized_texts):
        stemmed_words = [stemmer.stem(word) for word in words]
        stemmed_data.append(stemmed_words)

    return stemmed_data

stemmed_train = stem_text(data_tok_train)
stemmed_test = stem_text(data_tok_test)
```

Стемминг работает довольно грубо (просто отбрасывает окончания по
эвристическим правилам) и иногда даёт нелингвистичные «обрубки» слов —
поэтому у него есть более аккуратная альтернатива.

### Лемматизация

**Лемматизация** — приведение слова к его нормальной форме (лемме):
для существительных — именительный падеж, единственное число; для
прилагательных — именительный падеж, единственное число, мужской род; для
глаголов, причастий, деепричастий — инфинитив. В отличие от стемминга,
лемматизация опирается на словарь и грамматику, а не просто отсекает
окончания, поэтому даёт более «настоящие» слова. Для английского —
`nltk.stem.WordNetLemmatizer`, для русского — `pymorphy2.MorphAnalyzer`.

```python
from nltk.stem import WordNetLemmatizer

def lemmatize_text(tokenized_texts):
    lemmatized_data = []
    lemmatizer = WordNetLemmatizer()

    for _, words in enumerate(tqdm(tokenized_texts)):
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        lemmatized_data.append(lemmatized_words)

    return lemmatized_data
```

После стемминга/лемматизации текст снова векторизуют (например, TF-IDF),
но уже поверх обработанных токенов — для этого векторизатору передают
готовый список токенов напрямую, отключая его собственную токенизацию:

```python
tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x, token_pattern=None, lowercase=False, min_df=4)
tfidf_vectorizer.fit(lemmatized_train)

tfidf_lemmatized_train = tfidf_vectorizer.transform(lemmatized_train)
tfidf_lemmatized_test = tfidf_vectorizer.transform(lemmatized_test)

train_eval_model(tfidf_lemmatized_train, tfidf_lemmatized_test, labels_train, labels_test)
```

## Векторные представления (эмбеддинги) текста: визуализация и поиск похожих

TF-IDF-векторы — это, по сути, простейший вид векторного представления
(эмбеддинга) текста: разреженный вектор размерности словаря, где каждая
координата соответствует конкретному слову. Такие векторы можно
использовать не только для классификации, но и для визуализации и поиска
похожих текстов.

**Визуализация.** Размерность TF-IDF-вектора равна размеру словаря (часто
десятки тысяч), нарисовать это напрямую нельзя — размерность нужно понизить
до 2-3. Один из способов — метод t-SNE (подробнее об этом методе — в теме
про кластеризацию и визуализацию многомерных данных):

```python
from sklearn.manifold import TSNE

tsne_vectors = TSNE(
    n_components=2, learning_rate='auto', init='random', perplexity=30
).fit_transform(tfidf_train[-2000:])

tsne_vectors.shape
```

```python
colors = labels_train[-2000:]
classes = ['World', 'Sports', 'Business', 'Sci/Tech']

scatter = plt.scatter(tsne_vectors[:, 0], tsne_vectors[:, 1], c=colors)
plt.legend(scatter.legend_elements()[0], classes, title="Classes")
```

**Поиск похожих текстов.** Если у каждого текста есть векторное
представление, «похожесть» двух текстов можно измерить как косинусную
близость их векторов (косинус угла между векторами: 1 — направления
совпадают, 0 — векторы перпендикулярны/независимы). Это стандартный способ
информационного поиска по эмбеддингам:

```python
texts = data_train['text'][:1000]
text_embeddings = tfidf_train[:1000].toarray()

def cosine(embedding, other_embeddings):
    normed_embeddings = other_embeddings.T / np.linalg.norm(other_embeddings, axis=1)
    return embedding @ normed_embeddings

def find_nearest(query, k=10):
    embedding = tfidf_vectorizer.transform([query])
    similarities = cosine(embedding, text_embeddings)[0]
    nearest_idxs = np.argsort(similarities)[-k:][::-1]

    return list(texts[nearest_idxs])

nearest = find_nearest(query="Bill Gates is the head of Microsoft", k=5)
for text in nearest:
    print(text)
    print()
```

## Тематическое моделирование (бонус)

Смежная задача — не классификация с учителем, а поиск скрытых тем в
коллекции текстов без разметки. **Тематическая модель** (topic model) —
модель коллекции документов, которая для каждого документа определяет, к
каким темам он относится: на выходе для документа получается числовой
вектор из оценок степени принадлежности каждой теме (число тем задаётся
заранее или подбирается моделью).

Два основных семейства методов: **алгебраические** (например, латентно-
семантический анализ, LSA) и **вероятностные** (самый популярный —
латентное размещение Дирихле, LDA). Поскольку у LSA и LDA принципиально
разная математическая природа, на разных данных они могут давать разное по
качеству разбиение на темы, хотя практический интерфейс использования у
них похож.

```python
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary

id2word = Dictionary(lemmatized_train)
corpus = [id2word.doc2bow(text) for text in lemmatized_train]

# Обучаем модель LDA
lda_model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    num_topics=8
)

# визуализируем наиболее популярные слова в каждой теме
topics = lda_model.show_topics(formatted=False)

topics = {
    f'topic {i}': [pair[0] for pair in topic[1]] for i, topic in enumerate(topics)
    }

pd.DataFrame(topics)
```
