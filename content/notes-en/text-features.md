# Working with text data and embeddings (basic level)

These notes were assembled entirely from two course seminars ("Embeddings for text" and
"Transforming text data") — Evgeny Patochenko's repository has no
separate material on this topic, so there is no "repository as the basis"
layer here: everything is taken from the local notebooks.

## Why turn text into numbers

Machine learning algorithms (linear regression, logistic regression and
almost everything else) can work only with numbers — more precisely, with vectors
of numbers. Text in its original form (a string of characters) is not suitable for them, so
the first step of any work with text is **vectorization**: turning text into a
numeric vector of fixed length.

## Bag of Words

The simplest way of vectorially representing text. The name is a metaphor:
the words from the text are, as it were, thrown into a bag and shuffled — **the order
of words is not taken into account at all**.

To vectorize a set of documents (texts) with bag of words, you need to:
1. compile a dictionary of all unique words occurring across all documents;
2. fix the order of the words in the dictionary and assign each word
   a sequential index;
3. for each document, compile a vector of dimension equal to the size of the
   dictionary, where at the position of each word stands the number of occurrences of that word
   in the given document (or simply 1/0, if you count only presence).

The resulting document vector is, in essence, a counter of how many times each
word of the dictionary occurred in this document.

```python
from sklearn.feature_extraction.text import CountVectorizer

bow_vectorizer = CountVectorizer(min_df=1)
bow_vectorizer.fit(data_train['text'])

bow_vector = bow_vectorizer.transform(data_train['text'][:1])
bow_vector.shape
```

With real texts the dictionary turns out to be huge, and most
words occur very rarely or, conversely, too often (prepositions,
conjunctions) and carry almost no useful information. Therefore the size of the dictionary is usually
reduced by discarding words that are too rare and too frequent:

```python
bow_vectorizer = CountVectorizer(min_df=4, max_df=0.95)
bow_vectorizer.fit(data_train['text'])
```

`min_df` is the minimum number of documents in which a word must occur
to make it into the dictionary; `max_df` is the maximum fraction of documents
(here 95%) in which a word may occur, otherwise it is considered
too common (noise).

Training a classifier on BoW vectors is an ordinary sklearn pipeline:

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

The main drawback of BoW: a word that occurs very often in all documents
without exception ("and", "in", "on") gets the same weight as a
word specific precisely to this document and therefore more
informative. TF-IDF (Term Frequency — Inverse Document Frequency)
fixes this by weighting words by two components:

- **TF (term frequency)** — the frequency of a word within a specific document:
  the number of occurrences of the word in the document, divided by the total number of words in that
  document. The more often a word occurs in a document, the higher its TF.
- **IDF (inverse document frequency)** — the inverse frequency of a word across the whole
  collection of documents: the logarithm of the ratio of the total number of documents to
  the number of documents where this word occurs at least once. The rarer a word
  occurs in the collection as a whole (that is, the more "unique" it is to
  individual documents), the higher its IDF; words that are present in almost every
  document get an IDF close to zero.
- **TF-IDF** — the product of TF and IDF. The resulting weight of a word in a document is high
  if and only if the word occurs often precisely in this
  document, but rarely in the rest of the documents of the collection.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(min_df=4, max_df=0.95)
tfidf_vectorizer.fit(data_train['text'])

tfidf_train = tfidf_vectorizer.transform(data_train['text'])
tfidf_test = tfidf_vectorizer.transform(data_test['text'])

train_eval_model(tfidf_train, tfidf_test, labels_train, labels_test)
```

In practice TF-IDF almost always gives quality no worse than, and often better than,
plain BoW, thanks to the fact that it automatically downweights uninformative
frequent words.

## Basic text preprocessing

Before vectorizing text, it is usually preprocessed — this noticeably
affects the final quality of the model.

### Tokenization

**Tokenization** is dividing text into tokens (in the simplest case — into
individual words). It can be implemented via regular expressions or
without them:

```python
import re
from string import punctuation

# two ways to do the same thing

def tokenize(text):
    for p in punctuation:
        text = text.replace(p, ' ')

    text = text.strip().split()
    return text

def tokenize(text):
    reg = re.compile(r'\w+')
    return reg.findall(text)
```

The `nltk` library is a large library for working with text (in importance
for NLP comparable to `numpy` for matrices); it already implements tokenization,
lemmatization, stemming and much more, for example:

```python
from nltk.tokenize import wordpunct_tokenize

print(wordpunct_tokenize(data_train['text'][50]))
```

Before tokenization, text is usually lowercased so that "Dog" and
"dog" are treated as the same word:

```python
data_tok_train = [tokenize(t.lower()) for t in data_train['text']]
data_tok_test = [tokenize(t.lower()) for t in data_test['text']]
```

### Removing stop words

**Stop words** are very frequent function words (prepositions, conjunctions, pronouns,
etc.), which carry almost no semantic load for tasks like
text classification and are usually removed before vectorization:

```python
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

stop_words = stopwords.words('english')
stop_words += ['ap', 'us', 'u', '39']  # you can add your own function tokens

def remove_stopwords(tokenized_texts):
    clear_texts = []
    for words in tokenized_texts:
        clear_texts.append([word for word in words if word not in stop_words])
    return clear_texts

data_tok_train = remove_stopwords(data_tok_train)
data_tok_test = remove_stopwords(data_tok_test)
```

### Stemming

**Stemming** is finding the stem of a word: as a result, words with the same root
are transformed to an identical form (an example in Russian: "вагон",
"вагона", "вагоне", "вагонов", "вагоном", "вагоны" → all reduce to
"вагон"). For English, `nltk.stem.PorterStemmer` or
`nltk.stem.SnowballStemmer` is used, for Russian — `nltk.stem.SnowballStemmer(language="russian")`.

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

Stemming works rather crudely (it just drops endings by
heuristic rules) and sometimes produces non-linguistic word "stumps" —
so it has a more accurate alternative.

### Lemmatization

**Lemmatization** is reducing a word to its normal form (the lemma):
for nouns — the nominative case, singular; for
adjectives — the nominative case, singular, masculine; for
verbs, participles, gerunds — the infinitive. Unlike stemming,
lemmatization relies on a dictionary and grammar rather than simply cutting off
endings, so it produces more "real" words. For English —
`nltk.stem.WordNetLemmatizer`, for Russian — `pymorphy2.MorphAnalyzer`.

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

After stemming/lemmatization the text is vectorized again (for example, with TF-IDF),
but now on top of the processed tokens — for this the vectorizer is passed the
ready list of tokens directly, disabling its own tokenization:

```python
tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x, token_pattern=None, lowercase=False, min_df=4)
tfidf_vectorizer.fit(lemmatized_train)

tfidf_lemmatized_train = tfidf_vectorizer.transform(lemmatized_train)
tfidf_lemmatized_test = tfidf_vectorizer.transform(lemmatized_test)

train_eval_model(tfidf_lemmatized_train, tfidf_lemmatized_test, labels_train, labels_test)
```

## Vector representations (embeddings) of text: visualization and similar-text search

TF-IDF vectors are, in essence, the simplest kind of vector representation
(embedding) of text: a sparse vector of dictionary dimension, where each
coordinate corresponds to a specific word. Such vectors can be
used not only for classification, but also for visualization and searching for
similar texts.

**Visualization.** The dimension of a TF-IDF vector equals the size of the dictionary (often
tens of thousands), you cannot draw this directly — the dimension needs to be reduced
to 2-3. One way is the t-SNE method (more on this method — in the topic
about clustering and visualization of high-dimensional data):

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

**Searching for similar texts.** If each text has a vector
representation, the "similarity" of two texts can be measured as the cosine
similarity of their vectors (the cosine of the angle between the vectors: 1 — the directions
coincide, 0 — the vectors are perpendicular/independent). This is a standard method of
information retrieval by embeddings:

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

## Topic modeling (bonus)

A related task — not supervised classification, but finding hidden topics in a
collection of texts without labels. A **topic model** is a
model of a collection of documents that, for each document, determines which
topics it belongs to: at the output, for a document you get a numeric
vector of scores of the degree of belonging to each topic (the number of topics is set
in advance or tuned by the model).

Two main families of methods: **algebraic** (for example, latent
semantic analysis, LSA) and **probabilistic** (the most popular —
latent Dirichlet allocation, LDA). Since LSA and LDA have fundamentally
different mathematical natures, on different data they may produce a topic
partition of differing quality, although their practical usage interface is
similar.

```python
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary

id2word = Dictionary(lemmatized_train)
corpus = [id2word.doc2bow(text) for text in lemmatized_train]

# Train the LDA model
lda_model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    num_topics=8
)

# visualize the most popular words in each topic
topics = lda_model.show_topics(formatted=False)

topics = {
    f'topic {i}': [pair[0] for pair in topic[1]] for i, topic in enumerate(topics)
    }

pd.DataFrame(topics)
```
