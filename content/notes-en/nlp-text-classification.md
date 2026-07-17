# Text processing and text classification (NLP)

These notes were assembled entirely from a single source — the seminar "Text
preprocessing and text classification (NLP)" (stream 2, May 27). There is no
separate lecture transcript on this topic, so all the material comes from the
seminar.

## Why it is worth being able to work with text at all

Text data is one of the most common types of data: you have to work with it
in almost any subject area, even if text is not the main focus of the task.
It is no accident that large language models are called language models —
being able to work with text is useful regardless of whether it is classical
ML or working with an LLM.

## Tokenization

Any work with text begins with tokenization — splitting the text into pieces
(tokens). This is either a built-in stage of the pipeline or a manual
operation, but in one form or another it is always present.

The most naive way is to split the text on whitespace (the `split` function in
Python). This is not recommended: there are ready-made tokenizers, and there
are quite a lot of them (for example, in the NLTK library, which is good for
English but can also work with Russian).

Why are there so many tokenizers if the task seems simple? Each tokenizer is
usually implemented as a monolithic module with a hard-coded set of
heuristics, without flexible customization — instead of one universal
"mega-tokenizer," a separate implementation is written for each specific task.
This historically established field was called computational linguistics and
was very active roughly until the mid-2010s, until many of the tasks it solved
were "eaten up" by large language models.

The choice of tokenizer affects the final quality of the model. A simple
engineering approach — a primitive tokenizer (splitting on whitespace and
getting the start/end indices of each token via a regular expression) — is a
perfectly workable option if you do not want to overcomplicate things.

Specific tokenizers:
- **TreebankWordTokenizer** — splits, for example, `don't` into `do` and
  `n't`, breaking the negation particle out as a separate token.
- Tokenizers for mathematical expressions — can separately isolate variables
  (letters) and separately the symbols of the expression with brackets.
- **TweetTokenizer** — does a good job of extracting emoticons and hashtags
  from messages.

Conclusion: if the text is specific (technical, with formulas, with special
markup), it is worth choosing the tokenizer to fit the task — based on what it
should preserve and what it should extract.

## Stop words

Stop words (junk words, noise) — usually prepositions, conjunctions, articles
and similar constructions that are considered to carry no semantic load and
are usually removed from the text. In NLTK the stop words for different
languages (including Russian) are stored as a ready-made list
(`corpus.stopwords`); there is also a separate constant with punctuation
marks.

Removing stop words must be done carefully — the effect depends on the task.
For example, when classifying emotions the word "или" ("or") is by default
considered a stop word, but it can be a marker of the emotion of
doubt/uncertainty — by removing it along with the stop words, you can lose a
useful signal.

**A practical rule.** If a word or a set phrase regularly occurs in all the
documents of the corpus and does not help distinguish documents from each
other (for example, all the texts are about one and the same region anyway),
you can safely add it to the stop words — this will remove noise. But you need
to be careful: if in some of the documents the word does carry a
distinguishing meaning (for example, a comparison with another region),
removing it may cost you a loss of meaning. The general rule: the stop-word
list should be supplemented according to the meaning of the specific corpus,
and its effect on model quality must always be checked, rather than treating
it as a universally safe operation.

## Lemmatization and stemming

### Lemmatization

Bringing a word to its normal form according to the rules of the language (for
example, to the nominative singular for nouns or to the infinitive for verbs)
— a soft transformation. The point is to reduce the size of the vocabulary
when encoding text: the tokens "пью", "пил", "пьёт" ("I drink," "drank,"
"drinks") are reduced to a single lemma "пить" ("to drink"), assuming (though
this is a debatable claim) that the meaning is preserved. The effect of
lemmatization on quality should be checked, rather than assuming it is
unconditionally useful.

Lemmatization libraries for the Russian language:
- **Mystem** (Yandex) — takes the whole text as input, so it can take context
  into account and resolve homonymy (disambiguation) — that is, determine the
  exact meaning of a word when it has several meanings (a classic example:
  "лук" — a plant or a weapon; "ключ" — a door key, a spring, or a wrench).
- **Pymorphy (Pymorphy2/3)** — also works well with Russian, but, unlike
  Mystem, takes individual words as input rather than the whole text, so it
  does not take context into account and does not resolve homonymy. On the
  other hand, it gives a more detailed morphological analysis: part of speech,
  aspect (perfective/imperfective), mood, tense, etc.

Important: inside these libraries there is no machine learning and no neural
networks — they are heuristic functions, hand-written by linguists, that
formalize the rules of the language.

The practical choice: if it is important that the word retain its meaning —
use lemmatization (Mystem, Pymorphy). If you need to compress the text as much
as possible (reduce the vocabulary), even at the cost of some of the meaning —
use stemming.

### Stemming

A harsher transformation: it does not obey the grammatical rules of the
language as strictly, but simply cuts off the endings and suffixes (affixes),
bringing the word to some "stem." Different stemmers are built on different
heuristics specific to a particular language. Well-known algorithms: the
**Porter Stemmer** and its newer version — the **Snowball Stemmer** (works
well with Russian too, it is in NLTK).

In summary: a lemma is a form that obeys the rules of the language; stemming is
a cruder, not always linguistically correct procedure. Stemming sometimes cuts
too harshly and loses meaning, but on the other hand it compresses the
vocabulary better.

## Bag of words and TF-IDF

Two main static encoders of text into a vector: bag of words
(`CountVectorizer`) and TF-IDF (`TfidfVectorizer`).

**Bag of words** — frequency: how many times each unique word of the corpus
occurred in each document. We take the whole corpus, write out all the unique
words as features, the rows are documents, and the cells hold the number of
occurrences of a word in a document. The result is a sparse matrix (most cells
are zeros).

The drawback of bag of words is that it does not preserve any semantic or
thematic relationships between words, it just counts frequencies. The plus is
its simplicity and speed: in ML, simple solutions in practice often turn out
to be the most applicable.

Hyperparameters of `CountVectorizer`:
- `max_df`, `min_df` — cut off words that are too frequent (`max_df`,
  essentially a continuation of the fight against stop words) and too rare
  (`min_df`), carrying no information.
- `max_features` — limits the size of the vocabulary to a maximum number of
  features.
- you can specify your own tokenizer and stop-word list right inside the
  vectorizer.
- `ngram_range` — count not only individual words (unigrams), but also
  phrases: bigrams, trigrams, etc. — takes the adjacency of words into
  account.

Practical conclusion: sweeping the vectorizer's hyperparameters (`min_df`,
`max_df`, `max_features`, `ngram_range`) via grid search is a workable and
useful practice even for a task as seemingly simple as encoding text.

A reminder about n-grams: an n-gram is a sequence of n **consecutive** units
(letters or words): unigram (1), bigram (2), trigram (3), 4-gram, etc. —
precisely consecutive elements, not ones randomly chosen from different places
in the text.

**TF-IDF** — the product of two factors: TF (term frequency, a normalized
version of the word frequency from bag of words) and IDF (inverse document
frequency). IDF increases the weight of a word if it occurs frequently in a
particular document/topic but does not occur frequently across the whole
corpus, and lowers the weight of a word that occurs equally often across all
documents alike (carrying no distinguishing information). The point is to
reweight the words so as to highlight those that characterize a particular
category/document, rather than occurring equally often everywhere. The IDF
formula uses a logarithm — it smooths out the scale of the weight (TF has no
logarithm, IDF does). The resulting TF-IDF values are additionally normalized
to the interval [0, 1].

The hyperparameters of `TfidfVectorizer` are broadly the same as those of
`CountVectorizer` (`max_df`, `min_df`, `max_features`, `ngram_range`); an
additional one is `strip_accents` (removing diacritical marks like é in
French).

## Task 1: classifying emotions in text

Paul Ekman's classic psychological classification distinguishes 6 basic
emotions expressed through facial expressions across all cultures: happiness,
sadness, fear, anger, disgust and surprise.

In the dataset used (inspired by one of the works on feature engineering for
Russian-language texts), Ekman's classification is reduced to 5 classes:
"disgust" was removed (hard to identify from text), and "fear" and "surprise"
were combined into the category "uncertainty" (because of their semantic
closeness).

**Data:** a corpus labeled with 5 emotion labels, 50,000 objects, the texts
already lemmatized by a special neural-network-based morphological analyzer (as
opposed to the rule-based Mystem/Pymorphy).

**Checking class balance:** the difference between the most frequent and the
rarest class is on the order of 3 percentage points (21% vs 18%) — within the
norm, the data can be considered balanced (the boundary of "balancedness" is
always conventional and depends on the task).

**The solution pipeline:**
1. Train/test split with stratification by class (with 5 classes — mandatory).
2. `TfidfVectorizer` with the given `min_df`/`max_df`.
3. The model — logistic regression (surprisingly good at text classification).
   It is important to fix `max_iter` (the model may not converge on the default
   number of iterations) and `random_state`.

**Result:** the confusion matrix shows that the "neutrality" class is
predicted best, and "joy" and "sadness" worst.

**A practical technique.** If in multiclass classification one class is
consistently predicted worse than the others, it makes sense not to try to
cover all the classes with a single model, but to break out a separate binary
classifier specifically for the problem class (on a "this class or not this
class" scheme), and to handle the remaining classes with a separate model that
has fewer classes.

**Interpreting the model.** Logistic regression is a linear model, it has
coefficients, one per feature (a vocabulary token). The magnitude and sign of a
coefficient show the feature's contribution: the larger the positive value —
the more strongly the token raises the probability of the corresponding class,
the more negative — the more strongly it lowers it, and values near zero have
almost no effect. The top 5 features for the "joy" class are words like
"спасибо" ("thank you"), "круто" ("cool"), "крутой" ("cool"), "вау" ("wow").
For the "uncertainty" class, one of the top features turned out to be the word
"мочь" ("to be able").

## Task 2: classifying the sentiment of tweets

Binary classification of the sentiment of short posts (tweets/posts):
positive/negative. The dataset is about 226 thousand tweets. The texts are
short because they were historically constrained by the social network's limit.

**The experiment — logistic regression on top of different vectorizations:**
- Baseline: `CountVectorizer` on unigrams — accuracy ≈ 0.87.
- Switching to trigrams — the quality dropped noticeably.
- `TfidfVectorizer` with `ngram_range` from 1 to 2 — accuracy ≈ 0.78 (the
  increase was only relative to the unsuccessful trigram variant, not relative
  to the original baseline of 0.87).

**The key observation.** By default the vectorizers (both `CountVectorizer`
and `TfidfVectorizer`) use a tokenizer that throws out punctuation marks. If
you explicitly specify a tokenizer that preserves punctuation (for example,
`nltk.WordPunctTokenizer`), the quality jumps sharply — up to 0.96. For
Twitter text, punctuation turned out to be a very important signal.

The feature with the largest coefficient in absolute value after including
punctuation is the closing-bracket symbol ")": the model essentially "learned"
the emoticon ")" as a strong marker of a positive tweet. Hence an experiment:
you can dispense with ML entirely and write a one-line rule: if the text
contains a bracket ")" — the tweet is positive, if not — negative. Such a
simple heuristic gives an accuracy of about 0.90-0.91 — almost without any
model at all. Conclusion: the sentiment classification task in many cases is
successfully solved not only by ML models, but also by simple heuristics
(regular expressions, searching for special characters, emoticons, marker
words).

**Character n-grams.** Switching to `ngram_range` + `analyzer='char'`
(sequences of letters instead of words) raised the quality even more — up to
0.98, again thanks to the fact that character n-grams "catch" those same
brackets and other special characters. This approach is also useful in
detecting the language of a text (by characteristic letter combinations) and
in working with rare/low-resource languages, for which there are no quality
ready-made libraries.

**Practical applications of sentiment classification:**
- Monitoring brand mentions on social media. Ready-made commercial monitoring
  platforms often handle this poorly: on the order of 90% of
  reactions/comments are mistakenly classified as neutral.
- Classifying reviews on marketplaces and responding to negative reviews.
- Processing support tickets — by the emotional coloring of a ticket you can,
  at the first step of the pipeline, classify the tone of the message and route
  the request through different handling scenarios (for example, to different
  prompts of a support LLM agent).

**Multi-stage classification.** If the style of communication (punctuation,
emoji, individual symbols) differs greatly between groups of users (by age,
professional field, region), it may be more effective to first classify the
user/text into such a group, and then apply a separate, more specialized
sentiment classification model for each group — this can give better quality
than a single common model for all the groups at once. An important practical
nuance: a model trained on the texts of one social/age group with a certain
punctuation style may perform poorly on the texts of another group — this is
worth taking into account when collecting training data.

## The spaCy library

An industrial library for working with text — **spaCy**: it lets you build a
single reusable pipeline out of all the text processing stages considered
(tokenization, removing stop words, lemmatization/stemming, morphological
analysis), rather than assembling them by hand from separate tools. It
supports many languages at once. It is worth paying attention to when working
with a large corpus of texts in practice (for example, in a term paper or a
graduation thesis on NLP).

## Optional task

Separately in the class an optional example was outlined for independent study
— the task of classifying SMS spam (in essence analogous to the sentiment
classification task), inspired by one of the competitions.
