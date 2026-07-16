# RAG (Retrieval-Augmented Generation)

These notes were assembled entirely from a single source — the recording of the lecture "13. Lecture
recording, April 25, 2026". This is part of a block on LLMs/agents/RAG that is unique to the course
and isn't in Evgeny Patochenko's repository — there is no separate `repo_lesson` for this topic, so the
lecture is the only source of knowledge.

## Recap: the stages of training large language models

1. **Pretraining** — the model is trained on a huge volume of data (roughly — the entire available
   text internet). At this stage the model learns the "target distribution of language", that is, it
   learns to speak: to plausibly continue text. It does not solve specific tasks but only predicts the
   most likely continuation — thanks to an enormous amount of computation and data, diverse knowledge
   about the world settles into its weights. If you give such a "raw" model a specific task (for
   example, an arithmetic one), it won't solve it but will just continue the text in a plausible but
   meaningless-for-solving way.
2. **SFT (Supervised Fine-Tuning)** — the model is further trained to generate text in accordance with
   specific tasks: hold a dialogue, summarize text, write code, and so on. Since almost any task can
   be described in text (including classical ML tasks), after SFT the model starts answering applied
   questions in a meaningful way.
3. **Alignment (RLHF)** — an even finer tuning of the model to user preferences, based on a reward
   model that rewards or punishes the model for one behavior or another. It's built on reinforcement
   learning.

## Why we need RAG

RAG (Retrieval-Augmented Generation) is a way to answer a user's query using an additional external
data source rather than relying solely on the model's internal knowledge. Technically it's a generative
answer over external sources: good search (classical or vector-based, over a vector database) + a good
generator.

Why not put all knowledge about the world directly into the model at the pretraining or SFT stage?
There is a large class of queries a model trained and frozen once cannot answer:

- **Personal, user-specific data.** For example, "how many vacation days do I have left this year" —
  the model knows nothing about the specific user or the current balance of their vacation days,
  because this information constantly changes and can't be "baked into" the model at the training
  stage.
- **Fresh, frequently updated data.** The weather, football match results, the current price of a
  barrel of oil, and the like.
- **Specific, accuracy-critical queries**, where a wrong answer (a hallucination) can have serious
  consequences — for example, "what is the treatment protocol for fungal pneumonia". Here you need a
  reliable answer strictly grounded in sources — preferably with links to them and the ability to
  verify.

**The problem of hallucinations.** Hallucinations in language models are a systemic feature: their
nature is stochastic, and hallucinations cannot be entirely ruled out. RAG by itself does not
completely eliminate the probability of a hallucination; it only reduces it compared to the model's
answers "from memory". Additionally, the Alignment/RLHF stage strongly influences the reduction of
hallucinations.

## How to quickly assemble a basic version of RAG

RAG consists of two basic components: **search** and the **generator**. The workflow:

1. There is a user query and an external knowledge base (external — relative to the LLM's own
   knowledge; for a specific company this base may be internal, but "external" from the model's point
   of view).
2. A base of documents is gathered, the documents are cut into pieces (chunks) — by number of
   characters, tokens, paragraphs, sentences, or sections. The cutting is done with overlap between
   neighboring chunks so as not to lose information at the boundaries.
3. The chunks are put into a vector database. It's important to plan up front for regular updating of
   this base (for example, via periodic parsing of the data source or an API), because the knowledge
   base changes over time.
4. On a user query, the search component returns the top-k most relevant (semantically close)
   documents/chunks.
5. The generator (an LLM, for example, via the API of one of the open-source models) receives a
   prompt into which the query and the found documents are inserted, and generates an answer.

Such a basic solution can be assembled very quickly — "over a cup of coffee", in the course of a single
evening, with some skill — using ready-made frameworks: LangChain — for working with texts,
sentence-transformers — for ready-made pretrained embedding models, and any available LLM (open or via
an API) — as the generator.

## A methodology for iteratively improving RAG

A basic solution (the RAG baseline) assembled "over an evening" usually immediately spawns a whole heap
of problems: hallucinations, no answer where there should be one, wrong or incomplete answers, and so
on. Just tweaking framework parameters at random is a poor strategy: some problems go away, others pop
up. You need a systematic approach analogous to classical ML — first a basic version is built, then
iteratively, step by step, problems are diagnosed and fixed:

1. **Honest labeling of existing answers.** An honest slice of the real query stream is taken (or of
   what you expect to receive in production). The slice is labeled — preferably by several
   participants (developers, QA, product managers, marketers, analysts, managers — anyone who
   understands the product is suitable), to get an objective picture.
2. **Localizing problems and counting the percentage of "good" answers.** From the labeling you get
   the percentage of answers that are satisfactory and the percentage that needs work.
3. **Categorizing and prioritizing problems.** The remaining "bad" answers are broken down by problem
   type in some percentage ratio, after which the problems are prioritized — by degree of impact on
   quality. You start solving the most widespread problems, and return to the rare ones later.
4. **Iterative fixing.** A problem is solved — quality is measured again — move on to the next problem.
   And so around the loop.

Almost all problems, with rare exceptions, are unrelated to each other: solving one has practically no
effect on other parts of the system, so they can be fixed separately. As a rule, the analysis reveals
that some problems fall on the search component and some on the generator component (for example,
roughly 60% on search, 40% on the generator).

**Typical causes of problems:**
- **Faithfulness** — the model writes things not present in the sources (hallucinates).
- **Completeness** — there is an answer, but it's incomplete, doesn't answer the question in full.
- **No answer, even though the knowledge is in the base** — the model failed to extract the needed
  information from the documents and generate a correct answer.
- **"Frozen" answers** — instead of answering, the model asks the user again or doesn't answer on the
  substance, even though all the necessary information is in the context.

## Problem: no relevant documents in the results (a search problem)

It's known that the needed document is in the base, but when searching by the user's query it doesn't
even make it into the top-30/40/50 of the results:

- If classical (text) search is used — the problem is most likely in the search mechanism itself.
- If vector search is used — the problem is most likely in the embedder. You need to check whether the
  embedder in use is trained specifically for the document-retrieval-by-query task (retrieval), and
  not just for "good embeddings in general" — a model not tailored to the specific task often works
  poorly precisely on retrieval.
- The solution is to pick a different, more suitable embedding model, or to further train the existing
  one on your own knowledge base / search results.
- A frequent and poorly understood problem: locally (on a small set) search works well, but on the
  whole base — poorly. **Hybrid search** can help here — a combination of text search (for candidates)
  and vector search.

## Problem: relevant documents exist, but not at the top of the results (reranking)

Documents containing the needed information make it into the results, but not to the top — they're
"smeared" somewhere across the list. Since the model's context window is limited (and this limitation
is genuinely painful even for large companies with big budgets), you can't just stuff the entire result
set into the context — you need the relevant documents to end up at the top and the irrelevant ones at
the bottom.

For this the **reranking** (re-ranking) technique is used with an additional model — the **reranker**.
The reranker takes a (query, document) pair as input and predicts a score of their match. Taking, for
example, the top-10 documents from the initial results, you can compute a score for each query-document
pair and re-sort the documents in descending order of score. As a reranker you can use ready-made
models (for example, from the BERT/cross-encoder family) — there are quite a few of them, and if
needed they too can be further trained for your task.

An additional technique: if after reranking some of the chunks have a score below some threshold, such
chunks can simply be discarded — they're weakly relevant and only waste space in the context. This,
like many similar tricks, "may work or may not" — you need to check on your own data.

**BERT vs GPT (for reference).** BERT is not a single model but a whole family (RoBERTa, ALBERT, and
others, on the order of fifty variants), which was very popular in 2018–2019. Now the BERT family is
used much more selectively — for narrow tasks (classification, detection), because large language
models solve almost all such tasks better, just with far greater computational cost. So where you need
to close out a small, narrow task, "firing" a big LLM at it is not cost-effective — this is exactly
where a small specialized model like a BERT classifier can come in handy, and it can also be deployed
locally.

## Problems with document quality in the knowledge base

- **Low quality of individual documents** — for example, due to recognition errors (OCR of images) or
  poor page parsing, or documents that were "garbage" to begin with. The solution is to train or use
  ready-made "garbageness" classifier models (FastText, BERT classifiers), or heuristics: the fraction
  of non-alphabetic/special characters, the fraction of function words with no semantic load, repeated
  characters, and so on.
- **Duplicates and near-duplicates of documents.** If several nearly identical documents make it into
  the relevant results, they take up the whole top and impoverish the answer. You can combat this by
  bringing in a quality embedding model capable of extracting semantics and finding semantic
  duplicates, and keeping only unique documents in the base.
- **Thematically useless documents** — for example, half-empty template pages. They're detected by
  heuristics (structural templates), classifier models, or by running through an LLM with a
  verification prompt (more expensive computationally).
- **Generated ("synthetic") sources.** With classical data collection from the internet, a lot of
  LLM-generated content can make it into the base — it's worth getting rid of such content: the
  information in it may be unreliable, plus it carries reputational risks.
- **Custom document formats** (PDF, presentations, DOCX, etc.) — for quality data extraction it's
  better to use a separate, specially configured parser for each format.
- **Outdated documents.** Knowledge bases become outdated over time, which is especially critical for
  domains like legislation. Ways to combat this: a rule for automatic removal of outdated documents (by
  TTL or upon the appearance of a new version of the source); using the document's creation/addition
  date as part of the context passed to the generator (so the model itself takes into account how old
  the document is). In certain cases (for example, legislation, where it's important to distinguish the
  regulatory framework of different periods) you may need to store several versions of the same
  document at once for different time intervals.

## Problems with chunking

Problems with chunks usually manifest as search problems: a piece of a document that is relevant to the
query's topic but cut off at important information makes it into the context — the model loses part of
the data needed for a complete answer.

The main strategies for forming chunks:

1. **Fixed-size chunking** — cutting by a fixed size (for example, by number of tokens) with mandatory
   overlap between neighboring chunks (a classic ratio — for example, chunk size 1000 tokens, overlap
   100). The optimal size is chosen empirically: it depends on speed, the model's context size, the
   number of documents, and the final answer quality. The trade-off: the larger the chunk, the more of
   the model's context it takes up and the slower everything works, but the more useful information
   fits into it.
2. **Enriching chunks with meta-information.** You can add a heading, the document's author, tags, and
   so on to a chunk (for example, for a medical assistant — data about the article's author as an
   indicator of reliability).
3. **Splitting by document structure.** If a domain has a clear structure (legislative acts broken into
   articles and clauses; internal wiki pages with sections), you can and should cut along this
   structure. Downsides: the structure doesn't always reflect meaning; you encounter overly large
   sections that still have to be split further, trying to preserve the general information (for
   example, the subsection title) in each resulting sub-chunk.
4. **Clustering based on embeddings.** The document is split into sentences, and each sentence is
   vectorized by a lightweight embedding model. Sentences are added to clusters sequentially: for each
   new sentence the cosine distance to the averaged embedding of the existing cluster is computed — if
   it's close, the sentence is added to the cluster; if not, a new one is created.
5. **Forming chunks at runtime ("on the fly").** Instead of cutting all documents into chunks in
   advance (offline), the relevant document is found in its entirety via search, and only after that
   chunks are cut from it at runtime and scored against the specific user query.
6. **Chunking with an LLM.** The document is run through an LLM with a request to summarize the content
   (without changing its essence), and the resulting summary is split into several parts — these parts
   are put into the base as chunks.

## Problems with unclear user queries

User queries are often phrased carelessly, with typos, in fragments. Ways to combat this:

- **Typo correction** — long-established models (an analog of autocorrect on smartphones).
- **Query rephrasing.** A poorly composed query is reformulated into a "good" one with a model, instead
  of asking the user again. You have to act delicately — an unfortunate rephrasing can distort the
  original meaning of the query.
- **Generating additional queries (multi-query).** Several similar variations are generated from the
  original query, a separate selection of relevant chunks is made for each, and the results are
  combined.
- **Keyword extraction** — additionally expands the search query.

## Summary of the RAG improvement methodology

After creating the basic version (the RAG baseline): we analyze the problem → if quality is poor (and
at first it's almost always poor somewhere) → we reduce the analysis to large clusters of problems that
are important to fix first. If there are no relevant documents in the results — we work on the
embedder/search. If documents don't make it to the top — we add a reranker. If there is garbage in the
base — we filter with heuristics and/or additional models and tune the chunking. If unclear queries
come in — we deal with rephrasing or with generating additional queries.

Most of the listed techniques don't require super-complex engineering solutions — they're relatively
small, accessible tricks that can quite well be implemented solo.

## Practice: building a RAG system on Kinopoisk reviews

The goal is to build a simple assistant that answers arbitrary questions about films in Russian, using
Kinopoisk reviews as the knowledge base.

**Data.** A dataset of reviews is downloaded from the Hugging Face Hub (an analog of GitHub for models
and datasets in Deep Learning) via the `datasets` library and the `load_dataset` function. After
preprocessing (replacing garbage characters, extra spaces via `map`) — 36,591 records. Key fields: the
film title and the review content (text + the user's rating).

**Baseline (no-RAG) LLM.** The generator is a small open model, Qwen, at 1.5 billion parameters, an
instruct version from GPT-like architectures, multilingual. It's loaded via `pipeline` from the
`transformers` library (a wrapper over PyTorch), specifying the tensor type (float16/float32) and the
device.

A test query without RAG: "In which 5 films did Robert De Niro act?". Generation parameters: 256
tokens, temperature and top_p (sampling parameters responsible for the model's "creativity"/diversity —
they narrow or widen the distribution of possible answers). The result — the model produced a list,
some of the films turned out to be entirely made up. The reason — a lack of Russian-language data about
cinema in the model's training set: the model still plausibly generates an answer even without the
needed knowledge, that is, it hallucinates.

**Building the vector knowledge base:**
1. An embedding model is chosen — a large multilingual transformer-based model, loaded via
   `sentence-transformers`.
2. A vector database is chosen — Qdrant. Via the Qdrant client a collection is created
   (`create_collection`) specifying the vector size (must match the dimensionality of the embedding
   model), the proximity metric (cosine distance), and a flag for saving to disk.
3. The documents (reviews) are cut into chunks, since they're too long to fit into the model's context
   in full. A recursive splitter from LangChain is used (`RecursiveCharacterTextSplitter`) — one of the
   most "harmless" methods, works more often than not.
4. The chunks are encoded by the embedding model (`encode`), specifying vector normalization and the
   device (GPU).
5. The resulting vectors, together with metadata (the review text, the film's release year, etc.), are
   loaded into the Qdrant collection as "points", each containing an id, a vector, and a payload.

**Checking semantic search (without the generator).** The query "Robert De Niro" is encoded by the same
embedding model, and the 5 nearest documents are found — the result turned out to be fairly accurate.
In narrow domains like "film reviews" search metrics usually come out noticeably higher than in real
applied tasks.

**Building the RAG generator.** The prompt: "You are a Russian-speaking expert in cinema. You have
access to a set of film reviews; use them to answer the following question fully and accurately... Make
sure the answer is detailed, specific, and directly relevant to the question, without adding
information that is not supported by the provided reviews". The key element of the prompt is the
explicit restriction to answer only on the basis of the provided context, without drawing on other
sources: this is precisely what limits the model's hallucinations.

Assigning the model a role (in the prompt — "you are a cinema specialist") plays a positive part,
especially for a small model.

**First RAG test.** For the query about Robert De Niro the model still produced two nonexistent titles —
because the film title isn't always mentioned in the review texts, and the model can't match a review
to a specific title.

**Improvement 1: adding the film title to the context.** The film title from the metadata was added to
the context (not just the review text). The result improved sharply — the model started correctly
listing films. The only change was adding the title to the context; the search itself wasn't changed.

**Improvement 2: adding a reranker.** A reranker based on a cross-encoder (`HuggingFaceCrossEncoder`
from `langchain_community`). The number of candidates from the base search was increased from 5 to 50;
for each (query, chunk) pair a score is computed by the cross-encoder, the documents are sorted in
descending order of score, and the top-10 are taken for the generator's context. After adding the
reranker, a review that had previously been duplicated in the results started appearing only once.

**Test: "Which film won the most Oscars?"** The model answered correctly — "Titanic" (11 Oscars),
relying on a review where this was explicitly mentioned.

**Test: "How old is Tom Holland?"** The model correctly gave no answer, since there's no such
information in film reviews, and the prompt restricts the model to answering only on the basis of the
context — the hallucination-limiting mechanism worked as intended.

**Improvement 3: query rephrasing (multi-query).** The original query is reformulated by the model into
several (in the example — 3) different variations via a separate prompt ("write N different variations
of the user's question in order to retrieve relevant documents from the vector database... don't write
the answer to the question itself"). A search is performed separately for each variation, and the found
chunks are combined into a common context. A test on the query "recommend a light comedy" showed that
the approach didn't help much — among the recommendations were clearly not comedies. The limitation —
the amount of available context is capped by the context window size of the chosen (small) model (in
the example — 512 tokens).

**Improvement 4: filtering by metadata.** Vector databases usually support built-in filtering by
metadata keys (in the example — the film's release year, in Qdrant — via a range-setting function). A
test without a filter ("a list of the 5 best melodramas of the 80s") gave a clear hallucination — not a
single film matched either the decade or the genre. A test with a filter by years (the 1980s) gave a
noticeable improvement — all the films turned out to be from the right decade, though the genre match
still wasn't perfect.

## Conclusion

A basic RAG system on real data (search + generator) can be assembled literally in a single session,
without serious difficulties. But when the system is further tested with various questions, nuances
quickly surface — in some places the model hallucinates, in others it gives no answer even though it
could. This kicks off a long, iterative improvement process that can take not hours but weeks: the
result strongly depends on the data, the way of cutting into chunks, the meta-information added, the
choice of model, and so on. There is no single solution that you "fix once and you're done". At the
same time, most of the improvement techniques covered are not super-complex engineering solutions but
relatively small, accessible tricks that can quite well be implemented solo, without a large team.
