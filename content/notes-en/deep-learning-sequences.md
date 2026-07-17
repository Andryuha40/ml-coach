# Recurrent networks and transformers (RNN, LSTM, attention, BERT, GPT)

These notes are assembled from a course lecture (lecture recording 16 — essentially
all the architecture theory: RNN, LSTM, GRU, the attention mechanism, the
transformer, BERT, GPT) and the seminar notebook "Getting started with PyTorch"
(basic practice: tensors, automatic differentiation, training a fully connected
network — these code patterns will come in handy for PyTorch exercises, although
the notebook itself covers general framework fundamentals rather than recurrent
networks or transformers directly). Evgeny Patochenko's repository has no material
on this topic — it falls outside the scope of his program (this is a separate
course module on deep learning and LLMs, which isn't in the repository at all).

## From n-grams to neural text generation

An n-gram text-generation model is an ordinary probabilistic model in which the
next token depends only on a limited subset of the preceding tokens. This limits
generation quality: the more of the preceding text the model can take into
account, the better the generation. The idea behind neural models is to account
for all preceding tokens rather than a limited window.

The general shape of a text-generation architecture: text is fed in, processed,
and turned into a feature vector; a linear layer is applied to the vector,
transforming its dimensionality into the vocabulary size; a softmax is applied to
the output of that layer, turning the values into a vector of probabilities (the
coordinates sum to one, each coordinate being the probability of a specific
vocabulary token). One token is generated per pass, and the network is run many
times.

Convolutional networks turned out not to fit this task for two reasons: first,
convolution is slow (you have to process the entire input sequence at once, and
growing the number of convolutional layers sharply increases computation time);
second, all input tokens contribute to the prediction with equal weight, even
though in reality tokens closer to the current position usually influence the
prediction more strongly than tokens at the start of the sequence.

## Recurrent neural networks (RNN)

RNNs were designed specifically for sequential data: they process all the tokens
of the input sequence one after another. Today RNNs are almost never used in
practice for text, but they are used for time series — an architecture invented
for text unexpectedly turned out to be useful there too.

The general scheme: the same RNN block (with the same weights) is applied many
times. The first block takes the start of the sequence and predicts the next
word; each subsequent block takes the previous hidden state and the current word
and returns an updated state (making a prediction for the current step). This
way you can process text of arbitrary length by repeatedly applying the same
block.

### How an RNN block works

The block takes the previous hidden-state vector h and the vector representation
(embedding) of the current token x. The vector h is passed through a linear layer
with weight matrix U, and the vector x through a linear layer with weight matrix
W; the results are added, and the sum passes through an activation function
(sigmoid or hyperbolic tangent) — this gives the updated hidden-state vector h_t.
If you also need the network's output (a prediction), an additional linear layer
with weight matrix V and a bias is applied on top of the hidden state,
transforming the vector's dimensionality into the vocabulary size — this yields a
vector of logits.

### Training an RNN

Training means maximizing the product of the probabilities of each next token
given all the preceding ones (you need to raise the probability of the correct
token and lower the probability of the rest). To turn a maximization problem into
a minimization one, a negative logarithm is applied to the product of
probabilities — this gives the loss function, **cross-entropy**; for the whole
corpus, cross-entropy is averaged over all texts. The total loss over the whole
sequence is the sum of the losses over each block.

The main problem in computing gradients is the gradient with respect to matrix U:
since h_t depends on the previous hidden state h_(t-1), which itself depends on
the same matrix U, by the chain rule the derivative unrolls recursively across
all preceding steps — you get a sum of products of derivatives of the form
dh_t/dh_(t-1), multiplied by the derivative with respect to U at each step.

### Exploding and vanishing gradients

Two problems are hidden in this sum of products.

**Exploding gradient.** If the norm of the product of derivatives is greater than
one, then the more steps there are, the more products accumulate — the gradient
becomes huge, overflow occurs, and NaN values appear in PyTorch; the model stops
training. Solutions: regularizing the model, lowering the learning rate, and,
most importantly, **gradient clipping**:
- the "correct" way is normalization: if the gradient norm exceeds a specified
  maximum (max_norm), the gradient is multiplied by the minimum of one and the
  ratio of max_norm to the current norm — this controls the length of the
  gradient vector without changing its direction;
- the "lazy" way is component-wise clipping of all gradient coordinates to a
  specified interval (for example, from −c to c).

**Vanishing gradient.** If the activation function is the sigmoid (as is most
often the case), gradients are more prone to vanishing than exploding. The amount
of vanishing depends on the number of products in the chain, that is, on the
distance between tokens in the text: for tokens far apart the gradient comes out
small, for close ones — more significant. Because of this, an RNN captures
dependencies between nearby words well and dependencies between distant ones
poorly. This follows from the very nature of the recurrent network and cannot be
fixed with simple engineering tricks — you need to change the architecture. LSTM
and GRU were invented for exactly this.

## LSTM (Long Short-Term Memory)

To keep the model from "forgetting" distant tokens, a memory vector (**cell
state**) was added to the basic RNN block — a special vector carried from step to
step, and along with it three "filters" (gates):
- **forget gate** — a linear layer over the previous hidden state and the current
  input, with a sigmoid at the output (values from 0 to 1); it is multiplied
  element-wise by the memory vector — coordinates close to 0 "erase" the
  corresponding information from memory, coordinates close to 1 preserve it;
- **input gate** — computed by a similar formula, but with a hyperbolic tangent
  (values from −1 to 1); it extracts from the current input the information that
  needs to be added to the memory vector;
- **output gate** — extracts from the memory vector exactly the information needed
  for the prediction at the current step.

An example on a text-sentiment classification task: on encountering a marker word
("delicious"), the model can "forget" part of the information accumulated earlier
(the forget gate) and at the same time register that the text is probably
positive (the input gate); if the word is neutral, there is nothing to add to
memory. The gate values are not set by a heuristic in advance but are tuned
automatically during training — by minimizing the loss function.

The problem with LSTM: because of the memory vector and the three gates, the
number of parameters inside the block increases roughly fourfold compared to an
ordinary RNN — the network runs slower and more expensively and is more prone to
overfitting.

## GRU (Gated Recurrent Unit)

A modification that reduces the number of parameters while preserving the ability
to capture long dependencies: the forget gate and the input gate are merged into
one, so two gates are used instead of three (or four extra layers, as in LSTM) —
the number of weights inside the block is cut nearly in half.

GRU is not unambiguously "better" than LSTM: on a large text corpus LSTM usually
shows a slightly better result (it has more weights, so it is able to "learn"
more data); if there is little text, GRU usually performs better — thanks to its
smaller number of parameters and lower tendency to overfit.

## Bidirectional and multilayer RNNs

**Bidirectional RNNs.** The meaning of a word sometimes changes depending on the
word that follows it, so text can be read not only left to right but also right
to left: one independent network reads the text in one direction, another in
reverse, their outputs are concatenated and passed through a linear layer. It
uses twice as many parameters. For text generation it barely helps (there is no
sense in "reading backward" during generation), but for classification it often
gives a noticeable quality boost.

**Multilayer RNNs.** Several recurrent networks are stacked on top of each other:
the output of one network is fed into the next along with its hidden state — this
allows increasingly complex features to be extracted. The downside is that the
number of parameters grows, and a deep network becomes harder to train (vanishing
rather than exploding gradients occur more often, since the gradient simply
doesn't reach the far layers). In practice almost no one uses more than two
layers in such networks.

## The attention mechanism and the seq2seq task

The attention mechanism and the transformer architecture were proposed in the
paper "Attention Is All You Need" (2017) — this architecture turned out to be so
successful that further development of classical RNNs largely stopped.

The transformer was originally designed for the **seq2seq**
(sequence-to-sequence) task — transforming an input sequence into a new one
(machine translation, summarization, text-style transfer). Solving such a task
uses a two-model concept: an **encoder** encodes the input sequence, and a
**decoder** takes the encoder's output and generates the new sequence.

Before the transformer, this task was solved by a pair of recurrent networks: an
RNN encoder encoded the input block by block, the hidden state of the last block
was passed to the decoder as its initial state, and the decoder generated the
continuation. The main problem was the **bottleneck** between the encoder and the
decoder: all the information about the input sequence had to be compressed into a
single fixed-dimensional vector, so that a significant part of the information
about a long sequence was lost.

### The idea of the attention mechanism

Instead of relying only on a single compressed vector, the decoder is allowed to
"look" at arbitrary encoder outputs. At each step the decoder computes **scores**
— measures of closeness (for example, the dot product) between its own current
vector and each of the encoder's hidden vectors. The scores are passed through a
softmax, giving a probability distribution over the tokens of the input sequence;
these probabilities are used as weights to weight the encoder vectors — this
gives a **context vector**, which is returned to the decoder and taken into
account when generating the next token together with the main hidden state.

The point: the decoder gains the ability at each step to choose which tokens of
the input sequence to extract information from, instead of relying on a single
fixed vector.

## The transformer architecture

The real breakthrough came when the attention mechanism was developed into a
standalone architecture of its own and recurrence was abandoned entirely. A
transformer consists of **encoder** and **decoder** blocks; each large "box" on
the diagram (Multi-Head Attention, Feed Forward) is actually not a single layer
but a composition of linear layers and activation functions.

### The encoder block

**1. Multi-Head Self-Attention.** Self-attention is a modification of attention
in which each token of the sequence can directly interact with any other token of
the same sequence. For each token, three vectors are created: **Query** — what
the token "requests" from the others; **Key** — information about what is
contained inside the token; **Value** — the information that ultimately gets
extracted.

The self-attention formula (in words): take the softmax of the product of the
Query matrix and the transposed Key matrix, divided by the square root of the
dimensionality of the Key vector (a normalization constant — a technical trick so
that after multiplication the variance of the values doesn't "blow up"), and
multiply the result by the Value matrix. The matrices Q, K, V are obtained by
applying separate linear layers (with different weights) to the whole input
sequence. The product of Q and the transposed K yields a square matrix of scores
for the relevance of one token to another.

**Multi-Head Attention** — unlike ordinary self-attention (which simply looks at
other tokens), it allows looking at different *aspects* of one and the same token
at the same time (for example, for the word "shines" — separately "what shines"
and "how it shines"). The vectors Q, K, V are split into several independent
parts ("heads"), each extracts its own information, and the results across all
heads are concatenated and passed through one more linear layer.

**2. Positional Encoding.** Self-attention by itself does not account for token
order — each token can look at any other, but "doesn't know" which one is where.
An embedding of the token's position in the text is added to the ordinary word
embedding (it is trained along with the rest of the model's parameters) — this
gives the final representation of the token.

**3. Feed Forward.** A fully connected network of two linear layers (the first
increases the dimensionality, the second returns it to the original, with an
activation function between them, usually ReLU) that processes the information
after self-attention independently for each token.

**4. Add & Norm.** As in ResNet — a skip connection (the value before the layer
is added to the value after it, to combat vanishing gradients), plus
normalization after the addition to stabilize training. A purely technical block.

The transformer encoder is a block of Multi-Head Self-Attention, positional
encoding, Feed Forward, and Add & Norm, repeated several times.

### The decoder block

Compared with the encoder, two new elements are added:

**Masked Multi-Head Self-Attention.** Since the decoder generates text, during
training it must be forbidden from "peeking" at tokens after the current one —
otherwise it would learn to rely on a future that hasn't been generated yet. A
masking matrix is added to the attention formula: for tokens after the current
one, minus infinity is substituted, and for tokens before the current one, zero,
so that the weights of future tokens are zeroed out after the softmax.

**Cross-Attention.** Allows the decoder to obtain information from the encoder:
the Query vectors are taken from the output sequence (the decoder), while Key and
Value are taken from the input sequence (the encoder). In essence, the decoder
"requests" from the encoder the information it needs for the current step.

**Summary of the transformer:** the seq2seq task is poorly solved by a pair of
RNNs because of the bottleneck, whereas the attention mechanism lets the decoder
"peek" into any part of the encoder's output and bypass this narrow spot. In its
original form (encoder + decoder together, as in the original paper) the
architecture is almost never used in modern practice — its "halves" are much more
often used separately.

## From the transformer to BERT and GPT

Text tasks can be classified by the ratio of input length to output length:
- **many-to-many, equal length** (NER, part-of-speech tagging) — an **encoder**
  alone is enough;
- **many-to-many, different length** (summarization, translation, style transfer)
  — a **decoder** is needed;
- **many-to-one** (classification, regression) — an **encoder** is enough.

Building on this observation, two companies independently proposed architectures
based on "halves" of the transformer: OpenAI in 2018 developed **GPT** — a
modified decoder; Google around the same time developed **BERT** — a modified
encoder.

### BERT (Bidirectional Encoder Representations from Transformers)

At its core is the transformer encoder block. The main advantage is that the
model can be **pre-trained** on a large mass of unlabeled data (the model learns
to "understand language") and then quickly **fine-tuned** for a specific local
task. The base BERT (2018) contained 110 million parameters and was trained on 16
gigabytes of text (BooksCorpus + English Wikipedia) for 4 days on 16 GPUs.

**BERT pre-training** consists of two tasks solved simultaneously:
1. **Masked Language Modeling (MLM)** — 15% of the tokens in the text are chosen
   at random: 80% are replaced with the [MASK] token, 10% with random tokens (so
   the model doesn't overfit to the mere fact of masking), and 10% are left
   unchanged. The model has to reconstruct the original tokens at these
   positions.
2. **Next Sentence Prediction (NSP)** — two concatenated texts are fed into the
   model (in 50% of cases the second really continues the first, in 50% it is a
   random unrelated text); from the representation of the service token [CLS] the
   model predicts whether the second text is a logical continuation of the first.
   Additionally, **Segment Embeddings** are used, explicitly indicating whether a
   token belongs to the first or the second text.

On a benchmark for comparing classification models, BERT at the time of its
release outperformed its nearest competitor by 7%. It was later discovered (after
the fact, not designed in advance) that different layers of BERT extract
increasingly complex linguistic features as the layer number grows: early layers
— parts of speech, further on — syntactic constituents (subject, predicate), then
— named entities, then — semantic roles and coreference.

**Modifications of BERT:** RoBERTa (Meta) — trained longer and on a larger volume
of data, with dynamic masking (the MLM mask is generated anew each time an
example is fed in); ALBERT — reduces the number of parameters through parameter
sharing (sharing weights between layers) and factorization of the embedding
matrix; also DistilBERT, DeBERTa, ELECTRA, and others. BERT-like models are still
actively used for classification, where they are much lighter than full-fledged
large language models — data preprocessing, unwanted-content detection, and
similar tasks where text divides fairly unambiguously into classes.

### GPT (Generative Pre-trained Transformer)

At its core is the transformer decoder block, from which cross-attention has been
removed (since there is no longer an encoder to "check against"). The
architecture: embeddings + positional embeddings → a block of masked multi-head
attention, feed forward, and add & norm repeated several times → a linear layer
and softmax, giving a probability distribution over the vocabulary.

The growth in the scale of GPT models: GPT-1 — about 117 million parameters,
trained on ~1.5 gigabytes of data; GPT-2 — about 1.5 billion parameters, 40
gigabytes of data; GPT-3 — 175 billion parameters, 45 terabytes of text data.

The key feature thanks to which GPT "took off": the language-modeling task
(generating the next token) turned out to be so universal that almost any other
task can be reduced to it (for example, text classification can be reformulated
as generating the index of the required class).

## Fine-tuning large models in practice

Large models are almost never trained from scratch — you take a ready-made model
with weights trained by someone else and act according to one of the strategies:
- **Full fine-tuning** of all weights — risky for models with billions and
  trillions of parameters: a local dataset is usually much smaller than needed,
  the model quickly overfits, "memorizing" the sample.
- **Partial fine-tuning** — freeze most of the weights (usually the early layers,
  responsible for general language understanding) and fine-tune only the model's
  "head" or particular parts of it.
- **PEFT (Parameter-Efficient Fine-Tuning)** — methods in which only a small
  portion of the weights is trained while the main part stays frozen.
- **RAG (Retrieval-Augmented Generation)** — instead of fine-tuning the weights,
  additional relevant information is slipped into the context of a ready-made
  model on the fly — not training in the strict sense, but a way to "refine" the
  answer through context.
- **Agent systems** — models capable of "gathering" the needed information
  themselves and adding it to their context without a separate search
  infrastructure.

In practice, a combination of these approaches is most often used to solve
business tasks, rather than any single method in pure form.

## Practice: PyTorch fundamentals (reference code for exercises)

The seminar notebook "Getting started with PyTorch" doesn't specifically cover
RNN/LSTM/transformers, but it gives the general foundation for working with the
framework that is used when implementing any neural networks in the course —
tensors, automatic differentiation, and the training loop.

### Tensors

```python
import torch

# Creating tensors
t = torch.zeros((5, 3))
t = torch.ones((2, 3, 4))
t = torch.randn((2, 3, 2, 4))  # from the standard normal distribution

# in-place operations (with a trailing underscore in the method name)
t = torch.ones((2, 3))
t.zero_()  # zeroes "in place" without creating a new tensor
```

### Automatic differentiation (autograd)

```python
x = torch.tensor(-2., requires_grad=True)
y = torch.tensor(5., requires_grad=True)
z = torch.tensor(-4., requires_grad=True)

q = x + y
f = q * z

f.backward()

print(f'df/dz = {z.grad}')
print(f'df/dy = {y.grad}')
print(f'df/dx = {x.grad}')
```

### A fully connected network and the training loop

A fully connected layer is a combination of a linear transformation and an
activation function. A network is specified as a sequence of such transformations:

```python
import torch.nn as nn

NN = nn.Sequential(
    nn.Linear(1, 5, bias=True),
    nn.Tanh(),
    nn.Linear(5, 5, bias=True),
    nn.Tanh(),
    nn.Linear(5, 1, bias=True)
    )
```

A more flexible way is to subclass `nn.Module` and describe the forward pass in
the `forward` method:

```python
class Net(nn.Module):
    def __init__(self, dim=1):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(dim, 5)
        self.tanh1 = nn.Tanh()

        self.fc2 = nn.Linear(5, 5)
        self.tanh2 = nn.Tanh()

        self.fc3 = nn.Linear(5, 1)
        self.tanh3 = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh1(x)

        x = self.fc2(x)
        x = self.tanh2(x)

        x = self.fc3(x)
        x = self.tanh3(x)

        return x
```

The training function is a loop over epochs with five standard steps (forward →
loss → zero_grad → backward → step):

```python
from tqdm.auto import tqdm

def train(model, X, y, criterion, optimizer, num_epoch):
    '''
    args:
        model - the neural network model
        X and y - the training set
        criterion - the loss function, taken from the `torch.nn` module
        optimizer - the optimizer, taken from the `torch.optim` module
        num_epoch - the number of training epochs. I.e. the number of gradient
                    steps performed for each object in the sample
    '''
    for t in tqdm(range(num_epoch)):
        # Compute our model's predictions
        y_pred = model(X)

        # Compute the value of the loss function on the obtained prediction
        loss = criterion(y_pred, y)

        # Zero out the previously computed gradient values
        optimizer.zero_grad()

        # Compute the new gradients
        loss.backward()

        # Take a gradient-descent step
        optimizer.step()

    return model


criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(NN.parameters(), lr=1e-2)

NN = train(NN, X, Y, criterion, optimizer, 30)
```

### Moving computation to the GPU

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'

NN = Net(1)
NN = NN.to(device)  # move the model parameters onto the device

# during training the input data is also moved onto the device:
y_pred = model(X.to(device))
loss = criterion(y_pred, y.to(device))
```
