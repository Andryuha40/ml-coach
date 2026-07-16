# Model interpretability and explainability (XAI)

These notes were assembled entirely from a single source — the seminar
"Explainable AI" (stream 2, June 10). There is no separate lecture transcript
on this topic, so all the material is from the seminar.

## Why model explainability is needed

Explainable AI (XAI) is a direction in AI, machine learning, and deep learning
that deals with developing approaches, methods, and practices that make it
possible to explain the decisions made by a model.

For the last 3–4 years there has been a steady trend toward explainability
requirements: it either appears directly in job postings or is actively used
in practice, and the topic is asked about more and more often in interviews.
The reason is that it is important to understand why a model produces one
answer or another (the so-called transparency of a model's operation is often
a requirement of a specific project).

If we ground this in classical machine learning (as opposed to large language
models, where fully explaining a specific answer is not yet possible), we can
quite well understand which features and data have a decisive influence on a
model's answer.

## Two sources of explainability requirements: business and legislation

1. **Business requirements.**
2. **Legislative requirements** — these differ across jurisdictions. An
   example is the European GDPR (General Data Protection Regulation), which is
   associated with the familiar "Accept Cookies" banner on websites. Russia
   also has legislation regulating, among other things, personal data and, to
   some extent, model explainability.

There are separate laws in various jurisdictions requiring the transparency
of models — for example, in healthcare and in banking. If a bank uses a model
to deny a client a loan, it must be able to explain on what basis the decision
was made — what if the model is relying on an incorrect feature.
Explainability is not an abstract idea but a concrete task within data
analysis.

## Example: why explainability matters (husky vs wolf)

A classic example from the paper on the LIME method ("Why Should I Trust
You?"): they took photographs of huskies and wolves — the huskies were
photographed predominantly against a background of snow, and the wolves — not
against snow (that is how it turned out in the dataset). They trained a model
to classify "husky / wolf".

When the model was shown a photograph of a husky against a snowy background,
it made a mistake and said "wolf". Upon interpretation, it turned out that the
key part of the image the model looks at is the snow: if there is snow in the
photo, the model predicts "wolf"; if there is no snow — "husky". The face, the
shape of the ears, the color of the eyes did not matter to the model.

Conclusion: interpretation makes it possible to understand how adequate the
model is with respect to the set of features that turned out to be important
to it when forming a prediction.

## Interpretability and explainability: two approaches

- **Interpretability** — the goal is to initially create a clear,
  understandable, well-interpretable model (roughly — a
  statistical/econometric model in which it is clear where a given answer
  comes from).
- **Explainability** — we take an already-built complex model and try to
  understand what its answer depends on inside the model — we "wiggle" the
  inputs and see how the output changes.

## Interpretable models: linear and logistic regression

The family of models that are well-interpretable by design: linear
regression, logistic regression, decision trees, and random forest; boosting
models are interpretable to some degree as well.

In linear regression, interpretation is "built in" — each feature has a
weight obtained during training. A weight that is large in absolute value —
the feature matters for the prediction; a small weight — it matters almost
none.

### Example on the Boston Housing dataset

A dataset about real estate: the target is the price of housing.

Two different goals worth distinguishing:
- **The "grand" goal** — to understand how the model works in general (in the
  general case we do not know this).
- **The utilitarian goal** — to understand which features have the maximum
  influence on the answer of a specific model. It is this one you should focus
  on in practice.

The features of the dataset are initially of different scales, so comparing
them with each other is incorrect without scaling (just as when training
linear regression). After scaling and training, the coefficients of the
linear regression are sorted by absolute value.

In first place by importance was the feature **LSTAT** (the proportion of the
population with low social status / the proportion of unemployed and socially
vulnerable groups in the district) — the higher the level of such "social
disadvantage" in a district, the lower the price of housing (a negative
relationship). Logistic regression is interpreted similarly — a hyperplane is
built that separates the feature space, and the sign/magnitude of the
coefficients shows a feature's contribution.

## Interpretable models: decision trees and random forest

In trees the principle of interpretation is different. A tree is a sequence of
nodes with predicates (questions) to the data; during construction, each next
question is chosen so as to reduce the uncertainty (noisiness) of the data as
much as possible — for example, by the Gini metric (impurity).

To estimate a feature's importance for a tree, one looks at how often and how
effectively this feature was used in building the tree — how strongly it
reduced uncertainty at each split. A feature that was not used at all gets a
zero contribution; a feature that was used often and effectively gets a larger
contribution. Tree-based models (in particular, the random forest) have a
ready-made attribute that immediately outputs the importance of each feature
(`feature_importances_`).

**An important caveat.** Numeric features with a large number of unique values
may receive inflated importance compared to categorical ones simply because
more splits (questions) can be defined on them, whereas a binary or
low-cardinality feature is "asked about" only once.

On the Boston Housing dataset, for the random forest LSTAT again turned out to
be in first place, but the rest of the feature order changed compared to
linear regression (for example, RM — the number of rooms, and DIS — the
distance to the business center — took different positions). Checking the
number of unique values confirmed that LSTAT and RM indeed have many of them —
this is consistent with the hypothesis about the inflated importance of "rich"
numeric features.

**A practical conclusion.** The feature importance from a single model (for
example, the feature importance of a random forest) does not prove that a
feature is actually causally important for the target — it is important
specifically for the answer of this particular model. Feature importance does
not establish a causal relationship, but only highlights the features
significant for a specific model.

A stronger argument is obtained if you take several different
well-interpretable models and look at the intersection of their top features:
if, say, out of 10 important features 6 are confirmed by two different models,
this is a weightier (although still not conclusive) premise in favor of the
feature actually affecting the target.

## Other interpretable models

- **KNN** — interpretable thanks to the compactness hypothesis: similar
  objects are located close to each other in the feature space.
- **The naive Bayes classifier** — is also fairly well interpretable.
- **SVM** — to some degree.

## Model-agnostic methods: the general idea

Besides models that are interpretable "by construction", there is a class of
methods that do not depend on the internal structure of the model
(model-agnostic methods). The idea: it does not matter what is inside the
model (the "black box") — we "wiggle" the inputs and see how the output
changes.

## The ICE method (Individual Conditional Expectation)

One of the first model-agnostic methods:

1. We take a test sample and one specific object (sample) from it.
2. We choose one feature (for example, RM, LSTAT, or AGE).
3. For this object, we run through all values of the chosen feature — from the
   minimum to the maximum (in the range that occurs in the test sample), while
   the object's other features remain fixed.
4. We look at how the model's answer changes as the feature's value changes.
5. We repeat the same for all objects of the test sample — obtaining a set of
   curves (one per object).
6. By averaging all these curves, we obtain the average dependence of the
   target on the varied feature (essentially, a partial dependence plot).

On the Boston Housing dataset such a dependence is clearly visible for the
features RM and LSTAT: an increase in the number of rooms positively affects
the price of housing, an increase in LSTAT — negatively. The feature AGE (the
age of the housing) has almost no effect on the target.

From the individual curves you can notice outliers — objects that behave
atypically (the curve goes "the other way"). For example, for a pair of
objects, as the "socially disadvantaged" feature increases, the price, on the
contrary, grows — a possible explanation: someone deliberately prefers a
cheaper or not-the-most-prosperous district (for example, because of a
familiar social environment).

**Limitations of the ICE method:**
- Only two features can be visualized at the same time.
- The method proceeds from the assumption of independence of features, which
  is often untrue in practice.
- With a large number of features and objects, the plot turns into a
  hard-to-read "blob".
- The main downside is that there is no numeric metric of a feature's
  contribution, only a visualization; there is no measurable importance
  estimate.

## The LIME method

LIME (Local Interpretable Model-agnostic Explanations) is a comparatively
young method; the paper came out in 2016–2017 and is titled "Why Should I
Trust You?". Unlike most other explainability methods, LIME has a more or less
rigorous mathematical justification. It is also the only method whose output
is not a plot or a coefficient but a whole (surrogate) model.

**The LIME algorithm:**
1. We take an object x* for which we want to obtain an explanation of the
   complex model's answer.
2. Around x* we randomly sample new objects in the feature space.
3. We run these new objects through our complex ("black box") model and obtain
   a prediction for each.
4. We weight the resulting objects by their distance from the original point
   x* — objects located closer get a larger weight.
5. On this local weighted dataset we train a simple model (for example, a
   linear one).

In the neighborhood of the point x*, this simple model behaves approximately
the same as the original complex one — we approximate the complex model with a
simple one, but not over the whole domain, only locally. From the coefficients
of the resulting simple model, one can judge the importance of features
specifically in this local area.

### LIME example: text classification (20 Newsgroups)

The 20 Newsgroups dataset — about 20,000 news documents split into 20
categories. A naive Bayes classifier was trained on TF-IDF features (accuracy
and F1 around 0.83).

LIME was applied to two classes — "atheism" and "christianity" — to understand
which words the model builds its prediction on. Part of the highlighted words
turned out to be expected (related to the topic of atheism/religion), but
strange, at first glance irrelevant words were also encountered. On examining
the source texts, it turned out that the model relies not so much on the
content of the message as on meta-information — for example, on the domain of
the sender's email address.

Conclusion: despite the formally high quality of the classifier, it turned out
to be unreliable — a message from the same domain but on a different topic
will most likely still end up in the usual category. LIME helped reveal this.
A possible solution is to build a separate model on the meta-information and a
separate one on the text, and then combine them into an ensemble.

### LIME example: image classification (superpixels, a frog)

For interpreting images, the concept of superpixels is used — not individual
pixels but segments made of a group of similar neighboring pixels. The
algorithm covers different superpixels of the original image in turn,
obtaining many slightly altered versions, and computes the model's prediction
for each version. By comparing the changes in the prediction with the covered
areas, one can understand which superpixels contribute most to the model's
answer.

In the example with classifying "is there a tree frog in the image", the
largest contribution to the prediction comes from the frog's face.

**Practical application:** for example, in the task of diagnosing equipment
failure from an image, you can determine which part of the component
contributes most to the prediction and, for example, turn the camera so that
it better captures precisely that area — this can improve the model's quality
without retraining.

## The SHAP method (Shapley values)

Shapley values are an approach from game theory that makes it possible to
estimate the averaged importance of features for a model, as well as the
contribution of each feature individually.

**The idea (an analogy with a team-building event).** Imagine a corporate
team-building event where the participants are divided into pairs/small groups
for tasks, and then reshuffled in different combinations (in groups of 2, 3,
4) and given tasks again. By observing how the result changes with different
team compositions — with and without the participation of a particular person
— one can estimate each specific participant's contribution to the overall
result.

By analogy, in SHAP all possible combinations of features (the analog of
"teams") are considered, and for each combination it is estimated how adding a
particular feature changes the model's prediction. By averaging the change in
the prediction over all possible combinations (pairs, triples, and so on), we
obtain the Shapley value for a feature — an estimate of its averaged, as well
as individual, contribution to the model's answer.

## Conclusion and a word of caution

The main explainability methods: ICE, LIME (including for texts and images),
and Shapley values (SHAP).

**An important word of caution.** When working with any of these methods, you
must not substitute conclusions about the model's operation for conclusions
about the target. You cannot unambiguously claim that a given feature affects
the target relying only on the feature's importance for a specific model —
feature importance shows what is important for the model's answer, but does not
establish a causal relationship.

Separately (and beyond the scope of this session) there is a direction called
**Causal Inference** — the search for cause-and-effect relationships: the
question is not simply "which features correlate with a user's click", but
"what really affects the click, and what needs to be changed to increase
conversion". The topic is closely related to model interpretation, but
represents a separate direction.

There is also a lot of research on interpreting large language models — the
direction is still at an active research stage, but there are noticeable
advances there (in particular, experiments of the alignment-faking type —
research into how a model behaves differently depending on whether its answers
are used for further training).
