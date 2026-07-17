# Class imbalance

These notes are assembled entirely from two course seminars (stream 2, weeks 14
and 17) — Evgeny Patochenko's repository has no separate material on this topic,
the whole basis is local. The sources are oral seminar transcripts, so there is no
ready-made code for the topic (unlike other course topics) — the techniques
mentioned were shown in the seminars as implementations from specialized libraries
(primarily `imbalanced-learn`), literally in one or two lines, without going
through their internal code.

From the second seminar (week 17), only the section directly about class imbalance
is taken into these notes — the rest of that seminar (about covariate shift,
feature selection, PCA) belongs to another course topic ("Feature selection and
dimensionality reduction") and is not duplicated here.

## Why class imbalance is a problem

Class imbalance is a very frequent situation in classification tasks: imbalance
occurs significantly more often than balance, only its degree can vary. There is
no hard threshold beyond which imbalance becomes a problem — much depends on the
specific task: sometimes strong imbalance doesn't stop a model from working well,
and sometimes even a small skew (around 20%) already prevents finding the
dependency. As a rough guide, you can treat a difference on the order of 10–20%
between the minority and majority class as a signal to pay attention to it.

The essence of the problem: there is a majority (predominant) class and a minority
class, of which there are significantly fewer objects in the sample. The model
learns and memorizes the patterns of the majority class better simply because it
sees far more of its examples, and works worse with the minority class — it simply
has fewer objects on which to learn the corresponding dependency. As a result the
model tends to classify objects toward the class it "knows" better — and a naive
quality metric (for example, accuracy) can look deceptively high in this case: a
model that always predicts the majority class will, at a 90/10 imbalance, get 90%
accuracy without actually having learned anything about the minority class. This
is precisely why tasks with imbalance need special metrics (ROC-AUC,
precision/recall on the minority class — see the topic on classification metrics)
and special methods for combating the imbalance itself.

The task is extremely common in practice: detecting fraudulent transactions,
predicting loan repayment/default, predicting disease/no disease, and many similar
scenarios by their nature contain strong class imbalance.

## A demonstration on a dataset (a typical pipeline for working with imbalance)

At one of the seminars, imbalance was demonstrated on a training dataset with
built-in imbalance (many features, a binary target variable). The typical order of
work:

1. The data is loaded and brought into DataFrame format.
2. The sample is split into training and test (`train_test_split`).
3. The numerical features are scaled (`MinMaxScaler`).
4. To visualize the data on a plane, the PCA (Principal Component Analysis)
   dimensionality-reduction method is applied — for example, two principal
   components are taken, into which all the information from the multidimensional
   feature space is "compressed." PCA is conceptually related to singular value
   decomposition (SVD) — it is essentially the same result from different sides.
   Here PCA is applied solely for the sake of visualization on a plane, although
   as an element of data preprocessing PCA is often useful in its own right too.

After the visualization, the graph makes it plainly visible that one class
predominates over another — and that building a classifier directly under such
conditions is pointless: first you need to do something about the imbalance.

## Methods for combating imbalance

### Resampling

**Random Oversampling.** Objects are randomly resampled (repeated) from the
minority class until its size matches the size of the majority class. Some objects
of the minority class may end up in the new sample several times, while some — not
at all.

Downside: for the duplicated objects the weight and importance to the model
effectively increases — it learns them and the pattern "hidden" in them better,
devoting disproportionately much attention to them compared with unique objects. In
essence, the idea of oversampling is, by copying, to even out the sizes of the
classes and thereby give additional weight to the objects of the minority class.

**Random Undersampling.** The opposite — a subsample comparable in size to the
minority class is randomly selected from the majority class.

Downside: objects that could carry valuable information are lost — especially
problematic when the minority class really is very small (for example, on the order
of a hundred objects): after undersampling the majority class is also cut down to a
comparable size, and the model may simply lack data. If objects that were located
close to the boundary with the minority class are discarded from the majority
class, there is a risk of distorting reality in that area: an object that would
previously have been confidently assigned to the majority class may now be
mistakenly assigned to the minority class, because there is now less information
about the majority class there.

Conclusion: which approach works better strongly depends on the specific data —
there is no universal rule. In the demonstration with a linear SVM: without
combating the imbalance, ROC-AUC was around 0.47-0.53 (essentially random
guessing); after oversampling — around 0.89; after undersampling — around 0.88.
Both methods noticeably improve quality, but which is better is not known in
advance.

### Synthetic generation: SMOTE, ADASYN, Tomek Links

**SMOTE.** Synthetic augmentation of the minority class: two objects of the
minority class are taken, a line is drawn between them on the plane (conditionally),
and in the middle a new artificial object of the same class is "synthesized." This
way the minority class is augmented with artificially created objects.

**ADASYN.** A similar principle, but the points for synthesis are chosen not at
random but meaningfully. A nearest-neighbors (kNN) classifier is taken, making
predictions on the sample; those objects of the minority class on which the
classifier makes mistakes (predicts the majority class instead of the minority) are
marked. It is precisely around such "problematic" objects that new synthetic
objects of the minority class are generated — the model is helped precisely where
it finds it hardest to separate the classes.

**Tomek Links.** The idea is to "push apart" the space between the classes so that
the separating boundary is easier to draw. The algorithm finds close pairs of
objects of different classes at the blurred boundary between them and removes one of
the pair (usually the one that "gets in the way") — thereby the space around the
boundary becomes sparser. It can remove objects of both the majority and the
minority class — which side it is more advantageous to thin out depends on the
nature of the data (a hyperparameter of the method is responsible for this).

**SMOTE + Tomek Links.** A combination: first the minority-class data is
synthetically augmented via SMOTE, then the boundary is "thinned out" via Tomek
Links.

**Final comparison in the demonstration.** Plain Tomek Links by itself did not help
much, while ADASYN, SMOTE, and the SMOTE+Tomek combination gave a result in the
region of 0.89 ROC-AUC — a noticeable rise compared with the baseline 0.47-0.53.

### Class weighting (class_weight)

A separate approach: instead of changing the sample itself — weight the classes
during model training, passing the `class_weight` parameter and training on the
original, imbalanced data. The idea is to penalize the model more heavily for
mistakes on the minority class, with a coefficient proportional to the degree of
imbalance.

In the demonstration the result turned out to be unexpectedly better than all the
advanced techniques — ROC-AUC around 0.90.

**Why this happens.** Most synthetic methods (SMOTE, ADASYN, Tomek Links, and
their combinations) are based on assumptions about the data that are not always
true. For example, SMOTE assumes that the point "in the middle" between two objects
of the minority class also belongs to the minority class — a reasonable but not
guaranteed-correct assumption, which does not work on all datasets. Undersampling
distorts reality in a different way — it lowers the representativeness of the
majority class, so that in some regions of the feature space the model begins to
make mistakes where it previously classified confidently.

Class weighting is in this sense "more honest": it is mathematically close to
oversampling (it also gives greater weight to the objects of the minority class),
but it does not create any artificial or duplicated data — it simply distributes
weight among the real objects differently. That is why it distorts the original
dataset less than synthetic methods or the physical copying of objects.

**General conclusion.** There is no universal correct answer as to which method
will work better. Synthetic methods work if the assumptions they make about the
nature of the data really correspond to reality; if not — there may be no effect,
or it may turn out worse than expected. The approach to combating imbalance must
always be selected experimentally, testing different methods on the specific data.

## Ensembles + undersampling: Balanced Random Forest

A separate strategy is combining ensemble methods with undersampling. Ordinary
undersampling reduces the entire majority class down to the size of the minority
class once, irretrievably losing part of the data. Balanced Random Forest is set
up differently: when building **each** tree in the ensemble (the random forest),
the entire minority class is taken in full plus a random subsample from the
majority class of the same size. In this way each individual tree is trained on
balanced data, but collectively, across all the trees, the ensemble "sees" the
entire volume of the majority class — the majority-class data is, as it were,
smeared across different trees, and information is not completely lost, unlike
one-off undersampling.

## Other approaches to data synthesis (beyond classical ML)

- **Image augmentation.** If there is little data (for example, a thousand labeled
  images of animals for classification), you can synthesize new labeled examples:
  rotate the image, mirror it, stretch it, compress it, convert it to black and
  white, etc. The object's class does not change in the process, while the training
  set grows.
- **Generating synthetic data with the help of large language models.** A modern
  approach to synthesizing minority-class objects is to use an LLM that generates
  similar objects, reflecting the nature of the original data often more accurately
  than classical statistical methods like SMOTE. It can give a more realistic
  synthetic data increment, if the task is designed correctly.

## Summary: how to choose a method

There is no single correct answer — the methods are worth trying experimentally on
the specific task:
- **Class weighting (class_weight)** — the most "honest" and often the most
  effective method, distorts nothing in the original data, worth trying first as
  the cheapest option.
- **Oversampling / SMOTE / ADASYN** — work well if the method's assumptions about
  the data structure correspond to reality; the risk is overfitting on
  duplicated/synthetic objects.
- **Undersampling** — the risk of losing valuable information, especially with a
  small minority class; better to combine with ensembles (Balanced Random Forest)
  so as not to lose data irretrievably.
- **Tomek Links** — more of an auxiliary "boundary-cleanup" method, usually used in
  combination with others (SMOTE+Tomek) rather than on its own.
