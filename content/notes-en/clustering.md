# Clustering and visualization of high-dimensional data

These notes were assembled from a lecture and seminar in Evgeny Patochenko's
repository (`lesson_11` — the basis for the algorithms, metric formulas, and
practical code) and from a course lecture (recording of lecture 19 — unique
details: intuitive, "back-of-the-envelope" explanations of the algorithms,
practical applications of clustering, spectral clustering, as well as the
visualization of high-dimensional data with MDS and t-SNE, which logically
continues that same lecture and is reflected in the title of this topic).

## Clustering as an unsupervised learning task

**Unsupervised learning** is a class of machine learning methods in which there
is no target variable and the goal is to recover structure in the data: the
input data is not labeled, and since there are no correct answers, problems
arise with the very measurability of quality (more on this in the section on
metrics).

**Clustering** is an unsupervised learning task whose goal is to use the
internal information about the objects in the sample to find "similar" objects
and assign them to the same class. In the English-language literature this
method is sometimes called unsupervised classification.

Formally: we are given objects x1, ..., xl. We need to identify K clusters in
the data — regions such that objects within one cluster are as similar to each
other as possible, while objects from different clusters are as dissimilar as
possible. In other words, we need to build an algorithm that assigns each
object x a cluster number (a value from 1 to K).

**Clustering vs. classification.** In classification, we use a labeled training
set to learn to recover the relationship between features and already-known
classes. In clustering there is no labeling at all — the classes are
"discovered" by the algorithm from the feature description of the objects alone.

### Why clustering is useful: applications

- If you have unlabeled data and don't want to label it yourself, clustering
  lets you identify groups of objects automatically.
- Anomaly detection, identifying user segments, grouping documents/texts,
  processing geodata.
- A utilitarian scenario: for example, in a real-estate price prediction task,
  objects are loosely assigned to categories like "economy", "luxury",
  "business" — but such categories are fairly arbitrary (developers assign them
  differently). It's better to cluster the object space yourself — the
  resulting clusters may differ from the originally assigned labels and turn out
  to be more meaningful.
- Split the object space into clusters and, for each cluster, build its own
  separate machine learning model (if there is enough data). At inference time a
  new object is assigned to the cluster whose center is nearest to it, and the
  corresponding model is run for it — a kind of refinement of the object space
  before modeling.
- A fun niche application: **image compression**. Each pixel's color is a set of
  RGB intensities; k-means lets you find fewer color clusters in the data than
  there originally were — colors that are close in distance merge into a single
  cluster, thereby compressing the image.

It's important to remember: the clustering algorithm does not know what you
want. There is no way to guarantee that, say, clustering texts will produce a
partition specifically by topic — quite often the clusters turn out to be
uninterpretable, and you have to study each cluster in detail to understand why
exactly those groups emerged.

## K-means

One of the most popular methods, and often the first one tried: if the data is
simple enough and easily clusterable, the method works quite well, and people
often just stop there.

**Algorithm:**
1. Set the number of clusters k, and randomly choose k cluster centers
   (centroids).
2. Each object is assigned to the cluster whose center it is closest to.
3. A new center is recomputed for each cluster — as the arithmetic mean of the
   coordinates of all objects inside the cluster (the center of mass).
4. Steps 2-3 are repeated until convergence — until the centroids stop moving
   significantly (the difference between the old and new centroid falls below a
   threshold).

**Objective function.** K-means aims to choose centroids that minimize the total
squared distance from each object to the center of its cluster (this quantity is
also called inertia). The distance is usually taken to be the squared Euclidean
distance — not mandatory, but convenient, because a quadratic function is easier
to differentiate. Since the function is quadratic (convex), it has a single
minimum — the method is guaranteed to converge, and the movement of the
centroids eventually stabilizes.

**The elbow method** is a way of choosing k: you pick the value of k at which
there is a significant ("at the kink in the graph") reduction in the
within-cluster distance — further increasing k no longer yields a noticeable
gain.

**Limitation.** K-means works well only when the data distribution has
pronounced, roughly convex cluster centers with all objects concentrated around
them. For complex, non-convex geometry (for example, a ring of points
surrounding another cluster), k-means fails — no matter how many centroids you
add, the ring cannot be correctly separated, because by its very nature the
algorithm looks for compact, roughly convex clusters.

A drawback of k-means is that the random initialization of centers can lead to
different clustering results from run to run. An advantage is speed: on each
iteration only the distances to the cluster centers are recomputed, which is
cheap.

```python
from sklearn.cluster import KMeans

k_means = KMeans(n_clusters=3)
k_means = k_means.fit(X)
clusters = k_means.predict(X)

plt.scatter(X[:,0], X[:,1], c=clusters)
plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1], color='red', marker='x', s=200)
plt.title('K-means clustering')
plt.show()
```

## Hierarchical clustering

The idea: represent the data as a graph where the vertices are objects and the
edges are the distances between them. The result is visualized as a
**dendrogram** — a tree whose root is a single cluster uniting all objects, and
whose leaves are clusters consisting of a single object.

Two ways of building the hierarchy:
- **Agglomerative** (bottom-up) — on each iteration the algorithm merges the two
  nearest clusters into one; initially each object is a separate cluster (for a
  sample of size N there will be N clusters on the first iteration). The step is
  repeated until everything is merged into one common cluster.
- **Divisive** (top-down) — on each iteration one cluster is split into two
  smaller ones.

**The distance between clusters** can be computed in different ways (linkage):
- the average distance across all pairs of objects from two clusters
  (`average`);
- the minimum distance between objects of two clusters (`min` / `single`);
- the maximum distance between objects of two clusters (`max` / `complete`);
- **Ward's distance** — the distance between clusters is taken as the increase
  in the sum of squared distances of objects to the center of the cluster
  formed by their merger: the two clusters that produce the smallest such
  increase are merged (i.e., the ones most "similar" to each other by this
  criterion).

The final number of clusters is chosen from the dendrogram: by increasing the
distance threshold (the Y axis), you can watch objects gradually glue together
into larger clusters. A practical way to choose the threshold is to look for a
value at which a small shift of the threshold up or down does not cause a strong
rearrangement of the clusters (a stable picture), or to look for a sharp jump in
the dendrogram. If there are no specialized conditions (for example, it is known
that there should be no more than K clusters), the number of clusters can be
chosen precisely this way; it also helps to draw on the domain area of the task.

```python
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    dendrogram(linkage_matrix, **kwargs)


model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
model = model.fit(X)
plt.title("Hierarchical Clustering Dendrogram")
plot_dendrogram(model, truncate_mode="level", p=5)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()
```

## DBSCAN (Density-Based Spatial Clustering)

This develops the idea of clustering by identifying connected components based
on the density of the data rather than the distance to a centroid, so it
successfully handles complex (non-convex) geometry where k-means doesn't work.
An object's density is defined as the number of other sample points inside a
ball of radius epsilon (ε) around it.

**Hyperparameters:**
- **eps (ε)** — the radius of the neighborhood around a point;
- **min_samples** — the minimum number of points that must be inside this
  neighborhood (including the point itself) for the point to be considered a
  "core" point.

**Three types of points:**
- **Core points** — there are at least min_samples points within the eps
  neighborhood;
- **Border points** — there are fewer than min_samples points in the
  neighborhood, but the point itself is reachable from some core point;
- **Noise points / outliers** — neither core points nor reachable from any core
  point.

**Algorithm:**
1. Pick an unlabeled point.
2. If there are fewer than min_samples points in the eps neighborhood — mark it
   as noise.
3. Otherwise the point is a core point: find all points reachable from it
   (directly or through a chain of other core points) and merge them into a
   single cluster.
4. Border points are assigned to the cluster of the nearest core point to them.
5. The procedure repeats until all points have been processed.

The smaller the eps radius, the more numerous and smaller the resulting
clusters; the larger the eps, the larger and fewer the clusters.

**Pros:** it determines the number of clusters itself (modulo the given eps and
min_samples); it successfully handles complex cluster shapes; out of the box it
can find noise points (outliers) — k-means has no such capability at all.

**Cons:** longer runtime; sensitive to hyperparameter tuning; handles clusters
of varying density poorly.

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.2, min_samples=10)
clusters = dbscan.fit_predict(X)

plt.scatter(X[:,0], X[:,1], c=clusters)
plt.show()
```

**HDBSCAN** — a hierarchical modification of DBSCAN, for cases where ordinary
DBSCAN can't handle very complex data shapes, clusters of varying density, or
when hyperparameter tuning is especially difficult. It doesn't require an
explicit eps — it determines density automatically and builds a hierarchy,
selecting stable clusters by their stability; some points may be left without a
cluster if they are unstable (noise).

```python
import hdbscan

hdb = hdbscan.HDBSCAN(min_cluster_size=10)
hdb_labels = hdb.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=hdb_labels, cmap='Spectral', s=30)
plt.title("HDBSCAN (min_cluster_size=10)")
plt.show()

# Visualizing the confidence of cluster membership
plt.scatter(X[:, 0], X[:, 1], c=hdb.probabilities_, cmap='viridis', s=30)
plt.colorbar(label='Membership probability')
plt.title("HDBSCAN: clustering confidence")
plt.show()
```

## Spectral clustering (briefly)

Another approach based on a graph representation of the data (similar to the
idea of hierarchical clustering): objects are represented as a graph where the
edges reflect the distances between them, and clusters are identified based on
the distances in this graph — objects with a small distance are assigned to the
same cluster, while objects whose distance exceeds some threshold are assigned
to a different one. In practice it gives good results on complex, non-convex
data, on a par with DBSCAN.

## Graph-based method (minimum spanning tree)

The entire sample is represented as a complete graph whose vertices are objects
and whose edges carry the distance between them.

**Algorithm:**
1. Build a minimum spanning tree using Prim's or Kruskal's algorithm.
2. Based on the hyperparameter K (number of clusters), remove the K−1
   "heaviest" (longest) edges — as a result the graph breaks into K connected
   components, which become the clusters.

## Clustering quality metrics

Since clustering has no correct answers in the usual sense, quality assessment
is set up differently than in classification/regression — there is a family of
metrics of its own (analogous to the separate families of metrics for ranking
and for time series).

The metrics fall into two groups:
- **External metrics** use additional information — the true class labels of the
  objects (if they are in fact known) — comparing the resulting partition into
  clusters against them.
- **Internal metrics** assess the quality of the clustering on its own, without
  any external information about labeling.

### External metrics

- **Rand Index (RI)** — the proportion of object pairs for which the original
  partition into classes and the resulting partition into clusters agree (that
  is, either both times in the same cluster/class, or both times in different
  ones). Computed as the ratio of the number of "agreeing" pairs to all possible
  pairs of objects.
- **Adjusted Rand Index (ARI)** — a normalized version of RI that does not
  depend on the number of objects N or the number of clusters: a value of 1
  means the partitions coincide, greater than 0 means the partitions are
  similar, around 0 means the partitions are random and unrelated to each other,
  less than 0 means the partitions are dissimilar (rare in practice).
- **Homogeneity** — the extent to which each cluster consists of objects of a
  single class. The worst case: the distribution over clusters did not reduce
  the uncertainty (entropy) about the original classes at all; the best: each
  cluster contains objects of only one class. Trivially, the "best" homogeneity
  can be obtained by putting each object in its own separate cluster (which is
  why homogeneity cannot be used in isolation from the other metrics).
- **Completeness** — the extent to which objects of a single class could be
  gathered entirely into one cluster (the flip side of homogeneity). Trivially,
  the "best" completeness can be obtained by placing all objects into one common
  cluster.
- **V-measure** — the harmonic mean of homogeneity and completeness (analogous
  to the F-measure from classification, where homogeneity and completeness are
  the analogs of precision and recall). Homogeneity, completeness, and V-measure
  take values from 0 to 1, but are not normalized and depend on the number of
  clusters: with a large number of clusters and a small number of objects it's
  better to use ARI; with fewer than 10 clusters and more than 1000 objects this
  dependence is less pronounced, and the metrics can be used without concern.

### Internal metrics

- **Average within-cluster distance** — the sum of distances across all pairs of
  objects within the same cluster, divided by the number of such pairs. The
  denser (more compact) the clusters, the smaller this quantity — it is
  minimized.
- **Average between-cluster distance** — similar, but computed for pairs of
  objects from different clusters. This quantity is maximized — the more distant
  the clusters are from each other, the better.
- **Dunn Index** — the ratio of the minimum between-cluster distance (over all
  pairs of clusters) to the maximum within-cluster distance (over all clusters).
  It accounts for both the compactness of the clusters and their separation at
  once; useful when the number of clusters is already known.
- **Silhouette score** — does not require knowing the true labels. For each
  object, two quantities are computed: a — the average distance to all objects
  of the same cluster; b — the average distance to the objects of the nearest
  other cluster. An object's silhouette shows how large the relative difference
  between b and a is (normalized so that the value lies in the interval from −1
  to 1). The silhouette of the sample is the average silhouette over all
  objects: around 1 — clearly defined, well-separated clusters; around 0 —
  overlapping clusters; around −1 — poor, scattered clustering. The silhouette
  reaches its highest values on compact, convex-shaped clusters and may "fib" a
  bit for non-convex shapes — so, in the absence of labeling, it's better to
  rely on several metrics at once.

**How to choose a metric in practice:**
- if the true labels are known — better to use V-measure or ARI;
- if the number of clusters is known — the Dunn Index works well;
- if nothing at all is known — the silhouette works decently;
- if it's possible to collect labeling in sufficient volume — it may be worth
  solving a classification task altogether rather than a clustering one.

**Practical use of metrics to select k.** Since choosing the number of clusters
remains a problem for almost all algorithms, you can iterate over different
values of k, compute the metric value (for example, the silhouette) for each,
and choose the k at which the metric is maximal — a systematic approach instead
of eyeballing it.

```python
from sklearn.metrics import silhouette_score, homogeneity_score, v_measure_score

silhouette = []
homogeneity = []
v_measure = []
for n_c in range(2, 8):
    k_means = KMeans(n_clusters=n_c)
    k_means = k_means.fit(X)
    clusters = k_means.predict(X)
    silhouette.append(silhouette_score(X, clusters))
    homogeneity.append(homogeneity_score(y, clusters))
    v_measure.append(v_measure_score(y, clusters))

plt.figure(figsize=(11, 8))
plt.plot(range(2,8), silhouette, label='silhouette')
plt.plot(range(2,8), homogeneity, label='homogeneity')
plt.plot(range(2,8), v_measure, label='v_measure')
plt.xlabel('n_clusters')
plt.ylabel('metric')
plt.legend(loc='best')
plt.show()
```

## Comparing methods on different data

Different clustering algorithms handle different data geometries differently
(circles, crescents, clusters of varying density, clusters of varying size, no
structure at all). The general recommendation is not to rely on a single method
but to compare several algorithms (K-means, Agglomerative Clustering, DBSCAN,
HDBSCAN) on your own data, using internal metrics (the silhouette) where there
is no labeling:

```python
from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler

n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)

X, y = noisy_moons
X = StandardScaler().fit_transform(X)

k_means = cluster.KMeans(n_clusters=2)
dbscan = cluster.DBSCAN(eps=.3)
average_linkage = cluster.AgglomerativeClustering(linkage="average", n_clusters=2)

for name, algorithm in (('KMeans', k_means), ('DBSCAN', dbscan), ('Agglomerative', average_linkage)):
    algorithm.fit(X)
    y_pred = algorithm.labels_.astype(int)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.title(name)
    plt.show()
```

In summary, the clustering algorithms covered: k-means, hierarchical
(agglomerative) clustering, DBSCAN and HDBSCAN, spectral clustering, and the
graph-based method on a minimum spanning tree. All of the metrics covered
(RI/ARI, homogeneity/completeness/V-measure, silhouette, Dunn Index) are
applicable to any of these approaches and can be used for hyperparameter tuning
(for example, by maximizing the silhouette).

## Visualization of high-dimensional data (MDS, t-SNE)

A related task — not clustering itself, but the **visualization** of a
high-dimensional feature space, when you can only draw in one, two, or three
dimensions but want the picture to match the structure of the original
high-dimensional space as closely as possible.

**Multidimensional scaling (MDS).** The distance between objects in the original
space and the distance between their projections in the new (for example,
two-dimensional) space are computed. The coordinates of the projections are
chosen so that the distances between the projections are as similar as possible
to the original distances — that is, to preserve the proportions of the original
distances. A limitation of the method: if you try to compress objects from a
very high-dimensional space into a two-dimensional one, preserving all pairwise
distances is often simply impossible.

**t-SNE.** Since MDS cannot always preserve the original distances under
projection, t-SNE relaxes the requirement: instead of preserving the distances
themselves, the method tries to preserve the proportions of proximity — distant
objects should stay far apart, close ones should stay close. The algorithm
builds a probability distribution of point proximity in the original space (the
neighborhood of each point is assumed to be roughly normally distributed) and
chooses an arrangement of points in the target (2-3-dimensional) space such that
the analogous proximity distribution in the new space is as similar as possible
to the original — the similarity of the two distributions is measured via the
Kullback-Leibler divergence (KL divergence). For the target space a distribution
with "heavier tails" (Cauchy) is used, which does not penalize an increase in
the distance between objects as strongly as the normal distribution does.

On the MNIST dataset (handwritten digits) t-SNE visualizes similar objects
noticeably better than PCA: with PCA the digit classes get mixed and overlap
each other, whereas t-SNE better spreads the classes apart (although certain
overlaps remain there too). If your task is to visualize a high-dimensional
object, t-SNE is usually a good choice, though on complex high-dimensional
structures it can be quite slow.

Clustering and dimensionality-reduction/visualization methods (PCA, MDS, t-SNE)
most often serve as preprocessing or an auxiliary subtask for a larger task, but
they are also used as standalone tools.
