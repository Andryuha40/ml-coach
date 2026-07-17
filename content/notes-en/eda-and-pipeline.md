# Exploratory data analysis, sklearn Pipeline, feature encoding

These notes were assembled from the course lecture "The Pipeline class" (the
structure and motivation of Pipeline is almost word-for-word identical in
content to the seminar `07_Класс_Pipeline.ipynb`, so the code was taken from
the seminar as the more complete and working version), the course seminar
"Exploratory data analysis" (EDA on a telecom customer churn dataset — the
main source for EDA technique), the repository seminar `lesson_01` (Sweetviz
— a short addition on auto-EDA libraries), the repository seminar `lesson_04`
(regularization + exploratory analysis on housing-price data — the source of
the examples on data leakage and alpha tuning) and the hint notebook
`lesson_04` ("Feature construction" — the source on categorical feature
encoding, scaling and generating features from dates).

## Why exploratory data analysis (EDA) is needed

Before training any model, it is important to understand well the data you
are going to work with. Exploratory Data Analysis (EDA) helps to:
- understand the structure and content of the data;
- spot possible problems (missing values, duplicates, outliers);
- determine the main characteristics of the variables and their
  distributions;
- determine the relationships between variables and with the target variable.

## First look at the table

A basic set of actions on a new dataset (example — a telecom customer churn
dataset, the task being to predict whether a customer will leave):

```python
df.head(5)          # first rows, a quick glance at the structure
df.info()           # column types, number of non-null values
df.shape             # number of rows and columns
df.isnull().sum()   # checking for missing values in each column
df.describe().T                    # statistics for numeric columns
df.describe(include='object').T    # statistics for categorical columns
```

### Missing values

What to do with missing values depends on their proportion and nature:
- if there are few rows with missing values (a small percentage of all the
  data) — you can drop those rows;
- if a particular column has a very large number of missing values — it may
  be worth dropping the whole column;
- fill missing values with a statistic (the median or the mean);
- there are also more advanced approaches — predicting the missing value
  with a machine learning model based on the other features.

**Hidden missing values.** A missing value does not always look like `NaN`.
In the customer churn example, the column `TotalCharges` is numeric by
meaning but is stored as a string (`object`) — on inspection it turned out
that some of the values are just a space `' '`, which is why pandas could not
automatically recognize the column as numeric:

```python
df[df['TotalCharges'] == ' ']   # find the "hidden" missing values masked as a space

def to_float(x):
    try:
        return float(x)
    except:
        return 0

df['TotalCharges_num'] = df['TotalCharges'].apply(to_float)
```

Further analysis showed that `TotalCharges` is actually equal to
`MonthlyCharges * tenure` (the number of months of service), so the missing
values in it could be correctly replaced with zero (corresponding to new
customers with tenure = 0). This also showed that `TotalCharges` is a linear
combination of two other features and adds no new information; this
conclusion will come in handy later, when building models
(multicollinearity).

### Outliers

One of the standard ways to detect outliers in a numeric feature is the
interquartile range (IQR):

```python
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = data[(data < lower_bound) | (data > upper_bound)]
```

### Feature distributions

For numeric features — histograms (`sns.histplot`), including a split by the
target variable, so you can see whether the feature's distribution differs
between classes; and boxplots — for a visual comparison of the median,
spread and outliers between groups:

```python
for var in num_columns:
    sns.histplot(df, x=var, hue='Churn', bins=30)
    plt.title(f'Distribution of {var}')
    plt.show()

for var in num_columns:
    sns.boxplot(df, x=var, hue='Churn')
    plt.show()
```

For categorical/binary features — a countplot split by the target variable:

```python
for var in categorical_variables:
    sns.countplot(data=df, x=var, hue='Churn')
    plt.xticks(rotation=90)
    plt.show()
```

A universal function for going through each column one at a time (numeric or
categorical) — it prints statistics and builds a histogram + boxplot (for
numeric ones) or the category frequency distribution (for categorical ones):

```python
def column_eda(df, column, dtype='numeric', figsize=(15, 5)):
    data = df[column]
    print(f"Column: {column}")
    print(f"Missing: {data.isna().sum()}")

    if dtype == 'numeric':
        print(f"Min: {data.min()}, Max: {data.max()}, Mean: {data.mean()}")
        print(f"Median: {data.median()}, Std: {data.std()}")
        Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = data[(data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)]
        print(f"Outliers: {len(outliers)}")

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].hist(data.dropna(), bins=min(50, len(data.unique())), density=True)
        data.dropna().plot(kind='kde', ax=axes[0], color='red')
        axes[1].boxplot(data.dropna(), vert=True)
        plt.show()

    elif dtype == 'categorical':
        value_counts = data.value_counts()
        print(f"Unique values: {data.nunique()}")
        # ... printing the top values and bar charts of frequencies/proportions
```

### Relationships between variables

**Pearson correlation** — a measure of the linear relationship between two
numeric columns, the most common one ("numeric-numeric"):

```python
correlation_matrix = df[num_columns].corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
```

For a fuller picture across different types of feature pairs:

- **Spearman correlation** (`df.corr(method='spearman')`) — based on ranking
  the values, it measures the degree of a **monotonic** (not necessarily
  linear) relationship; it is also suitable for ordinal variables.
- **Kendall correlation** (`df.corr(method='kendall')`) — similar to
  Spearman, more often used for a nominal-nominal feature pair.
- **Cramér's V** ("categorical-categorical") — a normalization of the
  chi-square statistic (χ²) by the number of categories, bringing the result
  to the interval [0, 1] for easy interpretation. The chi-square statistic is
  computed from the contingency table: χ² = sum over all cells of (O − E)² /
  E, where O is the observed frequency and E is the expected frequency under
  independence of the features. The larger χ² is, the stronger the
  relationship.
- **ANOVA** ("numeric-categorical") — compares the mean values of the
  numeric feature across three or more groups defined by the categorical
  feature; it produces an F-statistic and a p-value (`scipy.stats.f_oneway`).
  If the p-value < 0.05, the differences between groups are considered
  statistically significant (the features are more likely related).
- **φₖ (phik)** — a correlation coefficient uniformly applicable to all
  feature types: for categorical ones it is essentially the same χ², and for
  numeric ones the values are first binarized (split into intervals), and
  then χ² is computed as well (the `phik` library).

**An important caveat: correlation is not the same as a cause-and-effect
relationship** ("correlation is not causation"). Example: ice cream sales and
the number of shark attacks correlate by month, but ice cream sales do not
affect sharks — both phenomena depend on a third factor (hot weather).
Similarly, a positive correlation between a person's height and weight does
not mean that one parameter directly affects the other — both depend, for
example, on age.

## Automated EDA (AutoEDA libraries)

There are libraries that build a detailed HTML report on a dataset
(distributions, missing values, correlations, warnings about problems in the
data) in literally a single line of code:

```python
# YData Profiling
from ydata_profiling import ProfileReport
profile = ProfileReport(df, title="Churn dataset Report", explorative=True)
profile.to_notebook_iframe()
profile.to_file("my_report.html")

# Sweetviz
import sweetviz as sv
report = sv.analyze(df)
report.show_html('sweetviz_report.html')
```

Such tools are handy for a quick first look at a new dataset, but they do not
replace careful manual analysis (for example, they will not "catch" a hidden
missing value masked as a space, as in the example above, if it is not
formally recognized by the library as NaN).

## Why Pipeline is needed

### The problem: code without Pipeline

The standard sequence of actions before training a model is to split the
features by type, apply different preprocessing to them (scaling numeric
ones, encoding categorical ones), combine the result and train the model:

```python
numerical_features = ['Age', 'Height', 'Weight']
to_dummies = ['Gender', 'Left - right handed', 'Village - town', 'Smoking', 'Alcohol']

scaler = StandardScaler()
data_train_scaled = scaler.fit_transform(data_train[numerical_features])

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
data_train_ohe = ohe.fit_transform(data_train[to_dummies])

data_train_transformed = pd.concat([
    pd.DataFrame(data_train_scaled, columns=numerical_features),
    pd.DataFrame(data_train_ohe, columns=ohe.get_feature_names_out()),
], axis=1)

model = LogisticRegression(class_weight='balanced')
model.fit(data_train_transformed, data_train[target].values.ravel())
```

The problem is that for the validation/test set you will have to **manually
repeat** those same transformation steps (and it is important to call
`transform`, not `fit_transform` — see the note on data leakage below):

```python
data_test_scaled = scaler.transform(data_test[numerical_features])
data_test_ohe = ohe.transform(data_test[to_dummies])
data_test_transformed = pd.concat([...], axis=1)
```

**The upshot of this approach**: code duplication, hard to scale to new
features, hard to maintain, takes a lot of time, and — most importantly — it
is unclear how to save this whole sequence of steps as a single object for
reuse in production.

### The solution: Pipeline

`Pipeline` — a class from `sklearn.pipeline` that combines several data
processing steps and the model itself into a single chain. Capabilities:
- combines several actions into one object;
- removes code duplication (the transformation steps are written once and
  automatically applied identically to train and test — `fit` on train,
  `transform` on train and test);
- with a pipeline you can work as with a single sklearn object, calling the
  standard `.fit()` and `.predict()`;
- a pipeline can be saved in its entirety (for example, via `pickle`) for
  later reuse in production.

Key helper classes:
- **`ColumnTransformer`** (from `sklearn.compose`) — applies different
  transformations to different subsets of columns;
- **`SimpleImputer`** — filling in missing values;
- **`SelectKBest`** — selecting the k best features by a statistical
  criterion;
- **`FunctionTransformer`** — turns an arbitrary function into a transformer;
- **`FeatureUnion`** — combines the results of several transformers into one
  dataset;
- **`make_pipeline`** — a convenient wrapper for creating a pipeline without
  having to come up with step names manually.

**When to apply Pipeline**: not right away — first the EDA stage should be
finished and an understanding of the final model structure should emerge
(which features to use, what kind of preprocessing is needed). Pipeline is
applied at the stage of transitioning from EDA to training/fine-tuning the
model, stabilizing it and preparing the model for moving it into production.

### Assembling a Pipeline with ColumnTransformer

```python
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

# for numeric features — fill missing values, then scale, then select features
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer()),
    ("scaler", StandardScaler()),
    ("fs", SelectKBest(score_func=f_classif, k="all")),
])

# for categorical features — fill missing values, then OneHotEncoder
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer()),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

# combine the transformers for the different column types
data_transformer = ColumnTransformer(transformers=[
    ("numerical", numerical_transformer, numerical_features),
    ("categorical", categorical_transformer, categorical_features),
])

preprocessor = Pipeline(steps=[("data_transformer", data_transformer)])

classifier_pipline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression())
])

classifier_pipline.fit(data_train[all_features], data_train[target].values.ravel())
preds = classifier_pipline.predict(data_test[all_features])
```

### Tuning hyperparameters of the whole Pipeline

The hyperparameters of any pipeline step are accessible via a double
underscore `<step_name>__<parameter>` (including nested ones — you can reach
a hyperparameter of a transformer inside a `ColumnTransformer` inside a
`Pipeline`):

```python
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer

f1 = make_scorer(f1_score, average="macro")

param_grid = {
    'classifier__C': np.logspace(-5, 2, 200),
    'classifier__penalty': ['l1', 'l2'],
    'classifier__solver': ['liblinear'],
    'classifier__class_weight': ['balanced', None],
    'preprocessor__data_transformer__numerical__imputer__strategy': ['median', 'mean'],
}

search = GridSearchCV(classifier_pipline, param_grid, n_jobs=1, cv=3, scoring=f1)
search.fit(data_train.drop(target, axis=1), data_train[target].values.ravel())

search.best_params_
search.best_score_
```

### Pipeline with several models: voting and stacking

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
import xgboost

# Voting (blending) of several models from different families
clf1 = LogisticRegression(multi_class="multinomial", random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()

blending_classifier = VotingClassifier(estimators=[
    ("log_regression", clf1), ("random_forest", clf2), ("gnb", clf3)
])

classifier_pipline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", blending_classifier)
])

# Stacking: each base model is its own separate pipeline (possibly with different preprocessing)
estimators = [
    ("SVM", make_pipeline(preprocessor, PCA(), LinearSVC())),
    ("Random_Forest", make_pipeline(preprocessor, RandomForestClassifier())),
    ("Xgboost", make_pipeline(preprocessor, xgboost.XGBClassifier())),
]

stacking_classifier = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(n_jobs=-1, verbose=True),
    n_jobs=-1,
    verbose=True,
)
```

The flexibility of Pipeline lets you reach any nested parameter directly
along the chain of attributes (useful for debugging the assembled
composition):

```python
stacking_classifier.estimators[0][1][0].steps[0][1].transformers[0][1].steps[2][1].k
```

### Saving and loading the whole Pipeline

```python
import pickle

with open("awesome_pipeline.pkl", "wb") as f:
    pickle.dump(stacking_classifier, f)

with open("awesome_pipeline.pkl", 'rb') as f:
    pipeline_from_saved = pickle.load(f)
```

## Data leakage from incorrect preprocessing

**The key rule**: any statistics used for preprocessing (the mean and
standard deviation for scaling, the mean target values for target encoding,
regularization coefficients, etc.) must be computed **only on the training
set**, and then this same, already-computed transformation is applied to both
train and test/valid — without recomputing on the new data.

In practice this shows up as the difference between the `fit_transform`
methods (for train) and `transform` (for test):

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # computes the mean/std and applies immediately
X_test_scaled = scaler.transform(X_test)          # applies the same transformation, WITHOUT recomputing — not fit_transform!
```

If you mistakenly call `fit_transform` on the test set (or, worse, combine
train and test before preprocessing), the test statistics will "leak" into
the preprocessing — the model ends up tuned taking into account information
it should not have had at prediction time. This is exactly what is called
**data leakage**: the quality estimate of the model on such "leaked" data
comes out artificially inflated and does not reflect the real quality of the
model on genuinely new data.

An analogous principle applies when tuning hyperparameters: the
regularization coefficient (`alpha` in Ridge/Lasso) must not be tuned either
on the training set (the model will simply pick the smallest possible
regularization and overfit) or on the test set (the test set will cease to be
an "honest" quality estimate — it will imperceptibly "leak" into the training
process). The correct way is to tune on a validation set or via
cross-validation, while the test set stays untouched until the very end:

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

alphas = np.logspace(-2, 3, 20)   # a sweep over a logarithmic grid — to feel out the order of magnitude
searcher = GridSearchCV(Ridge(), [{"alpha": alphas}],
                         scoring="neg_root_mean_squared_error", cv=10)
searcher.fit(X_train_scaled, y_train)
best_alpha = searcher.best_params_["alpha"]
```

Another characteristic source of leakage is identifier features (`id`) or
features directly computed from other features in one and the same
deterministic way: for example, the `TotalCharges` column in the customer
churn example turned out to equal `MonthlyCharges × tenure` — such a hidden
linear dependency is not in itself a "leak" in the strict sense (it is not
information from the future), but it just as reliably leads to overfitting and
uninterpretable weights of the linear model if it is not detected at the EDA
stage.

### Cross-validation as a more honest quality estimate

Cross-validation is a way to estimate the quality of a model more reliably
than on a single random train/test split: the training set is divided into n
parts (folds), n models are trained — each on all folds except one (the held
out, "out-of-fold" one), and each is evaluated on that held-out fold. The
final metric is the average over the n resulting values:

```python
from sklearn.model_selection import cross_val_score

model = LinearRegression()
cv_scores = cross_val_score(model, X_train[numeric_features], y_train,
                             cv=10, scoring="neg_root_mean_squared_error")
print("Mean CV RMSE = %.4f" % np.mean(-cv_scores))
```

Note the sign: standard sklearn scorers for metrics that need to be minimized
(for example, RMSE) are named `neg_*`, because sklearn's internal mechanism
always **maximizes** the scoring function passed to it — which is why the
result comes back with a minus sign.

## Categorical feature encoding

Linear models cannot work with text categories directly — they need to be
encoded as numbers. There are many ways to do it, and the choice depends on
the type of categorical feature (nominal/ordinal) and the number of unique
values:

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, TargetEncoder

df = pd.DataFrame({
    'район': ['Центр', 'Север', 'Юг', 'Север', 'Центр'],
    'тип_дома': ['Кирпич', 'Панель', 'Монолит', 'Кирпич', 'Панель'],
    'состояние': ['Отличное', 'Требует ремонта', 'Хорошее', 'Удовлетворительное', 'Хорошее'],
    'цена_млн': [12.5, 7.2, 9.8, 8.1, 10.3]
})

# 1. Label Encoding — each category is assigned a number 0, 1, 2, ...
# Dangerous for nominal features: the model may decide that "2 > 1", even though the categories are not ordered
le = LabelEncoder()
df['район'] = le.fit_transform(df['район'])

# 2. One-Hot Encoding — a separate binary column for each category
df_ohe = pd.get_dummies(df, columns=['район'])
# The same via sklearn:
encoder_ohe = OneHotEncoder(sparse_output=False)
encoded = encoder_ohe.fit_transform(df[['район']])

# One-Hot with drop_first=True — removes one column (avoids redundancy/multicollinearity
# when used with linear models, for which the full set of dummy variables is redundant)
df_ohe_drop = pd.get_dummies(df, columns=['район'], drop_first=True)

# 3. Ordinal Encoding — for ORDINAL features, where the order of categories has meaning
encoder_ord = OrdinalEncoder(
    categories=[['Требует ремонта', 'Удовлетворительное', 'Хорошее', 'Отличное']]
)
df['состояние'] = encoder_ord.fit_transform(df[['состояние']])

# 4. Target (Mean) Encoding — a category is replaced by the mean target value for that category
# cv and smooth are needed to reduce overfitting/leakage of the target variable
encoder_target = TargetEncoder(cv=3, smooth=0.2)
df['район'] = encoder_target.fit_transform(df[['район']], df['цена_млн'])

# 5. Binary Encoding and Leave-One-Out Encoding — from the separate category_encoders library,
# a compromise between the compactness of One-Hot and the informativeness of Target Encoding
import category_encoders as ce
encoder_bin = ce.BinaryEncoder(cols=['район', 'тип_дома'])
encoder_loo = ce.LeaveOneOutEncoder(cols=['район', 'тип_дома'])
```

**A practical rule for the choice**: One-Hot works well with a small number
of unique categories (otherwise the dataset gets heavily inflated in width and
becomes sparse); with a large number of categories, mean/target encoding or
counters are preferable. They are often combined: if a feature has fewer
categories than a threshold (for example, 5) — one-hot, if more — target
encoding.

## Feature scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# StandardScaler — subtract the mean, divide by the standard deviation (when the data resembles a normal distribution)
scaler_std = StandardScaler()

# MinMaxScaler — squeeze the values into the range [0, 1] (when the exact range boundaries matter)
scaler_mm = MinMaxScaler()

# RobustScaler — uses the median and the interquartile range instead of the mean and std,
# so it is more robust to outliers than StandardScaler
scaler_rob = RobustScaler()

X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit only on the train set
X_test_scaled = scaler.transform(X_test)          # transform — on both (not fit_transform!)
```

## Other feature engineering techniques

**Log transformation** — smooths heavily skewed distributions (for example,
prices, incomes), reduces the influence of large outliers:

```python
df['цена_log'] = np.log(df['цена_млн'])
df['цена_log+1'] = np.log1p(df['цена_млн'])   # log(x + 1) — works even when x = 0
```

**Polynomial features** — allow a linear model to capture a nonlinear
dependency by adding, as new features, powers and pairwise products of the
original ones:

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(df[['площадь']])
```

**Features from dates** — unpacking a single date column into a set of
meaningful numeric/categorical features:

```python
df_dates['год'] = df_dates['дата_заявки'].dt.year
df_dates['месяц'] = df_dates['дата_заявки'].dt.month
df_dates['день_недели'] = df_dates['дата_заявки'].dt.dayofweek  # 0 = Monday
df_dates['квартал'] = df_dates['дата_заявки'].dt.quarter
df_dates['номер_недели'] = df_dates['дата_заявки'].dt.isocalendar().week

df_dates['выходной'] = df_dates['день_недели'].isin([5, 6]).astype(int)

import holidays
ru_holidays = holidays.Russia()
df_dates['праздник'] = df_dates['дата_заявки'].apply(lambda x: int(x in ru_holidays))

df_dates['сезон'] = df_dates['месяц'].map({12: 'зима', 1: 'зима', 2: 'зима', ...})

# The difference between two dates is also a useful feature
df_dates['дней_до_заезда'] = (df_dates['дата_заезда'] - df_dates['дата_заявки']).dt.days
```

## Overfitting and regularization: a reminder on a numeric example

A demonstration of overfitting on synthetic data: the true dependency is a
cosine, and three polynomial models of different degrees (1, 4 and 20) are
trained on 30 noisy points:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

for degree in [1, 4, 20]:
    X_objects = PolynomialFeatures(degree, include_bias=False).fit_transform(x_objects[:, None])
    regr = LinearRegression().fit(X_objects, y_objects)
    y_pred = regr.predict(X)
    # degree=1: the model is too simple, it cannot capture the nonlinear dependency (underfitting)
    # degree=20: the model passes perfectly through all 30 points, including the noise (overfitting) —
    #            and gets very large weights in absolute value
```

Overfitting of linear models is usually connected precisely with large
weights — hence the idea of regularization: to penalize the loss function for
large weight values (a detailed breakdown of L1/L2/ElasticNet is in the topic
`linear-regression-gd`). What matters here is the connecting link:
**identifier features** (for example, a row's `id`) during training almost
guarantee overfitting — the model can "latch onto" them as a feature that
randomly correlates with the target, so such columns are usually removed at
the EDA/preprocessing stage, even before Pipeline.

## Final housing-price prediction pipeline (a full-cycle example)

A final example combining exploratory analysis, categorical feature encoding,
scaling and regularized linear regression into a single `Pipeline`:

```python
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

categorical = list(X_train.dtypes[X_train.dtypes == "object"].index)
X_train[categorical] = X_train[categorical].fillna("NotGiven")  # a missing value as a separate category
X_test[categorical] = X_test[categorical].fillna("NotGiven")

column_transformer = ColumnTransformer([
    ('ohe', OneHotEncoder(handle_unknown="ignore"), categorical),
    ('scaling', StandardScaler(), numeric_features)
])

lasso_pipeline = Pipeline(steps=[
    ('ohe_and_scaling', column_transformer),
    ('regression', Lasso())
])

model = lasso_pipeline.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("RMSE = %.4f" % root_mean_squared_error(y_test, y_pred))

# sweeping alpha directly through the Pipeline parameters (note the regression__alpha syntax)
alphas = np.logspace(-2, 4, 20)
searcher = GridSearchCV(lasso_pipeline, [{"regression__alpha": alphas}],
                         scoring="neg_root_mean_squared_error", cv=10, n_jobs=-1)
searcher.fit(X_train, y_train)
```

A useful side effect of L1 regularization (Lasso) shown here in practice:
some of the Lasso weights are zeroed out completely (unlike Ridge), which you
can check and use as built-in feature selection:

```python
lasso_zeros = np.sum(pipeline.steps[-1][-1].coef_ == 0)
print("Zero weights in Lasso:", lasso_zeros)
```
