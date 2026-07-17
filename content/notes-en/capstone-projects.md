# Capstone projects: application to real-world tasks

An optional topic (`optional: true`). These notes were assembled from two
ready-made bonus notebooks in Evgeny Patochenko's repository
(`content/ml_basics_course/bonus/Прогнозирование оттока клиентов.ipynb` and
`content/ml_basics_course/bonus/Предсказание отклика клиента.ipynb`). These are
solved reference cases — their purpose in the course is not to show "correct code"
to copy, but to give a realistic business framing of the task, a description of
the available features, and the general structure of the pipeline. The final code
for comparing models and tuning hyperparameters is deliberately not reproduced
verbatim: the capstone assumes that you go the whole way yourself, from raw data
to a trained model, relying on what has already been worked through in the
previous topics of the course (`eda-and-pipeline`, `classification-metrics`,
`decision-trees`, `ensembles-bagging-boosting`).

## The general idea of the capstone

Both cases are complete business tasks of binary classification on tabular data:
each object (customer) has a set of features and a binary target variable. The
capstone task is to go the whole way yourself:

1. Loading and an initial inspection of the data.
2. EDA: missing values, outliers, feature distributions, the difference in
   distributions between the classes of the target variable.
3. Handling problems in the data (missing values, anomalous categories,
   outliers).
4. Encoding categorical features.
5. A non-ML baseline (a simple rule) and an ML baseline (for example, logistic
   regression without scaling) — so that there is something to compare the more
   complex models against.
6. Scaling features where it matters for the model (linear models, KNN, SVM).
7. Training and comparing several models from different families (linear,
   distance-based, trees, ensembles, boosting).
8. Choosing a quality metric that matches this specific business task, and
   selecting the best model by it (rather than by a "default" metric).
9. Interpreting the model: which features turned out to be important and whether
   this agrees with common sense / business logic.

This is exactly the ML-task-solving process discussed in the topic `intro-ai-ml`
(obtaining the data → EDA → feature selection → model choice and training →
quality assessment → deployment), only applied in its entirety, end to end, on a
real dataset.

## Case 1: predicting bank customer churn

**Data source:** a dataset from Kaggle
(`https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers`) — data on
bank customers and their cards.

**Business framing.** It is important for the bank to identify in advance the
customers who are highly likely to give up service (leave — "churn",
"attrition"), so as to have the chance to retain them (for example, by offering
better terms) before they close their account. The target variable is
`Attrition_Flag` (in the source data: `Existing Customer` / `Attrited Customer`,
in the task it is re-encoded into 0/1, where 1 means the customer left).

**Available features** (after removing service and "leaking" fields — see the
section on data leakage below for more detail):

- `Customer_Age` — the customer's age.
- `Gender` — gender.
- `Dependent_count` — number of dependents.
- `Education_Level` — level of education.
- `Marital_Status` — marital status.
- `Income_Category` — annual income category.
- `Card_Category` — card type (Blue, Silver, Gold, Platinum).
- `Months_on_book` — length of the relationship with the bank (in months).
- `Total_Relationship_Count` — the total number of the bank's products held by the
  customer.
- `Months_Inactive_12_mon` — the number of months without activity over the last
  12 months.
- `Contacts_Count_12_mon` — the number of contacts with the bank over the last 12
  months.
- `Credit_Limit` — the credit limit.
- `Total_Revolving_Bal` — the total revolving balance (the amount of debt carried
  over to the next month with interest charged; a high balance may indicate active
  use of credit).
- `Avg_Open_To_Buy` — the average available credit (the difference between
  `Credit_Limit` and `Total_Revolving_Bal`; shows how much the customer can still
  spend — an indicator of financial flexibility).
- `Total_Amt_Chng_Q4_Q1` — the change in the transaction amount (Q4 relative to
  Q1).
- `Total_Trans_Amt`, `Total_Trans_Ct` — the total amount and number of
  transactions over 12 months.
- `Total_Ct_Chng_Q4_Q1` — the change in the number of transactions (Q4 to Q1).
- `Avg_Utilization_Ratio` — the average card utilization ratio
  (`Total_Revolving_Bal` / `Credit_Limit`; an important indicator for credit
  scoring — a high ratio, close to 1, may signal financial strain).

**An important nuance with the data (leakage / junk features).** The source
dataset contains two columns with long names of the form
`Naive_Bayes_Classifier_Attrition_Flag_...` — this is the output of someone
else's naive Bayes classifier, accidentally left in the file by the authors of
the Kaggle dataset. Such columns must be removed before training — otherwise the
model would effectively be trained on someone else's ready-made prediction of the
target variable, which would give deceptively high quality and is a classic
example of data leakage (see the topic `eda-and-pipeline` for more detail). It is
also worth removing the customer identifier (`CLIENTNUM`) — it carries no feature
meaning.

**Data specifics worth taking into account during EDA:**
- There are no missing values in the data.
- The classes of the target variable are imbalanced: roughly 8500 "stayed" and
  1627 "left" (the churn share is about 16%) — this should be taken into account
  when choosing a metric and, possibly, during training (see the topic
  `class-imbalance`).
- The categorical features (`Education_Level`, `Marital_Status`,
  `Income_Category`, `Card_Category`, `Gender`) require encoding (for example,
  one-hot via `pd.get_dummies`) before being fed into most models.
- It is useful to plot the distributions of the numeric features separately for
  the two classes of the target variable (`Attrition_Flag` = 0 and = 1) — this
  often immediately suggests which features are potentially significant.
- For linear models, scaling the features (for example, `StandardScaler`) is
  important — without it, logistic regression performs noticeably worse.

**What to pay attention to when choosing a metric.** Since the classes are
imbalanced and, from a business standpoint, it is much more costly to "miss" a
customer who will actually leave (false negative) than to mistakenly offer
retention to someone who would have stayed anyway (false positive), accuracy alone
is not enough — it can be deceptively high even for a useless model (for example,
if the model always predicts "stayed"). It is worth focusing on recall for the
churn class, precision, F1, and ROC-AUC — and be sure to compare against a non-ML
baseline (for example, a simple rule based on the number of months of inactivity)
to understand what real gain the model provides.

## Case 2: predicting a customer's response to a marketing campaign

**Data source:** a dataset from Kaggle
(`https://www.kaggle.com/datasets/jackdaoud/marketing-data/data`), loaded directly
by link (`https://github.com/evgpat/datasets/raw/main/ifood_df_raw.csv`).

**Business framing.** A company runs marketing campaigns (offers) for its
customers. The task is to predict whether a particular customer will respond to
the next (latest) campaign, so as to target offers at those most likely to respond
and not spend budget contacting customers who will definitely decline. The target
variable is `Response` (1 — the customer responded to the latest campaign, 0 —
not).

**Available features:**

- `Year_Birth` — year of birth (in the dataset it is converted into `Age`).
- `Education` — level of education.
- `Marital_Status` — marital status (in the raw data there are "junk" categories
  like `Absurd`, `YOLO`, `Alone` — it is reasonable to merge them with the nearest
  category in meaning, for example `Single`).
- `Income` — household income (contains missing values and an outlier — there is a
  customer with an income of 666666, which is worth either removing or clipping by
  a quantile).
- `Kidhome`, `Teenhome` — the number of small children and teenagers.
- `Dt_Customer` — the date from which the person became a customer of the company
  (from it you can construct a feature "number of days since the relationship
  began").
- `Recency` — the number of days since the last purchase.
- Spending on specific product categories: `MntWines`, `MntFruits`,
  `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`.
- `NumDealsPurchases` — the number of purchases made with a discount.
- `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases` — the number of
  purchases made online, from the catalog, and in physical stores.
- `NumWebVisitsMonth` — the number of website visits per month.
- `AcceptedCmp1`–`AcceptedCmp5` — whether the customer accepted the offer of each
  of the five previous campaigns (1/0) — potentially very informative features,
  since past behavior usually predicts the future well.
- `Complain` — whether the customer complained in the last two years.
- `Z_CostContact`, `Z_Revenue` — constant technical columns (the same value for
  every row) — they carry no information and should be removed already at the EDA
  stage.

**Data specifics worth taking into account during EDA:**
- The `Income` column has missing values (24 rows out of 2240) — you should
  explicitly decide what to do with them (removal, imputation by the
  median/mean).
- The target variable is strongly imbalanced: a response is present in about 15%
  of customers.
- In `Marital_Status` there are nonstandard, essentially non-meaningful categories
  (`Absurd`, `YOLO`) that are worth merging with a more general category during
  preprocessing, rather than leaving as separate rare classes.
- The spending features (`Mnt...`) and the per-channel purchase-count features
  (`Num...Purchases`) are strongly correlated with each other — this should be
  taken into account when interpreting the weights of a linear model (see the topic
  `linear-regression-gd`, the section on multicollinearity) and, if desired, used
  as a basis for constructing aggregated features (for example, total spending
  across all categories, total number of purchases via web and catalog).

**What to pay attention to when choosing a metric.** As in the first case, the
task is strongly imbalanced (about 85% vs. 15%), so accuracy alone is not enough —
a constant model "no one will respond" will already give high accuracy but zero
usefulness for the business. It is important to state the business cost of the
errors explicitly: a missed "response" (false negative) means a lost sale, while a
wasted contact with a customer who would not have responded (false positive) means
marketing costs for a useless contact — and these two kinds of errors usually have
different costs. It is worth evaluating recall, precision, F1, and ROC-AUC, as well
as comparing against naive baselines (for example, "no one will respond" or "the
response is the same as to one of the specific past campaigns") — it is precisely
the comparison with a baseline that shows whether making the model more complex is
justified at all.

## General lessons for both cases

- **EDA is not a formality.** In both datasets, the real data problems (leakage
  through a service column in the first case, junk categories and an income
  outlier in the second) are discovered precisely at the exploratory-analysis
  stage — without it they are easy to miss, resulting in unreliably high or,
  conversely, understated model quality.
- **A non-ML and a simple ML baseline are a mandatory reference point.** Before
  comparing complex models against each other, it is worth explicitly computing the
  quality of a simple rule (for example, a threshold on a single "intuitively
  logical" feature) and the quality of a basic logistic regression without special
  feature processing — this keeps you from being fooled by the "impressive"
  metrics of a complex model when in fact the simple rule works almost as well.
- **Feature scaling** noticeably affects the quality of linear models and
  distance-based models (KNN, SVM), but does not matter for trees and
  tree-based ensembles — when comparing models this should be taken into account
  separately for each family.
- **The choice of metric must follow from the business goal, not be a default
  technical choice.** In both tasks accuracy is a poor guide because of class
  imbalance; the decision about what matters more — recall or precision — is made
  based on what is more costly for the business: missing the right customer or
  spending a resource on an unnecessary contact.
- **Interpreting the model is part of solving the task, not an optional bonus.**
  Checking that the features important to the model agree with common sense (for
  example, `Total_Revolving_Bal` and `Total_Trans_Ct` are significant for churn;
  `AcceptedCmp` — for the response to a new campaign) increases trust in the model
  and its fitness for real deployment (see the topic `explainable-ai`).

## What is deliberately not covered in these notes

The final comparison of models (which model gave the best ROC-AUC/recall, which
exact hyperparameters turned out optimal during the sweep) and the final code for
training and comparing models from the bonus notebooks are deliberately not
included in these notes — the goal of the capstone is to build the whole pipeline
yourself, relying on the tools and models from the previous topics of the course
(`eda-and-pipeline`, `classification-metrics`, `decision-trees`,
`ensembles-bagging-boosting`), rather than checking against a ready-made answer in
advance.
