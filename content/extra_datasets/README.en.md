# Extra datasets

🇷🇺 [Читать на русском](README.md) | 🇬🇧 English (current)

Extra datasets for `_coach-en/cases.yaml` that aren't part of the
`content/ml_basics_course` submodule (which only contains datasets from
Evgeny Patochenko's repository). Everything below is a public,
openly-licensed dataset downloaded directly from the [UCI Machine Learning
Repository](https://archive.ics.uci.edu/) and re-packaged as CSV with an
explicit header (some of the originals had no header at all).

## [yeast.csv](yeast.csv)
Source: [UCI Yeast](https://archive.ics.uci.edu/dataset/110/yeast) (1484
rows). Classifying a protein's localization site within a yeast cell from
the biochemical analysis of its signal sequences. Features are numeric
measurements (`mcg`, `gvh`, `alm`, `mit`, `erl`, `pox`, `vac`, `nuc`); the
target variable `class` has 10 localization classes (CYT, NUC, MIT, ME3,
ME2, ME1, EXC, VAC, POX, ERL), strongly imbalanced.

## [mushroom.csv](mushroom.csv)
Source: [UCI Mushroom](https://archive.ics.uci.edu/dataset/73/mushroom)
(8124 rows). Mushroom edibility (`class`: `e` — edible, `p` — poisonous)
from 22 categorical morphological features (cap shape, odor, spore color,
etc.). One feature (`stalk_root`) has missing values (`?`).

## [wholesale_customers.csv](wholesale_customers.csv)
Source: [UCI Wholesale customers](https://archive.ics.uci.edu/dataset/292/wholesale+customers)
(440 rows). Annual spend (monetary units) of a wholesale distributor's
customers by product category (`Fresh`, `Milk`, `Grocery`, `Frozen`,
`Detergents_Paper`, `Delicassen`) plus `Channel`/`Region`. No target
variable — a dataset for clustering/segmentation.

## [sms_spam_collection.csv](sms_spam_collection.csv)
Source: [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
(5574 rows). SMS message text (`text`) and a `label` (`ham`/`spam`).
Classes are imbalanced (~13% spam) — used in the NLP text classification
case.
