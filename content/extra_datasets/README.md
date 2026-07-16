# Extra datasets

🇬🇧 [Read in English](README.en.md) | 🇷🇺 Русский (текущий язык)

Дополнительные датасеты для `_coach/cases.yaml`, не входящие в submodule
`content/ml_basics_course` (тот содержит только датасеты репозитория
Евгения Паточенко). Все ниже — публичные датасеты с открытой лицензией,
скачаны напрямую с [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
и заново собраны в CSV с явным заголовком (у части исходников заголовка не
было вообще).

## [yeast.csv](yeast.csv)
Источник: [UCI Yeast](https://archive.ics.uci.edu/dataset/110/yeast) (1484
строки). Классификация локализации белка в клетке дрожжей по
результатам биохимического анализа сигнальных последовательностей.
Признаки — числовые показатели (`mcg`, `gvh`, `alm`, `mit`, `erl`, `pox`,
`vac`, `nuc`), целевая переменная `class` — 10 классов локализации (CYT,
NUC, MIT, ME3, ME2, ME1, EXC, VAC, POX, ERL), сильно несбалансированных.

## [mushroom.csv](mushroom.csv)
Источник: [UCI Mushroom](https://archive.ics.uci.edu/dataset/73/mushroom)
(8124 строки). Съедобность гриба (`class`: `e` — edible, `p` — poisonous)
по 22 категориальным морфологическим признакам (форма шляпки, запах, цвет
спор и т.д.). Один признак (`stalk_root`) содержит пропуски (`?`).

## [wholesale_customers.csv](wholesale_customers.csv)
Источник: [UCI Wholesale customers](https://archive.ics.uci.edu/dataset/292/wholesale+customers)
(440 строк). Годовые траты (в у.е.) оптовых клиентов дистрибьютора по
категориям товаров (`Fresh`, `Milk`, `Grocery`, `Frozen`,
`Detergents_Paper`, `Delicassen`) + `Channel`/`Region`. Без целевой
переменной — датасет для кластеризации/сегментации.

## [sms_spam_collection.csv](sms_spam_collection.csv)
Источник: [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
(5574 строки). Текст SMS-сообщения (`text`) и метка `label` (`ham`/`spam`).
Классы несбалансированы (~13% spam) — используется в кейсе для
NLP-классификации текста.
