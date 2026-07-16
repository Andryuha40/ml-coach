# Разведочный анализ данных, sklearn Pipeline, кодирование признаков

Конспект собран из лекции курса «Класс Pipeline» (структура и мотивация
Pipeline — почти дословно совпадает по содержанию с семинаром
`07_Класс_Pipeline.ipynb`, поэтому код взят из семинара как более полный и
рабочий вариант), семинара курса «Разведочный анализ данных» (EDA на
данных об оттоке клиентов телекома — основной источник по EDA-технике),
семинара репозитория `lesson_01` (Sweetviz — короткое дополнение про
авто-EDA библиотеки), семинара репозитория `lesson_04` (регуляризация +
разведочный анализ на данных о цене жилья — источник примеров про
утечку данных и подбор alpha) и ноутбука-подсказки `lesson_04`
(«Конструирование признаков» — источник по кодированию категориальных
признаков, масштабированию и генерации признаков из дат).

## Зачем нужен разведочный анализ данных (EDA)

Перед тем как обучать любую модель, важно хорошо понять данные, с которыми
предстоит работать. Разведочный анализ данных (Exploratory Data Analysis,
EDA) помогает:
- понять структуру и содержание данных;
- выявить возможные проблемы (пропущенные значения, дубликаты, выбросы);
- определить основные характеристики переменных и их распределения;
- определить взаимосвязи между переменными и с целевой переменной.

## Первичный анализ таблицы

Базовый набор действий на новом датасете (пример — датасет об оттоке
клиентов телекома, задача — предсказать, уйдёт ли клиент):

```python
df.head(5)          # первые строки, беглый взгляд на структуру
df.info()           # типы столбцов, число непустых значений
df.shape             # число строк и столбцов
df.isnull().sum()   # проверка пропусков по каждому столбцу
df.describe().T                    # статистики по числовым столбцам
df.describe(include='object').T    # статистики по категориальным столбцам
```

### Пропуски

Что делать с пропусками, зависит от их доли и природы:
- если строк с пропусками мало (небольшой процент от всех данных) — можно
  удалить эти строки;
- если в конкретном столбце пропусков очень много — возможно, стоит
  удалить весь столбец;
- заполнить пропуски статистикой (медианой или средним);
- есть и более продвинутые способы — предсказывать пропущенное значение
  моделью машинного обучения по остальным признакам.

**Скрытые пропуски.** Пропуск не всегда выглядит как `NaN`. В примере с
оттоком клиентов столбец `TotalCharges` по смыслу числовой, но хранится как
строка (`object`) — при проверке оказалось, что часть значений — это просто
пробел `' '`, из-за чего pandas не смог автоматически распознать столбец
как числовой:

```python
df[df['TotalCharges'] == ' ']   # находим "скрытые" пропуски, замаскированные под пробел

def to_float(x):
    try:
        return float(x)
    except:
        return 0

df['TotalCharges_num'] = df['TotalCharges'].apply(to_float)
```

Дальнейший анализ показал, что `TotalCharges` на самом деле равен
`MonthlyCharges * tenure` (число месяцев обслуживания), поэтому пропуски в
нём можно было корректно заменить нулём (соответствует новым клиентам с
tenure = 0). Это заодно показало, что `TotalCharges` — линейная комбинация
двух других признаков и не добавляет новой информации; такой вывод пригодится
позже, при построении моделей (мультиколлинеарность).

### Выбросы

Один из стандартных способов обнаружения выбросов в числовом признаке —
межквартильный размах (IQR):

```python
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = data[(data < lower_bound) | (data > upper_bound)]
```

### Распределения признаков

Для числовых признаков — гистограммы (`sns.histplot`), в том числе с
разбивкой по целевой переменной, чтобы увидеть, отличается ли распределение
признака между классами; и boxplot — для визуального сравнения медианы,
разброса и выбросов между группами:

```python
for var in num_columns:
    sns.histplot(df, x=var, hue='Churn', bins=30)
    plt.title(f'Распределение {var}')
    plt.show()

for var in num_columns:
    sns.boxplot(df, x=var, hue='Churn')
    plt.show()
```

Для категориальных/бинарных признаков — countplot с разбивкой по целевой
переменной:

```python
for var in categorical_variables:
    sns.countplot(data=df, x=var, hue='Churn')
    plt.xticks(rotation=90)
    plt.show()
```

Универсальная функция для поочерёдного разбора каждого столбца (числового
или категориального) — печатает статистики и строит гистограмму +
boxplot (для числовых) или распределение частот категорий (для
категориальных):

```python
def column_eda(df, column, dtype='numeric', figsize=(15, 5)):
    data = df[column]
    print(f"Столбец: {column}")
    print(f"Пропусков: {data.isna().sum()}")

    if dtype == 'numeric':
        print(f"Мин: {data.min()}, Макс: {data.max()}, Среднее: {data.mean()}")
        print(f"Медиана: {data.median()}, Стд: {data.std()}")
        Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = data[(data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)]
        print(f"Выбросы: {len(outliers)}")

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].hist(data.dropna(), bins=min(50, len(data.unique())), density=True)
        data.dropna().plot(kind='kde', ax=axes[0], color='red')
        axes[1].boxplot(data.dropna(), vert=True)
        plt.show()

    elif dtype == 'categorical':
        value_counts = data.value_counts()
        print(f"Уникальных значений: {data.nunique()}")
        # ... вывод топ-значений и bar-графики частот/долей
```

### Взаимосвязи между переменными

**Корреляция Пирсона** — мера линейной взаимосвязи двух числовых столбцов,
самая распространённая («число-число»):

```python
correlation_matrix = df[num_columns].corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
```

Для более полной картины по разным типам пар признаков:

- **Корреляция Спирмена** (`df.corr(method='spearman')`) — основана на
  ранжировании значений, измеряет степень **монотонной** (не обязательно
  линейной) связи; подходит и для порядковых переменных.
- **Корреляция Кендалла** (`df.corr(method='kendall')`) — похожа на
  Спирмена, чаще используется для пары номинальный–номинальный признак.
- **V-мера Крамера** («категориальный–категориальный») — нормировка
  статистики хи-квадрат (χ²) на число категорий, приводит результат к
  отрезку [0, 1] для лёгкой интерпретации. Статистика хи-квадрат считается
  по таблице сопряжённости: χ² = сумма по всем ячейкам (O − E)² / E, где O
  — наблюдаемая частота, E — ожидаемая частота при независимости признаков.
  Чем больше χ², тем сильнее взаимосвязь.
- **ANOVA** («число–категориальный») — сравнивает средние значения
  числового признака между тремя и более группами, заданными категориальным
  признаком; выдаёт F-статистику и p-value (`scipy.stats.f_oneway`). Если
  p-value < 0.05, различия между группами считаются статистически
  значимыми (признаки скорее связаны).
- **φₖ (phik)** — коэффициент корреляции, единообразно применимый ко всем
  типам признаков: для категориальных считается по сути тот же χ², а для
  числовых значения предварительно бинаризуются (разбиваются на интервалы),
  и дальше тоже считается χ² (библиотека `phik`).

**Важная оговорка: корреляция — не то же самое, что причинно-следственная
связь** («correlation is not causation»). Пример: продажи мороженого и
число нападений акул коррелируют по месяцам, но продажи мороженого не
влияют на акул — оба явления зависят от третьего фактора (жаркая погода).
Аналогично, положительная корреляция между ростом и весом человека не
означает, что один параметр напрямую влияет на другой — оба зависят,
например, от возраста.

## Автоматизированный EDA (AutoEDA-библиотеки)

Существуют библиотеки, которые строят подробный HTML-отчёт по датасету
(распределения, пропуски, корреляции, предупреждения о проблемах в данных)
буквально в одну строчку кода:

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

Такие инструменты удобны для быстрого первого взгляда на новый датасет, но
не заменяют внимательный ручной анализ (например, они не «поймают» скрытый
пропуск, замаскированный под пробел, как в примере выше, если он формально
не распознаётся библиотекой как NaN).

## Зачем нужен Pipeline

### Проблема: код без Pipeline

Стандартная последовательность действий перед обучением модели —
разделить признаки по типам, применить к ним разную предобработку
(масштабирование числовых, кодирование категориальных), объединить
результат и обучить модель:

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

Проблема в том, что для валидации/теста придётся **вручную повторить** те же
самые шаги преобразования (при этом важно вызывать `transform`, а не
`fit_transform`, — см. про утечку данных ниже):

```python
data_test_scaled = scaler.transform(data_test[numerical_features])
data_test_ohe = ohe.transform(data_test[to_dummies])
data_test_transformed = pd.concat([...], axis=1)
```

**Итог такого подхода**: дублирование кода, сложно масштабировать на новые
признаки, сложно поддерживать, отнимает много времени, а главное — неясно,
как сохранить всю эту последовательность шагов как единый объект для
переиспользования в продакшене.

### Решение: Pipeline

`Pipeline` — класс из `sklearn.pipeline`, объединяющий несколько шагов
обработки данных и саму модель в единую цепочку. Возможности:
- объединяет несколько действий в один объект;
- убирает дублирование кода (шаги преобразования пишутся один раз и
  автоматически применяются одинаково к train и test — `fit` на train,
  `transform` на train и test);
- с пайплайном можно работать как с единым объектом sklearn, вызывая
  стандартные `.fit()` и `.predict()`;
- пайплайн можно сохранить целиком (например, через `pickle`) для
  дальнейшего переиспользования в продакшене.

Ключевые вспомогательные классы:
- **`ColumnTransformer`** (из `sklearn.compose`) — применяет разные
  преобразования к разным подмножествам столбцов;
- **`SimpleImputer`** — заполнение пропусков;
- **`SelectKBest`** — отбор k лучших признаков по статистическому критерию;
- **`FunctionTransformer`** — превращает произвольную функцию в трансформер;
- **`FeatureUnion`** — объединяет результаты нескольких трансформеров в один
  датасет;
- **`make_pipeline`** — удобная обёртка для создания пайплайна без
  необходимости вручную придумывать имена шагов.

**Когда применять Pipeline**: не сразу — сначала должен быть завершён этап
EDA и появиться понимание итоговой структуры модели (какие признаки
использовать, какого рода предобработка нужна). Pipeline применяется на
стадии перехода от EDA к обучению/дообучению модели, стабилизации и
подготовке модели к переводу в продакшен.

### Сборка Pipeline с ColumnTransformer

```python
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

# для числовых признаков — заполнение пропусков, затем масштабирование, затем отбор признаков
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer()),
    ("scaler", StandardScaler()),
    ("fs", SelectKBest(score_func=f_classif, k="all")),
])

# для категориальных признаков — заполнение пропусков, затем OneHotEncoder
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer()),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

# объединяем трансформеры для разных типов столбцов
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

### Подбор гиперпараметров у Pipeline целиком

Гиперпараметры любого шага пайплайна доступны через двойное подчёркивание
`<имя_шага>__<параметр>` (в том числе вложенное — можно достучаться до
гиперпараметра трансформера внутри `ColumnTransformer` внутри `Pipeline`):

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

### Pipeline с несколькими моделями: голосование и стекинг

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
import xgboost

# Голосование (blending) нескольких моделей разных семейств
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

# Стекинг: каждая базовая модель — свой отдельный пайплайн (можно с разной предобработкой)
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

Гибкость Pipeline позволяет достать любой вложенный параметр напрямую по
цепочке атрибутов (полезно для отладки собранной композиции):

```python
stacking_classifier.estimators[0][1][0].steps[0][1].transformers[0][1].steps[2][1].k
```

### Сохранение и загрузка Pipeline целиком

```python
import pickle

with open("awesome_pipeline.pkl", "wb") as f:
    pickle.dump(stacking_classifier, f)

with open("awesome_pipeline.pkl", 'rb') as f:
    pipeline_from_saved = pickle.load(f)
```

## Утечка данных (data leakage) при неправильном препроцессинге

**Ключевое правило**: любые статистики для предобработки (среднее и
стандартное отклонение для масштабирования, средние значения таргета для
target encoding, коэффициенты регуляризации и т.п.) должны считаться
**только по обучающей выборке**, а затем этим же, уже посчитанным,
преобразованием обрабатываются и train, и test/valid — без пересчёта на
новых данных.

На практике это выражается в разнице между методами `fit_transform` (для
train) и `transform` (для test):

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # считает среднее/std и сразу применяет
X_test_scaled = scaler.transform(X_test)          # применяет то же самое преобразование, БЕЗ пересчёта — не fit_transform!
```

Если по ошибке вызвать `fit_transform` на тестовой выборке (или, что ещё
хуже, объединить train и test перед предобработкой), статистики теста
«просочатся» в предобработку — модель окажется настроена с учётом
информации, которой у неё не должно было быть на момент предсказания. Это
и называется **утечкой данных (data leakage)**: оценка качества модели на
таких «утёкших» данных получается искусственно завышенной и не отражает
реального качества модели на действительно новых данных.

Аналогичный принцип действует при подборе гиперпараметров: коэффициент
регуляризации (`alpha` у Ridge/Lasso) нельзя подбирать ни по обучающей
выборке (модель просто выберет минимально возможную регуляризацию и
переобучится), ни по тестовой (тест перестанет быть «честной» оценкой
качества — она незаметно «просочится» в процесс обучения). Правильный
способ — подбор по валидационной выборке или через кросс-валидацию, а
тестовая выборка остаётся нетронутой до самого конца:

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

alphas = np.logspace(-2, 3, 20)   # перебор по логарифмической сетке — чтобы нащупать порядок величины
searcher = GridSearchCV(Ridge(), [{"alpha": alphas}],
                         scoring="neg_root_mean_squared_error", cv=10)
searcher.fit(X_train_scaled, y_train)
best_alpha = searcher.best_params_["alpha"]
```

Другой характерный источник утечки — признаки-идентификаторы (`id`) или
признаки, напрямую вычисляемые из других признаков одним и тем же
детерминированным образом: например, столбец `TotalCharges` в примере с
оттоком клиентов оказался равен `MonthlyCharges × tenure` — такая скрытая
линейная зависимость сама по себе не «утечка» в строгом смысле (это не
информация из будущего), но столь же исправно приводит к переобучению и
неинтерпретируемым весам линейной модели, если её не выявить на этапе EDA.

### Кросс-валидация как более честная оценка качества

Кросс-валидация — способ оценить качество модели надёжнее, чем на одном
случайном train/test-разбиении: обучающая выборка делится на n частей
(фолдов), обучаются n моделей — каждая на всех фолдах, кроме одного
(отложенного, «out-of-fold»), и оценивается на этом отложенном фолде.
Итоговая метрика — среднее по n полученным значениям:

```python
from sklearn.model_selection import cross_val_score

model = LinearRegression()
cv_scores = cross_val_score(model, X_train[numeric_features], y_train,
                             cv=10, scoring="neg_root_mean_squared_error")
print("Mean CV RMSE = %.4f" % np.mean(-cv_scores))
```

Обратите внимание на знак: стандартные скореры sklearn для метрик, которые
нужно минимизировать (например, RMSE), называются `neg_*`, поскольку
внутренний механизм sklearn всегда **максимизирует** переданную
скоринговую функцию — поэтому и результат приходит с минусом.

## Кодирование категориальных признаков

Линейные модели не умеют работать с текстовыми категориями напрямую — их
нужно закодировать числами. Способов много, и выбор зависит от типа
категориального признака (номинальный/ранговый) и числа уникальных
значений:

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, TargetEncoder

df = pd.DataFrame({
    'район': ['Центр', 'Север', 'Юг', 'Север', 'Центр'],
    'тип_дома': ['Кирпич', 'Панель', 'Монолит', 'Кирпич', 'Панель'],
    'состояние': ['Отличное', 'Требует ремонта', 'Хорошее', 'Удовлетворительное', 'Хорошее'],
    'цена_млн': [12.5, 7.2, 9.8, 8.1, 10.3]
})

# 1. Label Encoding — каждой категории присваивается число 0, 1, 2, ...
# Опасно для номинальных признаков: модель может решить, что "2 > 1", хотя категории не упорядочены
le = LabelEncoder()
df['район'] = le.fit_transform(df['район'])

# 2. One-Hot Encoding — для каждой категории отдельный бинарный столбец
df_ohe = pd.get_dummies(df, columns=['район'])
# То же через sklearn:
encoder_ohe = OneHotEncoder(sparse_output=False)
encoded = encoder_ohe.fit_transform(df[['район']])

# One-Hot с drop_first=True — убирает один столбец (избегает избыточности/мультиколлинеарности
# при использовании с линейными моделями, для которых полный набор дамми-переменных избыточен)
df_ohe_drop = pd.get_dummies(df, columns=['район'], drop_first=True)

# 3. Ordinal Encoding — для РАНГОВЫХ признаков, где порядок категорий имеет смысл
encoder_ord = OrdinalEncoder(
    categories=[['Требует ремонта', 'Удовлетворительное', 'Хорошее', 'Отличное']]
)
df['состояние'] = encoder_ord.fit_transform(df[['состояние']])

# 4. Target (Mean) Encoding — категория заменяется средним значением таргета по этой категории
# cv и smooth нужны, чтобы уменьшить переобучение/утечку целевой переменной
encoder_target = TargetEncoder(cv=3, smooth=0.2)
df['район'] = encoder_target.fit_transform(df[['район']], df['цена_млн'])

# 5. Binary Encoding и Leave-One-Out Encoding — из отдельной библиотеки category_encoders,
# компромисс между компактностью One-Hot и информативностью Target Encoding
import category_encoders as ce
encoder_bin = ce.BinaryEncoder(cols=['район', 'тип_дома'])
encoder_loo = ce.LeaveOneOutEncoder(cols=['район', 'тип_дома'])
```

**Практическое правило выбора**: One-Hot хорошо работает при небольшом
числе уникальных категорий (иначе датасет сильно раздувается вширь и
становится разреженным); при большом числе категорий предпочтительнее
mean/target encoding или счётчики. Часто комбинируют: если категорий в
признаке меньше порога (например, 5) — one-hot, если больше — target
encoding.

## Масштабирование признаков

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# StandardScaler — вычесть среднее, поделить на стандартное отклонение (когда данные похожи на нормальное распределение)
scaler_std = StandardScaler()

# MinMaxScaler — сжать значения в диапазон [0, 1] (когда важны точные границы диапазона)
scaler_mm = MinMaxScaler()

# RobustScaler — использует медиану и межквартильный размах вместо среднего и std,
# поэтому устойчивее к выбросам, чем StandardScaler
scaler_rob = RobustScaler()

X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit только на трейне
X_test_scaled = scaler.transform(X_test)          # transform — на обоих (не fit_transform!)
```

## Другие приёмы конструирования признаков (feature engineering)

**Логарифмирование** — сглаживает сильно скошенные распределения (например,
цены, доходы), уменьшает влияние больших выбросов:

```python
df['цена_log'] = np.log(df['цена_млн'])
df['цена_log+1'] = np.log1p(df['цена_млн'])   # log(x + 1) — работает и при x = 0
```

**Полиномиальные признаки** — позволяют линейной модели уловить нелинейную
зависимость, добавляя в качестве новых признаков степени и попарные
произведения исходных:

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(df[['площадь']])
```

**Признаки из дат** — распаковка одного столбца с датой в набор
содержательных числовых/категориальных признаков:

```python
df_dates['год'] = df_dates['дата_заявки'].dt.year
df_dates['месяц'] = df_dates['дата_заявки'].dt.month
df_dates['день_недели'] = df_dates['дата_заявки'].dt.dayofweek  # 0 = понедельник
df_dates['квартал'] = df_dates['дата_заявки'].dt.quarter
df_dates['номер_недели'] = df_dates['дата_заявки'].dt.isocalendar().week

df_dates['выходной'] = df_dates['день_недели'].isin([5, 6]).astype(int)

import holidays
ru_holidays = holidays.Russia()
df_dates['праздник'] = df_dates['дата_заявки'].apply(lambda x: int(x in ru_holidays))

df_dates['сезон'] = df_dates['месяц'].map({12: 'зима', 1: 'зима', 2: 'зима', ...})

# Разница между двумя датами — тоже полезный признак
df_dates['дней_до_заезда'] = (df_dates['дата_заезда'] - df_dates['дата_заявки']).dt.days
```

## Переобучение и регуляризация: напоминание на числовом примере

Демонстрация переобучения на синтетических данных: истинная зависимость —
косинус, три полиномиальные модели разной степени (1, 4 и 20) обучаются на
30 зашумлённых точках:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

for degree in [1, 4, 20]:
    X_objects = PolynomialFeatures(degree, include_bias=False).fit_transform(x_objects[:, None])
    regr = LinearRegression().fit(X_objects, y_objects)
    y_pred = regr.predict(X)
    # degree=1: модель слишком простая, не может уловить нелинейную зависимость (недообучение)
    # degree=20: модель идеально проходит через все 30 точек, включая шум (переобучение) —
    #            и получает очень большие по модулю веса
```

Переобучение линейных моделей обычно связано именно с большими весами —
отсюда и идея регуляризации: штрафовать функцию потерь за большие значения
весов (подробный разбор L1/L2/ElasticNet — в теме `linear-regression-gd`).
Здесь важно связующее звено: **признаки-идентификаторы** (например, `id`
строки) при обучении почти гарантированно приводят к переобучению — модель
может «зацепиться» за них как за случайно коррелирующий с таргетом признак,
поэтому такие столбцы обычно удаляют на этапе EDA/предобработки, ещё до
Pipeline.

## Итоговый пайплайн предсказания цены жилья (пример полного цикла)

Финальный пример, объединяющий разведочный анализ, кодирование
категориальных признаков, масштабирование и регуляризованную линейную
регрессию в один `Pipeline`:

```python
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

categorical = list(X_train.dtypes[X_train.dtypes == "object"].index)
X_train[categorical] = X_train[categorical].fillna("NotGiven")  # пропуск как отдельная категория
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

# перебор alpha прямо через параметры Pipeline (обратите внимание на синтаксис regression__alpha)
alphas = np.logspace(-2, 4, 20)
searcher = GridSearchCV(lasso_pipeline, [{"regression__alpha": alphas}],
                         scoring="neg_root_mean_squared_error", cv=10, n_jobs=-1)
searcher.fit(X_train, y_train)
```

Полезный побочный эффект L1-регуляризации (Lasso) здесь же на практике:
часть весов Lasso обнуляется полностью (в отличие от Ridge), что можно
проверить и использовать как встроенный отбор признаков:

```python
lasso_zeros = np.sum(pipeline.steps[-1][-1].coef_ == 0)
print("Zero weights in Lasso:", lasso_zeros)
```
