# Time series

These notes were assembled from the course lecture (the recording of lecture
19, second half — theory: the components of a series, stationarity,
autocorrelation, statistical and adaptive models, an overview of ML
approaches to forecasting) and a seminar from Evgeny Patochenko's repository
(`lesson_13` — practice on real data: working with date/time in pandas,
decomposing a series, the ARIMA model, quality evaluation via RMSE).

## Problem statement

A **time series** is a sequence of values describing a process unfolding over
time, usually at equal time intervals (although strictly speaking this is not
required — what matters is that the observations come in sequence). Formally
it is an array of numbers representing the values of some measured variable at
successive moments in time with some constant step.

Examples of time series: sales volumes in retail chains, consumption volumes
and electricity prices, warehouse inventory levels, market stock prices, road
traffic, cryptocurrency prices.

**The forecasting task:** build a model that, given the values of the series
up to moment t, produces a forecast for moment t + d, where d is the forecast
horizon.

**The key difference from classification and regression:** there, the
observations are usually assumed to be independent of each other; in time
series, on the contrary, the assumption is that the data are dependent on each
other — the very existence of the forecasting task relies on the fact that a
future value depends on past ones.

## Components of a time series

Usually several components are distinguished:
1. **Trend** — a long-term change in the level of the series.
2. **Seasonality** — cyclical changes in the level of the series with a
   constant period tied to calendar dates (for example, tangerines get
   cheaper in January; corporate lending has pronounced seasonality tied to
   the business cycle — in summer and during the New Year holidays activity
   freezes, and the bulk of loans are taken out in May).
3. **Cyclicity** — similar to seasonality, but tied not to a fixed calendar
   period, but to a longer and less strictly regular cycle (for example, the
   economic cycle or the market life cycle of a product) — both the length and
   the amplitude of the cycle can vary, unlike seasonality.
4. **Noise (the random/irregular component)** — the remaining unpredictable
   part.

A series can be (approximately) decomposed into the sum of these components —
an **additive model** (Y = trend + seasonality + residual), or into their
product — a **multiplicative model** (Y = trend × seasonality × residual). The
difference: in the multiplicative model the seasonal and random components are
expressed in relative terms (coefficients), in the additive one — in absolute
terms. The models give close results if the amplitude of the fluctuations in
the levels of the series changes little over time.

### Decomposing a series in practice

```python
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(dfBang.priceModLog, model="additive")
decomposition.plot()

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
```

## Stationarity

A series is called **stationary** if its statistical properties do not change
over time. Three properties must remain unchanged:
1. **The mean value** — must not depend on time, must not grow or decrease
   (the presence of a trend makes a series non-stationary).
2. **The variance** — must be constant over time, the amplitude of the
   fluctuations must not grow or shrink.
3. **Autocorrelation (covariance)** — must not depend on time, only on the
   distance between observations; there must be no systematically repeating
   patterns over time.

In summary: presence of a trend → the series is non-stationary; presence of
seasonality → the series is non-stationary; a change in variance → the series
is non-stationary. The presence or absence of cyclicity by itself does not
affect stationarity.

**Why stationarity is needed.** A stationary series is much easier for models
to work with and makes it easier to produce a robust forecast — most time
series models in one way or another model and predict the statistical
characteristics of the series (the mean, the variance), and if the series is
non-stationary, these predictions will turn out to be wrong. Unfortunately,
most real-world series are not stationary, but this can and should be dealt
with.

### Checking stationarity: tests

- **The Dickey-Fuller test** — the most well-known criterion for checking
  stationarity. In practice it is applied in literally a single line of code.
- **The KPSS test** — the Dickey-Fuller test does not always reliably "catch"
  non-stationarity, so it is worth additionally checking the series with this
  test as well.

```python
test = sm.tsa.adfuller(dfBang.priceMod)
print('adf: ', test[0])
print('p-value: ', test[1])
print('Critical values: ', test[4])
if test[0] > test[4]['5%']:
    print('There are unit roots, the series is non-stationary')
else:
    print('There are no unit roots, the series is stationary')

# p-value > 0.05: we do not reject H0 — the series has a unit root, non-stationary.
# p-value <= 0.05: we reject H0 — the series has no unit root, stationary.
```

### Bringing a series to a stationary form

1. **Stabilizing the variance — the Box-Cox transformation.** If the variance
   changes over time, the Box-Cox transformation helps to stabilize it. Taking
   the logarithm of the series is a special case of this kind of
   transformation and often helps stabilize the spread of the values.
2. **Removing the trend — differencing.** Computing the difference between the
   current and the previous value of the series stabilizes the mean (removes
   the trend). It can be applied repeatedly until the trend disappears.
3. **Removing seasonality — seasonal differencing.** Moving to pairwise
   differences of the values in neighboring seasons (that is, differencing not
   against the neighboring point, but with a shift by the length of the
   season).

A typical practical order of actions: apply the Dickey-Fuller test → if the
series is non-stationary, apply a suitable transformation (Box-Cox for the
variance, differencing for the trend, seasonal differencing for seasonality) →
repeat the test → and so on, until the series becomes stationary.

```python
# Log transformation (a special case of variance stabilization)
dfBang['priceModLog'] = np.log(dfBang.priceMod)

# Computing lags and differences (differencing)
dfBang["priceModLogShift1"] = dfBang.priceModLog.shift()
dfBang["priceModLogDiff"] = dfBang.priceModLog - dfBang.priceModLogShift1
```

## Autocorrelation: ACF and PACF

An **autoregressive model** is a linear regression model that predicts the
value of the series based on several previous values of the series itself (the
assumption: the current value depends only on a few preceding values).

When building such a model, what is essentially estimated is the
**autocorrelation** — how much the series correlates with itself. The series
is shifted by several moments in time, and the correlation between the
original and the shifted series is computed. The function showing this
correlation at different shifts is called the **autocorrelation function
(ACF)**.

The **partial autocorrelation (PACF)** tries to isolate the "pure" influence
of a particular shifted value on the current one, removing the effect of all
the intermediate values of the series between them (the ordinary ACF reflects
the combined effect of the whole chain of intermediate dependencies, rather
than the isolated influence of a particular past value).

- The ACF helps determine the order q of the MA (moving average) component —
  from the correlogram you can determine how many autocorrelation coefficients
  are strongly different from zero.
- The PACF helps determine the order p of the AR (autoregression) component —
  from the correlogram you determine the maximum index of a coefficient that
  is strongly different from zero.

```python
from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(ts_diff, nlags=20)
lag_pacf = pacf(ts_diff, nlags=20, method='ols')

ACF = pd.Series(lag_acf)
ACF.plot(kind="bar")

PACF = pd.Series(lag_pacf)
PACF.plot(kind="bar")
```

## Overview of forecasting approaches

Conventionally, four directions are distinguished:
1. Classical statistical (econometric) methods.
2. Methods based on classical machine learning.
3. Methods based on deep learning (neural networks).
4. Methods based on large language models.

## Classical statistical models

- **AR (autoregression)** — predicts the current value of the series as a
  linear combination of several previous values.
- **MA (moving average, as a model component, not to be confused with the
  simple moving average)** — forecasts the value of the series as a linear
  combination of the **residuals** (the errors of previous forecasts), to
  refine the forecast.
- **ARMA** — combines AR and MA in a single model. Wold's theorem states that
  any stationary series can be approximated by an ARMA model to any desired
  accuracy — this is precisely why the effort is made to bring a series to a
  stationary form. This makes ARMA a very general and powerful model (by
  analogy with the universal approximation theorem for neural networks).
- **ARIMA** — an extension of ARMA for non-stationary series with a trend: the
  model itself performs the necessary differencing internally, applies ARMA,
  and then automatically "reverses" the differencing the required number of
  times to obtain the forecast in the original scale — all of this out of the
  box.
- **SARIMA** — extends ARIMA with an additional seasonality hyperparameter,
  allowing it to work with series that have both a trend and seasonality,
  without manually removing the seasonality.
- **SARIMAX** — additionally can account for **exogenous factors** — external
  variables besides the series itself that may influence it.

The order of an ARIMA model is set by three parameters: p — the order of the
AR component, d — the order of differencing (the degree of integration) of the
series, q — the order of the MA component. The values of p and q are selected
from the ACF (for q) and PACF (for p) correlograms.

These models are well implemented in Python libraries (`statsmodels`, as well
as specialized libraries for time series, for example the Russian "Etna").

### Building ARIMA in practice

```python
from statsmodels.tsa.arima.model import ARIMA

# ARIMA Model (p=1, d=0 — differencing already done manually, q=1)
model_AR1MA = ARIMA(ts_diff, order=(1, 0, 1))
results_ARIMA = model_AR1MA.fit()

results_ARIMA.fittedvalues.head()
```

```python
# Reverse the differencing to return to the original scale of the series
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()

predictions_ARIMA_log = pd.Series(ts.iloc[0], index=ts.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)

dfBang['priceARIMA'] = np.exp(predictions_ARIMA_log)
```

Tuning the parameters for ARIMA in practice is a complex and labor-intensive
process; there is no "silver bullet." Methods developed back in the middle of
the 20th century are still popular on a par with newer approaches based on
neural networks (LSTM, RNN).

## Adaptive methods

Unlike statistical models, where non-stationarity is removed once (manually or
out of the box in ARIMA/SARIMA/SARIMAX), adaptive methods continuously
**self-tune and self-correct** as new data arrives, rather than undergoing a
one-time bringing to a stationary form.

- **The simple moving average** — the forecast is the average of the last n
  values of the series. A simple and not very powerful model; it works well
  for short-term forecasting (for example, intraday trading) and for smoothing
  out noise.
- **The exponential moving average** — the values of the series are weighted
  with decreasing weight (the further in the past, the smaller the weight, in a
  geometric progression), rather than being averaged equally. The optimal
  weights are derived analytically by minimizing the squared error — the
  coefficients are not "pulled out of thin air." The formula is conveniently
  written recurrently: the new forecast in terms of the previous forecast and
  the new observation with a smoothing parameter α — the larger α, the more
  weight is given to the most recent points.
- **The Holt model** — extends exponential smoothing with a **trend**
  component; it performs quite well on series with a trend and can adjust even
  if the trend itself changes over time.
- **The Holt-Winters model** — additionally accounts for **seasonality**, on
  top of the trend; there are variants with additive and multiplicative
  seasonality (the trend and seasonality components are added or multiplied).

In practice it is not known in advance which family of models (statistical,
adaptive, ML, DL) will work better for a particular series — so several
candidate models are built, part of the data is held out for validation (with
respect to the temporal order!), and the best variant is chosen by a metric on
the held-out data, rather than "by eye."

## Forecasting with machine learning methods

The advent of machine learning did not so much replace statistical and
adaptive approaches as enrich the set of available models. The recommended
order of work on a real task: first a baseline is built on a
statistical/adaptive model (ARIMA/SARIMA/SARIMAX or something adaptive) — they
are often already good enough; only then, if the baseline is not of sufficient
quality, do you move on to ML.

**Important about validation:** in cross-validation and the train/test split
for time series, unlike ordinary tasks, you cannot shuffle the data chaotically
— the sequence of observations matters. Training always goes on older data,
evaluation — on newer data.

**The simplest approach — linear regression** with features specially
constructed for a time series:
1. **Lag features** — the values of the series at previous moments in time
   (yesterday, the day before yesterday, etc.) as separate input features.
2. **Aggregated features** — window statistics: the mean value over a period,
   the volatility (spread) over a period, or related external indicators (for
   example, the average temperature over a month when forecasting the cost of
   gas).
3. **Exogenous features** — features outside the series itself: an exchange
   rate, a transport accessibility indicator, etc.

**Libraries** that implement a significant part of this feature engineering
"under the hood":
- **Prophet** (Meta) — implements additive models with automatic accounting
  for seasonality, trend and automatic hyperparameter tuning; it often gives
  high quality in literally a single line of code.
- A library from T-Bank — contains deep-learning-based models for time series.

## Practice: the time series workflow (seminar)

### Preparing the data: working with date/time in pandas

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller

df = pd.read_csv(MARKET_ARRIVALS)

# Convert the date column to a real datetime type
df.date = pd.DatetimeIndex(df.date)
```

For analyzing a time series it is useful to set the date/time as the index of
the dataframe (it is important to sort the data by date first):

```python
dfBang = dfBang.sort_values(by="date")
dfBang.index = pd.PeriodIndex(dfBang.date, freq='M')
dfBang.drop('date', axis=1, inplace=True)

dfBang.priceMod.plot()  # a line chart — the standard way to visualize a series
```

### The quality metric and the simplest baselines

To evaluate the forecast quality, RMSE (Root Mean Squared Error — the square
root of the mean squared deviation of the predicted values from the actual
ones) was used:

```python
def RMSE(predicted, actual):
    mse = (predicted - actual)**2
    rmse = np.sqrt(mse.sum()/mse.count())
    return rmse
```

**Forecasting with the mean** (Mean Constant Model) — the simplest baseline:

```python
model_mean_pred = dfBang.priceModLog.mean()
dfBang["priceMean"] = np.exp(model_mean_pred)

model_mean_RMSE = RMSE(dfBang.priceMean, dfBang.priceMod)
```

**A linear trend model** (regression of the target variable on the ordinal
number of the month) via `statsmodels`:

```python
min_date = dfBang.date.min()
dfBang['months_since_min'] = dfBang.date.apply(lambda x: (x - min_date).n)

model_linear = smf.ols('priceModLog ~ months_since_min', data=dfBang).fit()
model_linear.summary()

model_linear_pred = model_linear.predict()
dfBang["priceLinear"] = np.exp(model_linear_pred)
```

**A linear model with additional regression** (several features at once — for
example, the month and the logarithm of the sales volume):

```python
model_linear_quantity = smf.ols('priceModLog ~ months_since_min + np.log(quantity)', data=dfBang).fit()
model_linear_quantity.summary()
```

**Forecasting into the future**: the model is trained on part of the sample,
the forecast is built on the remaining (newer) data, and compared with the
real values — this is exactly how validation for time series is set up, with
the temporal order preserved rather than a random shuffle:

```python
model_linear_quantity = smf.ols(
    'priceModLog ~ months_since_min + np.log(quantity)',
    data=dfBang.iloc[:-15, :]
).fit()

prediction = np.exp(model_linear_quantity.predict(dfBang.iloc[-15:, :]).values)
```

### Seminar takeaway

The models considered (the mean, a linear trend, a linear regression with
additional features, decomposition, ARIMA) were compared by RMSE in a single
results table — the typical approach when choosing the best forecasting model
for a particular series. The final conclusion of the seminar: there is no
silver bullet for forecasting time series — the task is largely creative and
exploratory, and for each series you often have to select and try something of
your own, taking into account the balance between quality and effort.
