import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

# 1) Загрузка данных
df = pd.read_csv("train.csv")

df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")

df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")

print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())
print(df.dtypes[["Order Date", "Ship Date", "Sales"]])
print(df.duplicated().sum())
print(df.describe())


df = df.dropna(subset=["Order Date", "Sales"])

# Гистограмма продаж
plt.figure()
df["Sales"].hist(bins=50)
plt.title("Распределение продаж")
plt.xlabel("Продажи")
plt.ylabel("Частота")
plt.show()

# Диаграмма размаха для выбросов
plt.figure()
plt.boxplot(df["Sales"], vert=False)
plt.title("Диаграмма размаха")
plt.show()


# A) Средние продажи по дню года
df["day_of_year"] = df["Order Date"].dt.dayofyear

daily_sales = df.groupby("day_of_year")["Sales"].mean()

plt.figure()
daily_sales.plot()
plt.title("Средние продажи по дню года")
plt.xlabel("День в году")
plt.ylabel("Средний объем продаж")
plt.tight_layout()
plt.show()

# B) Тренд продаж по месяцам

monthly_sales = (
    df.resample("ME", on="Order Date")["Sales"]
      .sum()
      .asfreq("ME", fill_value=0)
)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(monthly_sales.index, monthly_sales.values)

ax.set_title("Месячный тренд продаж")
ax.set_xlabel("Месяц")
ax.set_ylabel("Общий объем продаж")

# метка на каждый месяц
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.show()


# C) Тренд продаж по годам
yearly_sales = df.resample("YE", on="Order Date")["Sales"].sum()

plt.figure()
yearly_sales.plot()
plt.title("Годовой тренд продаж")
plt.xlabel("Год")
plt.ylabel("Общий объем продаж")
plt.tight_layout()
plt.show()

# D) Продажи по регионам
region_sales = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)

plt.figure()
region_sales.plot(kind="bar")
plt.title("Продажи по регионам")
plt.xlabel("Регион")
plt.ylabel("Общий объем продаж")
plt.tight_layout()
plt.show()

# E) Продажи по категориям
category_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)

plt.figure()
category_sales.plot(kind="bar")
plt.title("Продажи по категориям")
plt.xlabel("Категории")
plt.ylabel("Общий объем продаж")
plt.tight_layout()
plt.show()

#F) Вклад каждого года
year_sales = df.groupby(df["Order Date"].dt.year)["Sales"].sum()

plt.figure()
year_sales.plot(kind="pie", autopct="%1.1f%%", startangle=90)
plt.title("Распределение продаж по годам")
plt.ylabel("")
plt.tight_layout()
plt.show()

#Реализация моделей
df = df.drop_duplicates()

df["year"] = df["Order Date"].dt.year
df["month"] = df["Order Date"].dt.month
y = df["Sales"]

feature_cols = [c for c in ["year", "month", "Region", "Category", "Sub-Category", "Ship Mode", "Segment"] if c in df.columns]
X = df[feature_cols].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=77
)

numeric_features = [c for c in X.columns if X[c].dtype != "object"]
categorical_features = [c for c in X.columns if X[c].dtype == "object"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# 6) Модель 1: линейная регрессия
model_lr = Pipeline(steps=[
    ("prep", preprocess),
    ("model", LinearRegression())
])

model_lr.fit(X_train, y_train)
pred_lr_test = model_lr.predict(X_test)
pred_lr_train = model_lr.predict(X_train)

plt.figure(figsize=(8, 5))
plt.plot(y_test.values, label="Фактические продажи")
plt.plot(pred_lr_test, label="Прогноз модели")
plt.title("Фактические и предсказанные продажи")
plt.xlabel("Наблюдение")
plt.ylabel("Продажи")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# 7) Модель 2: случайный лес
model_rf = Pipeline(steps=[
    ("prep", preprocess),
    ("model", RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=20,
    min_samples_split=50,
    random_state=77,
    n_jobs=-1))
])

model_rf.fit(X_train, y_train)
pred_rf_test = model_rf.predict(X_test)
pred_rf_train = model_rf.predict(X_train)


plt.figure(figsize=(8, 5))
plt.plot(y_test.values, label="Фактические продажи")
plt.plot(pred_rf_test, label="Прогноз Случайный лес")
plt.title("Фактические и предсказанные продажи (Случайный лес)")
plt.xlabel("Наблюдение")
plt.ylabel("Продажи")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 7) Модель 3: Градиентный бустинг
gbr = Pipeline(steps=[
    ("prep", preprocess),
    ("model", GradientBoostingRegressor(random_state=77))
])

gbr.fit(X_train, y_train)
pred_gbr_test = gbr.predict(X_test)
pred_gbr_train = gbr.predict(X_train)

plt.figure(figsize=(8, 5))
plt.plot(y_test.values, label="Фактические продажи", linewidth=2)
plt.plot(pred_gbr_test, label="Прогноз Gradient Boosting")
plt.title("Фактические и предсказанные продажи (Gradient Boosting)")
plt.xlabel("Наблюдение")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 7) Модель 4: Многослойный персептрон
model_mlp = Pipeline(steps=[
    ("prep", preprocess),
    ("model", MLPRegressor(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        batch_size=16,
        max_iter=2000,
        learning_rate="constant",
        learning_rate_init=0.001,
        random_state=77
    ))
])

model_mlp.fit(X_train, y_train)
mlp = model_mlp.named_steps["model"]
print("Обучение завершено!")
print(f"Количество итераций: {mlp.n_iter_}")
print(f"Финальная потеря: {mlp.loss_:.4f}")
pred_mlp_test = model_mlp.predict(X_test)
pred_mlp_train = model_mlp.predict(X_train)

plt.figure(figsize=(8, 5))
plt.plot(y_test.values, label="Фактические продажи", linewidth=2)
plt.plot(pred_mlp_test, label="Прогноз Многослойного персептрона")
plt.title("Фактические и предсказанные продажи (Прогноз Многослойного персептрона)")
plt.xlabel("Наблюдение")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Метрики
def reg_metrics(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name}")
    print(f"MAE  = {mae:.2f}")
    print(f"RMSE = {rmse:.2f}")
    print(f"R2   = {r2:.3f}")

print("\nОбучающие метрики")
reg_metrics(y_train, pred_lr_train, "Линейная регрессия")
reg_metrics(y_train, pred_rf_train, "Случайный лес")
reg_metrics(y_train, pred_gbr_train, "Градиентный бустинг")
reg_metrics(y_train, pred_mlp_train, "Многослойный персептрон")

print("\nТестовые метрики")
reg_metrics(y_test, pred_lr_test, "Линейная регрессия")
reg_metrics(y_test, pred_rf_test, "Случайный лес")
reg_metrics(y_test, pred_gbr_test, "Градиентный бустинг")
reg_metrics(y_test, pred_mlp_test, "Многослойный персептрон")

#Анализ ошибок
models = {
    "LR": pred_lr_test,
    "RF": pred_rf_test,
    "GBR": pred_gbr_test,
    "MLP": pred_mlp_test
}

for name, pred in models.items():
    errors = y_test - pred

    plt.figure(figsize=(6,4))
    plt.hist(errors, bins=40)
    plt.title(f"Ошибки: {name}")
    plt.xlabel("Ошибка")
    plt.ylabel("Частота")
    plt.tight_layout()
    plt.show()

def plot_y_true_vs_pred(y_true, y_pred, title="y_true vs y_pred"):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx])  # линия идеала: y=x
    plt.title(title)
    plt.xlabel("Факт (y_true)")
    plt.ylabel("Прогноз (y_pred)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# пример для моделей:
plot_y_true_vs_pred(y_test, pred_rf_test, "RF: факт vs прогноз")
plot_y_true_vs_pred(y_test, pred_mlp_test, "MLP: факт vs прогноз")
plot_y_true_vs_pred(y_test, pred_lr_test, "LR: факт vs прогноз")
plot_y_true_vs_pred(y_test, pred_gbr_test, "GBR: факт vs прогноз")


#Интерпретация
importances = pd.Series(
    model_rf.named_steps["model"].feature_importances_,
    index=model_rf.named_steps["prep"].get_feature_names_out()
).sort_values(ascending=False)

print('\n' + 'Интерпретация')
print(importances.head(10))
