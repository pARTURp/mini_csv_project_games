import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error

# === Загружаем файл ===
data = pd.read_csv("Games.csv", encoding="cp1251")

# === Целевая колонка ===
target = "Rating Average"

# Удаляем строки без рейтинга
data = data[data[target].notna()]

# === Числовые признаки ===
num_features = [
    "Year Published",
    "Min Players",
    "Max Players",
    "Play Time",
    "Min Age",
    "Owned Users"
]

# === Категориальные признаки ===
cat_features = ["Domains"]

# Удаляем строки без числовых значений
data = data.dropna(subset=num_features)

# Заполняем NaN в категориальных
data[cat_features] = data[cat_features].fillna("Unknown")

# === Формируем X и y ===
X = data[num_features + cat_features]
y = data[target]

# === Делим на train/test ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# === Модель CatBoost, которая сама работает со строками ===
model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    random_seed=42,
    loss_function="MAE",
    verbose=0
)

model.fit(X_train, y_train, cat_features=[X.columns.get_loc(c) for c in cat_features])

# === Оценка ===
pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, pred))

# === Функция предсказания ===
def predict_game(Year, MinP, MaxP, Time, Age, Owned, Domain):
    df = pd.DataFrame([{
        "Year Published": Year,
        "Min Players": MinP,
        "Max Players": MaxP,
        "Play Time": Time,
        "Min Age": Age,
        "Owned Users": Owned,
        "Domains": Domain
    }])

    rating = model.predict(df)[0]
    print("Предсказанный рейтинг:", rating)
    return rating

# Пример
predict_game(2015, 4, 12, 0, 1231, 2, "Strategy Games")
