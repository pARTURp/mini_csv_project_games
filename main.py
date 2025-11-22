import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error

# === Загружаем файл ===
data = pd.read_csv("Games.csv", encoding="cp1251")

# === Целевая колонка ===
target = "Rating Average"

# Удаляем строки без рейтинга
data = data[data[target].notna()]

# === Числовые и категориальные признаки ===
num_features = [
    "Year Published",
    "Min Players",
    "Max Players",
    "Play Time",
    "Min Age",
    "Users Rated",
    "BGG Rank",
    "Owned Users"
]

cat_features = ["Domains"]

# Удаляем строки, где отсутствуют числовые признаки
data = data.dropna(subset=num_features)

# Заполняем пропуски в категориальных
data[cat_features] = data[cat_features].fillna("Unknown")

# === Разделение X и y ===
X = data[num_features + cat_features]
y = data[target]

# === Препроцессинг ===
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
    ]
)

# Fit + transform
X_processed = preprocess.fit_transform(X)

# === Train/Test ===
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.1, random_state=42
)

# === Модель CatBoost ===
model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    random_seed=42,
    verbose=0
)

model.fit(X_train, y_train)

# === Оценка ===
pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, pred))

# === Функция предсказания ===
def predict_game(Year, MinP, MaxP, Time, Age, UsersRated, Rank, Owned, Domain):
    df = pd.DataFrame([{
        "Year Published": Year,
        "Min Players": MinP,
        "Max Players": MaxP,
        "Play Time": Time,
        "Min Age": Age,
        "Users Rated": UsersRated,
        "BGG Rank": Rank,
        "Owned Users": Owned,
        "Domains": Domain
    }])

    df_prep = preprocess.transform(df)
    rating = model.predict(df_prep)[0]
    print("Предсказанный рейтинг:", rating)
    return rating

# Пример
predict_game(2010, 1, 1, 1, 1, 1, 1, 1, "Family Games, Strategy Games")
