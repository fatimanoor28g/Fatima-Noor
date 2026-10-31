from data import df
import pandas as pd 
import numpy as np
# print(df.head())
# print(df.tail())
# print(df.shape)
# print(df.describe())
# print(df.describe(include="object"))
# print(df.isnull().sum())
# print(df.nunique())
# print(df["City"].value_counts(dropna=False))
# print(df.isna().sum())
# missing_percent = df.isna().mean() * 100
# print(missing_percent)

# Median for Age
df["Age"] = df["Age"].fillna(df["Age"].median())
# Mean for MonthlySpend
df["MonthlySpend"] = df["MonthlySpend"].fillna(
df["MonthlySpend"].mean()
)
# Mode for Contract
df["Contract"] = df["Contract"].fillna(
df["Contract"].mode()[0]
)
# print(df.isnull().sum())

# print(df.duplicated().sum())
df = df.drop_duplicates()
# print(df)

df["City"] = df["City"].str.strip().str.title()
df["Contract"] = df["Contract"].str.strip().str.title()
# print(df["City"].unique())

df["JoinDate"] = pd.to_datetime(df["JoinDate"], errors="coerce")
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
# print(df.dtypes)

print(df["Age"].describe())
# Simple domain rule for demonstration
invalid_age = (df["Age"] < 0) | (df["Age"] > 100)
# print(df.loc[invalid_age])

df.loc[invalid_age, "Age"] = np.nan
df["Age"] = df["Age"].fillna(df["Age"].median())
# print(df)
df["JoinYear"] = df["JoinDate"].dt.year
df["JoinMonth"] = df["JoinDate"].dt.month
df["JoinDay"] = df["JoinDate"].dt.day

reference_date = pd.Timestamp("2026-09-14")
df["CustomerDays"] = (reference_date - df["JoinDate"]).dt.days
df["SpendPerDay"] = (
df["MonthlySpend"] / 30
)
df["AgeGroup"] = pd.cut(
df["Age"],
bins=[0, 25, 40, 60, 100],
labels=["Young", "Adult", "Middle", "Senior"],
include_lowest=True
)
print(df)
df["ChurnEncoded"] = df["Churn"].map({
"No": 0,
"Yes": 1
})
print(df)
city_encoded = pd.get_dummies(
df["City"],
prefix="City",
dtype=int
)
df = pd.concat([df, city_encoded], axis=1)
print(df)
priority_order = {
"Low": 0,
"Medium": 1,
"High": 2
}
# df["PriorityEncoded"] = df["Priority"].map(priority_order)
print(df)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[["Age_scaled", "Spend_scaled"]] = scaler.fit_transform(
df[["Age", "MonthlySpend"]]
)
print(df)
from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
df[["Age_0_1", "Spend_0_1"]] = minmax.fit_transform(
df[["Age", "MonthlySpend"]]
)
print(df)
from sklearn.preprocessing import RobustScaler
robust = RobustScaler()
df[["Spend_robust"]] = robust.fit_transform(
df[["MonthlySpend"]]
)
print(df)
from sklearn.model_selection import train_test_split
X = df[["Age", "MonthlySpend", "City", "Contract"]]
y = df["ChurnEncoded"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.5,
    random_state=42,
    stratify=y
)
print(df)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
numeric_features = ["Age", "MonthlySpend"]
categorical_features = ["City", "Contract"]
numeric_pipeline = Pipeline(steps=[
("imputer", SimpleImputer(strategy="median")),
("scaler", StandardScaler())
])
categorical_pipeline = Pipeline(steps=[
("imputer", SimpleImputer(strategy="most_frequent")),
("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(
transformers=[
("num", numeric_pipeline, numeric_features),
("cat", categorical_pipeline, categorical_features)
]
)
X_train_ready = preprocessor.fit_transform(X_train)
X_test_ready = preprocessor.transform(X_test)
print(X_train_ready.shape)
print(X_test_ready.shape)
print(df)

