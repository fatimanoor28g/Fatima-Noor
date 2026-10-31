import pandas as pd
import numpy as np
data = {
"CustomerID": [101, 102, 103, 104, 105, 105],
"Age": [25, 40, np.nan, 35, 120, 120],
"MonthlySpend": [2500, 5200, 3100, np.nan, 4500, 4500],
"City": ["Lahore", "Karachi", "lahore", "Islamabad", "Karachi", "Karachi"],
"Contract": ["Monthly", "Yearly", "Monthly", "Yearly", None, None],
"JoinDate": ["2025-01-10", "2024-05-15", "2025-03-20",
"2023-11-05", "2024-09-01", "2024-09-01"],
"Churn": ["No", "No", "Yes", "No", "Yes", "Yes"]
}
df = pd.DataFrame(data)
# print(df)
# print(df.shape)

# print(df.info())

df = df.drop_duplicates().reset_index(drop=True)

# print(df.isnull().sum())

# print(df["Churn"].value_counts())
df["Age"] = df["Age"].fillna(df["Age"].median())

df["MonthlySpend"] = df["MonthlySpend"].fillna(df["MonthlySpend"].median())

df["Contract"] = df["Contract"].fillna(df["Contract"].mode()[0])
# print(df.isnull().sum())
# print(df.dtypes)
# print(df["Age"].describe())
df.loc[df["Age"] > 100, "Age"] = np.nan

df["Age"] = df["Age"].fillna(df["Age"].median())
print(df["Age"])
df_no = df[df["Churn"] == "No"].copy()
df_yes = df[df["Churn"] == "Yes"].copy()
# print(df.shape)
# print(df.duplicated().sum())
df_no = df[df["Churn"] == "No"].copy()
df_yes = df[df["Churn"] == "Yes"].copy()
df_no = df[df["Churn"] == "No"].copy()
df_yes = df[df["Churn"] == "Yes"].copy()

print("No class:", df_no.shape)
print("Yes class:", df_yes.shape)


from ctgan import CTGAN
df_no_ctgan = df_no.drop(columns=["CustomerID", "Churn"])
print(df_no_ctgan)
from ctgan import CTGAN

ctgan_no = CTGAN(epochs=300)

ctgan_no.fit(
    df_no_ctgan,
    discrete_columns=["City", "Contract", "JoinDate"]
)
from ctgan import CTGAN

ctgan_no = CTGAN(epochs=100)
ctgan_no.fit(
    df_no_ctgan,
    discrete_columns=["City", "Contract", "JoinDate"]
)

print("CTGAN No class training complete")
synthetic_no = ctgan_no.sample(3000)

print(synthetic_no.shape)
print(synthetic_no.head())
synthetic_no["CustomerID"] = range(1001, 4001)
synthetic_no["Churn"] = "No"

synthetic_no = synthetic_no[
    ["CustomerID", "Age", "MonthlySpend", "City", "Contract", "JoinDate", "Churn"]
]

print(synthetic_no.shape)
print(synthetic_no.head())
df_yes_ctgan = df_yes.drop(columns=["CustomerID", "Churn"])

print(df_yes_ctgan)
from ctgan import CTGAN

ctgan_yes = CTGAN(epochs=100)

ctgan_yes.fit(
    df_yes_ctgan,
    discrete_columns=["City", "Contract", "JoinDate"]
)

print("CTGAN Yes class training complete")
synthetic_yes = ctgan_yes.sample(2000)

print(synthetic_yes.shape)
print(synthetic_yes.head())
synthetic_yes["CustomerID"] = range(4001, 6001)
synthetic_yes["Churn"] = "Yes"

synthetic_yes = synthetic_yes[
    ["CustomerID", "Age", "MonthlySpend", "City", "Contract", "JoinDate", "Churn"]
]

print(synthetic_yes.shape)
print(synthetic_yes.head())
df_final = pd.concat(
    [synthetic_no, synthetic_yes],
    ignore_index=True
)

print(df_final.shape)
print(df_final["Churn"].value_counts())
df_final = pd.get_dummies(
    df_final,
    columns=["City", "Contract"],
    drop_first=True
)

df_final["Churn"] = df_final["Churn"].map({
    "No": 0,
    "Yes": 1
})

print(df_final.head())
print(df_final.dtypes)
print(df_final.columns)
df_final["City_Lahore"] = (
    df_final["City_Lahore"] | df_final["City_lahore"]
)

df_final = df_final.drop(columns=["City_lahore"])

print(df_final.columns)

df_final["JoinDate"] = pd.to_datetime(df_final["JoinDate"])
print(df_final.dtypes)
df_final["JoinYear"] = df_final["JoinDate"].dt.year
df_final["JoinMonth"] = df_final["JoinDate"].dt.month
print(df_final.head())
df_final = df_final.drop(columns=["CustomerID", "JoinDate"])
print(df_final.head())
print(df_final.columns)
X = df_final.drop("Churn", axis=1)
y = df_final["Churn"]
print(X.shape)
print(y.shape)
print(y.value_counts())
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(
    random_state=42
)

model.fit(X_train, y_train)

print("Model training complete")
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(y_pred[:10])
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
import pickle

with open("churn_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully")