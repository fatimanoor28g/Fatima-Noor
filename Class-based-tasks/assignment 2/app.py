from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Trained model load karein
with open("churn_model.pkl", "rb") as file:
    model = pickle.load(file)


# Frontend page show karna
@app.route("/")
def home():
    return render_template("index.html")


# Customer churn prediction
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    age = data["Age"]
    monthly_spend = data["MonthlySpend"]
    city = data["City"]
    contract = data["Contract"]
    join_year = data["JoinYear"]
    join_month = data["JoinMonth"]

    input_data = pd.DataFrame([{
        "Age": age,
        "MonthlySpend": monthly_spend,
        "City_Karachi": int(city == "Karachi"),
        "City_Lahore": int(city == "Lahore"),
        "Contract_Yearly": int(contract == "Yearly"),
        "JoinYear": join_year,
        "JoinMonth": join_month
    }])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        result = "Customer may churn"
    else:
        result = "Customer may stay"

    return jsonify({
        "prediction": int(prediction),
        "result": result
    })


if __name__ == "__main__":
    app.run(debug=True)