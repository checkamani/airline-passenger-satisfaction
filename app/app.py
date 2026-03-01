from flask import Flask, request, render_template
import joblib
import os
import urllib.request
import pandas as pd

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

MODEL_URL = os.environ.get("MODEL_URL", "").strip()

os.makedirs(MODEL_DIR, exist_ok=True)


def try_load_model():
    if not os.path.exists(MODEL_PATH) and MODEL_URL.startswith("http"):
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        except Exception:
            return None

    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None

    return None


model = try_load_model()


def ensure_model_loaded():
    global model
    if model is None:
        model = try_load_model()
    return model is not None


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


@app.route("/", methods=["GET"])
def home():
    if not ensure_model_loaded():
        return (
            "App is running but the model is missing. "
            "Either include 'model/model.pkl' in the container or set MODEL_URL in Heroku Config Vars."
        )
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if not ensure_model_loaded():
        return (
            "Model not loaded. Include 'model/model.pkl' or set MODEL_URL, then restart the dyno.",
            503,
        )

    form_data = dict(request.form)

    numeric_cols = [
        "Age",
        "Flight Distance",
        "Departure Delay (min)",
        "Arrival Delay (min)",
        "Inflight wifi service",
        "Departure/Arrival time convenient",
        "Ease of Online booking",
        "Gate location",
        "Food and drink",
        "Online boarding",
        "Seat comfort",
        "Inflight entertainment",
        "On-board service",
        "Leg room service",
        "Baggage handling",
        "Checkin service",
        "Inflight service",
        "Cleanliness",
    ]

    X = {}

    for c in numeric_cols:
        X[c] = safe_float(form_data.get(c, ""), 0.0)

    gender = (form_data.get("Gender") or "").strip()
    cust_type = (form_data.get("Customer Type") or "").strip()
    travel_type = (form_data.get("Type of Travel") or "").strip()
    travel_class = (form_data.get("Class") or "").strip()

    X["Gender_Male"] = 1 if gender.lower() == "male" else 0
    X["Customer Type_disloyal Customer"] = 1 if cust_type.lower().startswith("disloyal") else 0
    X["Type of Travel_Personal Travel"] = 1 if "personal" in travel_type.lower() else 0

    X["Class_Eco"] = 1 if travel_class.lower() in ["eco", "economy"] else 0
    X["Class_Eco Plus"] = 1 if "plus" in travel_class.lower() else 0

    X_df = pd.DataFrame([X])

    try:
        pred = model.predict(X_df)[0]
        label = "Satisfied" if int(pred) == 1 else "Neutral or Dissatisfied"
        return render_template("index.html", prediction=label)
    except Exception as e:
        return render_template("index.html", error=str(e)), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
