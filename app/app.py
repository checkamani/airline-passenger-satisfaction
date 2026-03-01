from flask import Flask, request, render_template
import os
import joblib
import urllib.request
import pandas as pd

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

# Environment (optional)
MODEL_URL = os.environ.get("MODEL_URL", "").strip()

# Global model holder
model = None


def try_load_model():
    """
    Never raises.
    Tries local file first.
    If missing and MODEL_URL exists, downloads then loads.
    Returns (model_or_none, message)
    """
    global model
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1) local exists
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            return model, "Model loaded from local file."
        except Exception as e:
            model = None
            return None, f"Model file exists but failed to load: {e}"

    # 2) download if URL provided
    if MODEL_URL.startswith("http"):
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            model = joblib.load(MODEL_PATH)
            return model, "Model downloaded and loaded."
        except Exception as e:
            model = None
            return None, f"MODEL_URL provided but download/load failed: {e}"

    # 3) nothing available
    model = None
    return None, "Model not found. Upload model/model.pkl OR set MODEL_URL in Heroku config vars."


# Load once on startup (safe)
_, startup_message = try_load_model()


@app.route("/", methods=["GET"])
def home():
    # Always render the page; show a message if model missing
    return render_template("index.html", model_status=startup_message)


@app.route("/predict", methods=["POST"])
def predict():
    global model

    # If model is missing, try loading again (maybe it was added later)
    if model is None:
        _, msg = try_load_model()
        return render_template("index.html", error=msg, model_status=msg), 200

    try:
        form_data = dict(request.form)

        # ---- Convert inputs ----
        numeric_fields = [
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

        # Build base row
        row = {}
        for f in numeric_fields:
            row[f] = float(form_data.get(f, 0) or 0)

        # ---- One-hot categorical (match what your model was trained with) ----
        # Your error earlier showed the model expects these one-hot columns:
        # Class_Eco, Class_Eco Plus, Customer Type_disloyal Customer, Gender_Male, Type of Travel_Personal Travel

        gender = form_data.get("Gender", "Female")
        cust_type = form_data.get("Customer Type", "Loyal Customer")
        travel_type = form_data.get("Type of Travel", "Business travel")
        travel_class = form_data.get("Class", "Business")

        row["Gender_Male"] = 1 if gender == "Male" else 0
        row["Customer Type_disloyal Customer"] = 1 if cust_type == "disloyal Customer" else 0
        row["Type of Travel_Personal Travel"] = 1 if travel_type == "Personal Travel" else 0

        row["Class_Eco"] = 1 if travel_class == "Eco" else 0
        row["Class_Eco Plus"] = 1 if travel_class == "Eco Plus" else 0

        X = pd.DataFrame([row])

        pred = model.predict(X)[0]
        label = "Satisfied" if int(pred) == 1 else "Neutral or Dissatisfied"

        return render_template("index.html", prediction=label, model_status="Model OK")

    except Exception as e:
        return render_template("index.html", error=str(e), model_status="Model OK"), 200
