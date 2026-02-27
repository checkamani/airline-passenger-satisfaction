from flask import Flask, request, render_template
import joblib
import os
import urllib.request

# -------------------------------------------------
# CREATE FLASK APP
# -------------------------------------------------
app = Flask(__name__)

# -------------------------------------------------
# MODEL PATH SETUP
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

# 🔴 IMPORTANT:
# Replace this with your PUBLIC model download link
MODEL_URL = "PASTE_MODEL_DOWNLOAD_LINK_HERE"

# create model directory if missing
os.makedirs(MODEL_DIR, exist_ok=True)

# download model if it does not exist
if not os.path.exists(MODEL_PATH):
    if MODEL_URL.startswith("http"):
        print("Downloading model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")
    else:
        raise FileNotFoundError(
            "Model not found and MODEL_URL is not set."
        )

# load model
model = joblib.load(MODEL_PATH)

# -------------------------------------------------
# ROUTES
# -------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        numeric_cols = [
            "Age",
            "Flight Distance",
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
            "Departure Delay in Minutes",
            "Arrival Delay in Minutes"
        ]

        numeric_vals = [float(request.form[c]) for c in numeric_cols]

        gender = request.form["Gender"]
        customer_type = request.form["Customer Type"]
        travel_type = request.form["Type of Travel"]
        seat_class = request.form["Class"]

        gender_male = 1.0 if gender == "Male" else 0.0
        disloyal = 1.0 if customer_type == "disloyal Customer" else 0.0
        personal = 1.0 if travel_type == "Personal Travel" else 0.0
        class_eco = 1.0 if seat_class == "Eco" else 0.0
        class_eco_plus = 1.0 if seat_class == "Eco Plus" else 0.0

        features = numeric_vals + [
            gender_male,
            disloyal,
            personal,
            class_eco,
            class_eco_plus
        ]

        prediction = model.predict([features])[0]

        if isinstance(prediction, (int, float)):
            result = "Satisfied" if prediction >= 0.5 else "Neutral or Dissatisfied"
        else:
            result = str(prediction)

        return render_template("index.html",
                               prediction_text=f"Prediction: {result}")

    except Exception as e:
        return render_template("index.html",
                               prediction_text=f"Error: {str(e)}")


# -------------------------------------------------
# LOCAL RUN (Heroku uses gunicorn)
# -------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
