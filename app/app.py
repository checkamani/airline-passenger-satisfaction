from flask import Flask, request, render_template
import joblib
import os
# create app FIRST
app = Flask(__name__)

# load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "model.pkl")
model = joblib.load(MODEL_PATH)


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

        result = "Satisfied Passenger "if prediction == 1 else "Neutral/Dissatisfied Passenger"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=str(e))


if __name__ == "__main__":
    app.run(debug=True)