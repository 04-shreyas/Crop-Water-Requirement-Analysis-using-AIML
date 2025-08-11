from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib as jb

app = Flask(__name__)

# Load trained Random Forest model
model = jb.load("catboost_model.pkl")

# Valid options
crop_options = ["Wheat", "Rice", "Maize", "Sugarcane", "Cotton", "Groundnut", "Barley", "Soybean", "Sorghum", "Millet"]

# Crop Coefficient (Kc)
kc_values = {
    "Wheat": 1.05, "Rice": 1.15, "Maize": 1.10, "Sugarcane": 1.20,
    "Cotton": 1.15, "Groundnut": 1.00, "Barley": 0.95, "Soybean": 1.05,
    "Sorghum": 0.90, "Millet": 0.85
}

# Irrigation recommendation
def get_irrigation_method(crop):
    if crop == "Rice":
        return "Use flood irrigation — ideal for paddy fields."
    elif crop in ["Sugarcane", "Cotton", "Groundnut"]:
        return "Use drip irrigation for efficient water use."
    elif crop in ["Millet", "Sorghum"]:
        return "Use sprinkler irrigation — suitable for dry regions."
    else:
        return "Use surface or sprinkler irrigation based on field layout."

@app.route("/")
def index():
    return render_template("index.html", crops=crop_options)

@app.route("/result", methods=["POST"])
def result():
    try:
        form = request.form

        # Get and convert inputs
        crop = form["crop"]
        temperature = float(form["temperature"])
        humidity = float(form["humidity"])
        rainfall = float(form["rainfall"])
        wind_speed = float(form["wind_speed"])
        solar_rad = float(form["solar_rad"])
        precipitation = float(form["precipitation"])

        # Encode crop
        crop_encoded = crop_options.index(crop)

        # Compute ET using standard formula
        ET = (0.0023 * (temperature + 17.8) * np.sqrt(solar_rad) * (1 - humidity / 100)) + (0.1 * wind_speed)
        kc = kc_values.get(crop, 1.0)

        # Prepare input for model
        input_features = [[
            precipitation, wind_speed, ET, crop_encoded, rainfall, humidity
        ]]
        columns = ["Precipitation (mm)", "Wind Speed (km/h)", "Evapotranspiration (ET)",
                   "Crop", "Rainfall (mm)", "Humidity (%)"]
        df_input = pd.DataFrame(input_features, columns=columns)

        # Predict water requirement
        mm_day = model.predict(df_input)[0]
        liters_per_day = mm_day * kc * 10000
        prediction = round(liters_per_day, 2)

        # Render results
        return render_template("result.html",
                               crop=crop,
                               temperature=temperature,
                               humidity=humidity,
                               rainfall=rainfall,
                               wind_speed=wind_speed,
                               solar_rad=solar_rad,
                               precipitation=precipitation,
                               ET=round(ET, 2),
                               kc=kc,
                               prediction=prediction,
                               interval="every 2–4 days",
                               method=get_irrigation_method(crop))
    except Exception as e:
        print("Error:", e)
        return "Invalid input: " + str(e)

if __name__ == "__main__":
    app.run(debug=True)
