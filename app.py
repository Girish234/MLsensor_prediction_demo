from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained pipeline/model (trained with sklearn 1.7.2)
with open("final_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        data = {
            "Air temperature [K]": float(request.form["air"]),
            "Process temperature [K]": float(request.form["process"]),
            "Rotational speed [rpm]": float(request.form["rpm"]),
            "Torque [Nm]": float(request.form["torque"]),
            "Tool wear [min]": float(request.form["wear"]),
            "Type": request.form["type"]
        }

        df = pd.DataFrame([data])
        pred = model.predict(df)[0]

        prediction = "⚠️ Failure Predicted" if pred == 1 else "✅ No Failure"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
