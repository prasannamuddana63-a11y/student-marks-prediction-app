from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("model.h5")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        hours = float(request.form["hours"])

        result = model.predict(np.array([hours]))

        prediction = round(result[0][0], 2)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000, debug=True)