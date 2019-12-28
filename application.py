from flask import Flask,render_template,request
from sklearn.externals import joblib
import numpy as np
from waitress import serve

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")


@app.route("/ml", methods=["POST"])
def result():
    try:
        form = request.form
        model = joblib.load("mlmodel/ECommerce_LinearRegressionModel.pkl")

        session = float(form['session'])
        app = float(form['app'])
        web = float(form['web'])
        length = float(form['length'])

        
        new_user = np.array(
            [session,app,web,length]).reshape(1, -1)

        predicted_price = model.predict(new_user)
        
        return render_template("result.html", price=float(predicted_price))

    except ValueError:
        return render_template("error.html")

if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=8000)