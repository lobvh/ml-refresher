import pickle

from flask import Flask
from flask import request
from flask import jsonify

import os


model_file = 'model_C=1.0.bin'

#From the official docs:
#EXPOSE - While EXPOSE can be used for local testing, it is not supported in Heroku’s container runtime.
#Instead your web process/code should get the $PORT environment variable.

port = int(os.environ.get("PORT", 9696)) #Either use Heroku's default $PORT environment variable or use the 9696

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in) 

app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello_world():
    statement = 'Hello World!'
    return statement

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json() 

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]

    churn = y_pred >= 0.5

    result = {
        "churn_probability": float(y_pred),
        "churn": bool(churn)
    }

    return jsonify(result) 


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port)
