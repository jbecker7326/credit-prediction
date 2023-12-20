import pickle

from flask import Flask
from flask import request
from flask import jsonify

# data preparation pipeline
import pandas as pd

# open deployed svm model
model_path = f"model.bin"
with open(model_path, 'rb') as f_in:
    dv, model = pickle.load(f_in)

# route app to host port
app = Flask('credit_default')
@app.route('/predict', methods=['POST'])


def get_request():
    # parse json to dict
    data = request.get_json()
    data = dv.transform([data])
    result = predict(data)
    
    # return result as json
    return jsonify(result)

def predict(data):
    # binary classification response whether person will default on credit
    y_pred = model.predict(data)
    default = y_pred >= 0.5
    result = {
        'default_probability': float(y_pred),
        'credit_default': bool(default)
    }

    return result

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

if __name__ == "__main__":
    main()