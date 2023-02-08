import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pandas

app=Flask(__name__)
## Load the model
regressionmodel=pickle.load(open('boston-regression-model.pkl','rb'))
standardizing=pickle.load(open('standardizing.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=standardizing.transform(np.array(list(data.values())).reshape(1,-1))
    output=regressionmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=standardizing.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regressionmodel.predict(final_input)[0]
    return render_template('home.html',prediction_text="The House Price Prediction is {}".format(output))


if __name__=="__main__":
    app.run(debug=True)