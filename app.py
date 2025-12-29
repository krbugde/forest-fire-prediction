from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

## import ridge regressor and standars scaler pickle

ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))

## creating home page
@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Temperature=float(request.form.get('temperature'))
        RH=float(request.form.get('rh'))
        Ws=float(request.form.get('ws'))
        Rain=float(request.form.get('rain'))
        FFMC=float(request.form.get('ffmc'))
        DMC=float(request.form.get('dmc'))
        ISI=float(request.form.get('isi'))
        Classes=float(request.form.get('classes'))
        Region=float(request.form.get('region'))

        new_scaled_data=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_scaled_data) ## would be in form of list

        return render_template('home.html',results=result[0])
    else:
        return render_template('home.html')


if __name__ == "__main__": ## entry point for execution
    app.run(debug=True)
