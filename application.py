from flask import Flask,render_template,redirect,url_for,request
import pickle
import numpy as np
import pandas as pd
import re
import warnings
from predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

@application.route('/')
def index():
	return render_template('home.html')

@application.route('/home')
def home():
	return render_template('home.html')


@application.route('/gmppredict', methods=['GET','POST'])
def gmppredict():
    if request.method == "POST":
        
        na = request.form.get('na')
        data = CustomData(           
            carat = float(request.form.get('carat')),
            depth = float(request.form.get('depth')),
            table = float(request.form.get('table')),
            x = float(request.form.get('x')),
            y = float(request.form.get('y')),
            z = float(request.form.get('z')),
            cut = request.form.get('cut'),
            color= request.form.get('color'),
            clarity = request.form.get('clarity')
        )

        pred_df = data.get_data_as_dataframe()
        #print(pred_df)

        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(pred_df)
        result = round(pred[0],2)
        #print(result)

        return render_template('gmpshow.html',name=na,result=result)
    return render_template('gmp.html')




if __name__ == '__main__':
	application.run(debug=True)

