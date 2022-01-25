# importing the necessary dependencies
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            Temperature=float(request.form['Temperature'])
            TDS = float(request.form['TDS'])
            PH = float(request.form['PH'])
            model = open('do.pkl', 'rb')
            forest = pickle.load(model)
            model2 = open('standard.pkl', 'rb')
            forest2 = pickle.load(model2)
            # predictions using the loaded model file
            b = np.array([[Temperature, TDS, PH]])
            sb = pd.DataFrame(forest2.transform(b))
            prediction=forest.predict(sb)
            print('prediction is', prediction)
            # showing the prediction results in a UI
            return render_template('results.html',prediction=prediction[0])
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')



if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
	app.run(debug=True) # running the app