from flask import Flask, request, jsonify, render_template
import pickle
import traceback
import pandas as pd
import numpy as np

with open("C:/Users/willjdsouza/Flask2/model/hockey_model.pkl", 'rb') as f:
    model = pickle.load(f)
#model_columns = joblib.load("C:/Users/willjdsouza/Flask2/model_columns.pkl") # Load

app = Flask(__name__, template_folder='templates')

@app.route("/", methods = ['GET', 'POST'])
def main():
    if request.method == 'GET':
        return render_template('main.html')
    if request.method == 'POST':
        GF_GP = request.form['GF/GP']
        GA_GP = request.form['GA/GP']
        PP = request.form['PP%']
        PK = request.form['PK%']
        S_GP = request.form['S/GP']

        input_variables = pd.DataFrame([[GF_GP, GA_GP, PP, PK, S_GP]],
                                       columns=['GF/GP', 'GA/GP', 'PP%', 'PK%', 'S/GP'],
                                       dtype = float,
                                       index = ['input'])

        prediction = model.predict(input_variables)[0]

        return render_template('main.html',
                                 original_input={'GF/GP': GF_GP,
                                                 'GA/GP': GA_GP,
                                                 'PP%': PP,
                                                 'PK%': PK,
                                                 'S/GP': S_GP},
                                 result = (prediction / 100) * 82,
                                     )

if __name__ == '__main__':
    app.run()
