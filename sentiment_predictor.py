from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load(r"C:\Users\HP\Downloads\sentimental_analysis_internship\models\naive_bayes.pkl")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        prediction = model.predict([message])[0]
        return render_template('prediction.html', prediction=prediction)

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
