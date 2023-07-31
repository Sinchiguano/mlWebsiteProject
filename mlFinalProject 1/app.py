from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('model.pkl')  # Load the pre-trained model


@app.route('/')
def index():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    new_data = [[petal_length, petal_width]]
    prediction = model.predict(new_data)


    from sklearn import preprocessing
    labels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    decipher=preprocessing.LabelEncoder()
    decipher.fit(labels)
    predictionLabel = decipher.inverse_transform(prediction)






    return render_template('output.html', prediction=predictionLabel)

if __name__ == '__main__':
    app.run()
