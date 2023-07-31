from flask import Flask, render_template, request, redirect, session
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.secret_key = 'sinchi'

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train a logistic regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

@app.route('/')
def login():
    if 'username' in session:
        return redirect('/main')
    return render_template('login.html')

@app.route('/authenticate', methods=['POST'])
def authenticate():
    username = request.form['username']
    password = request.form['password']

    # Add your authentication logic here
    # Example: check if the username and password match
    if username == 'sinchiguano' and password == 'sinchiguano':
        session['username'] = username
        return redirect('/main')
    else:
        return redirect('/')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect('/')

@app.route('/main')
def main():
    if 'username' not in session:
        return redirect('/')

    return render_template('main.html')

@app.route('/result', methods=['POST'])
def result():
    if 'username' not in session:
        return redirect('/')

    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Make prediction using the trained model
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)[0]
    target_names = iris.target_names[prediction]

    return render_template('result.html', target_names=target_names)

if __name__ == '__main__':
    app.run(debug=True)
