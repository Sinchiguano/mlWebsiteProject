# Import the Flask module
from flask import Flask
from flask import render_template

# Create an instance of the Flask class
app = Flask(__name__)

# Define a route and a view function for the root URL '/'
@app.route('/')
def hello_world():
    return render_template('index.html')




# Route for the '/about' URL
@app.route('/about')
def about_page():
    return render_template('about.html')

# Route for the '/contact' URL
@app.route('/contact')
def contact_page():
    return render_template('contact.html') 
    
       

# Run the Flask application if this file is the main program
if __name__ == '__main__':
    app.run()
