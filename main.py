import flask 
from flask import Flask,render_template,request
from utils import restaurant_review

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_data',methods = ['GET','POST'])
def get_data():
    data = request.form
    class_object = restaurant_review(data)
    result = class_object.cleaned_data()
    if result == 0:
        return render_template('index.html',prediction = 'Negative Sentiment')
    else :
         return render_template('index.html',prediction = 'Positive Sentiment')


if __name__ == "__main__":
    app.run(port = 8080,debug = True)