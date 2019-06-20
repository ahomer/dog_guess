import json
import os

from flask import render_template, jsonify
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

from dog_guess import *

UPLOAD_FOLDER = '../data/uploads'


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# index webpage displays cool visuals and receives user input text for model


@app.route('/')
@app.route('/index')
def index():

    # render web page with plotly graphs
    return render_template('master.html')


# web page that handles user query and displays model results
@app.route('/go', methods=['GET', 'POST'])
def go():
    breed = ''
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            filename = os.path.abspath(os.path.join(
                app.config['UPLOAD_FOLDER'], filename))
            print(filename)
            file.save(filename)

    return json.dumps({'msg': 'good for you', 'breed': find_dog(filename), 'ret': 0})


def main():
    app.run(host='0.0.0.0', port=3001, debug=True, threaded=False)


if __name__ == '__main__':
    main()
