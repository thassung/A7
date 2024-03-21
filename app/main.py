from flask import Flask, render_template, request, jsonify
from script import *

app = Flask(__name__, template_folder='templates')

@ app.route('/', methods=['POST', 'GET'])
def main():
    if request.method == 'GET':
        return render_template('index.html')

@ app.route('/answer', methods=['POST', 'GET'])
def model_inference():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        inputData = request.json
        query = inputData['prompt'] 

        response = query_to_response(query)
        response = {'response': response}

        return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

