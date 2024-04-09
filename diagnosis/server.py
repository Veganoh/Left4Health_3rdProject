import pre_processing
import image_processor 
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
import os
app = Flask(__name__)


@app.route('/api/diagnosis/text', methods=['POST', 'OPTIONS'])
def diagnosis():
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
        }
        return '', 204, headers

    user_input_json = request.get_json()
    if not user_input_json or 'User_input' not in user_input_json:
        return jsonify({'error': 'Missing or invalid parameters'}), 400
    else:
        user_input = user_input_json['User_input']
        disease = pre_processing.runModel(user_input)
        response = jsonify({'diagnosis': disease})
        response.headers.add('Access-Control-Allow-Origin', '*')
    return response


UPLOAD_FOLDER = 'Uploads'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/api/diagnosis/image', methods=['POST', 'OPTIONS'])
def diagnosis_image():
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
        }
        return '', 204, headers

    if 'image' not in request.files:
        response = jsonify({'error': 'No file part'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 400

    file = request.files['image']

    if file.filename == '':
        response = jsonify({'error': 'No selected file'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 400

    if file:
        filename = secure_filename(file.filename)
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Chamar o método process_image() da classe ImageProcessor
        result = image_processor.process_image(filepath)
        response = jsonify({'diagnosis': result})
    else:
        response = jsonify({'error': 'Invalid file format'})

    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/')
def index():
    return 'Hello, world!'

if __name__ == '__main__':
    app.run(debug=False)
