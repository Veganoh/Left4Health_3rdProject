import pre_processing
from flask import Flask, jsonify, request
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


from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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
        response = jsonify({'error': 'No image part'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 400

    file = request.files['image']



    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        response = jsonify({'message': 'File uploaded successfully'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 200
    else:
        response = jsonify({'error': 'Invalid file type'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 400



if __name__ == '__main__':
    app.run(debug=False)
