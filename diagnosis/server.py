import joblib
import pandas as pd
import pre_processing
from flask import Flask, jsonify, request

app = Flask(__name__)

model = joblib.load('Models/LR/LR_stem_tfidf.pkl')


@app.route('/api/diagnosis', methods=['POST', 'OPTIONS'])
def diagnosis():
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
        }
        return '', 204, headers

    user_input = request.get_json()
    if not user_input:
        return jsonify({'error': 'Missing parameters'}), 400
    else:
        data = pd.DataFrame(columns=['User_input'])
        data.columns = data.columns.astype(str)
        data['User_input'] = [user_input]
        processed_data = pre_processing.tf_stemming(data)
        prediction = model.predict(processed_data)
        disease = prediction[0]
        response = jsonify({'diagnosis': disease})
    return response


if __name__ == '__main__':
    app.run(debug=False)
