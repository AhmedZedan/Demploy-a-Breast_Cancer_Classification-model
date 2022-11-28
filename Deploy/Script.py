import pandas as pd
import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/predict', methods=['post'])

def predict():
    req = request.get_json()
    input_data = req['data']
    input_data_df = pd.DataFrame.from_dict(input_data)
    
    model = joblib.load('model.pkl')
    scale_obj = joblib.load('scaler.pkl')

    input_data_scaled = scale_obj.transform(input_data_df)

    print(input_data_scaled)
    
    prediction = model.predict(input_data_scaled)
    
    if prediction[0] == 1:
        cancer_type = 'Malignant'
    else:
        cancer_type = 'Benign'
        
    return jsonify({'output':{'Cancer_type':cancer_type}})


@app.route('/')
def home():
    return "Welcome to Breast cancer diagnostic center"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000)