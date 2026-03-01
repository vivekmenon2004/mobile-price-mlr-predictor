import pandas as pd
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

# Load and prepare the transformer (matching MLR.ipynb logic)
def prepare_transformer():
    try:
        dataset = pd.read_excel('Mobile_MLR_Dataset.xlsx')
        X = dataset.drop('Price', axis=1)
        
        # Consistent with MLR.ipynb: Brand is at index 0
        ct = ColumnTransformer(
            transformers=[('encoder', OneHotEncoder(drop='first'), ['Brand'])],
            remainder='passthrough'
        )
        ct.fit(X)
        return ct, sorted(dataset['Brand'].unique().tolist())
    except Exception as e:
        print(f"Error preparing transformer: {e}")
        return None, []

# Load model and prepare transformer
transformer, brands = prepare_transformer()
model = pickle.load(open('mobile_price_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html', brands=brands)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Construct input dataframe for the transformer
        input_data = pd.DataFrame([{
            'Brand': data['brand'],
            'RAM_GB': data['ram'],
            'Storage_GB': data['storage'],
            'Battery_mAh': data['battery'],
            'Camera_MP': data['camera'],
            'Display_Inch': data['display'],
            'Processor_GHz': data['processor'],
            'Brand_Rating': data['rating']
        }])
        
        # Transform the features
        X_transformed = transformer.transform(input_data)
        
        # Predict
        prediction = model.predict(X_transformed)
        
        # Return result
        return jsonify({
            'success': True,
            'price': float(prediction[0])
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
