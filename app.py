from flask import Flask, request, render_template
import pandas as pd
import pickle
import os
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Ensure the correct paths to model and preprocessor
MODEL_PATH = os.path.join(os.getcwd(), 'model.pkl')
PREPROCESSOR_PATH = os.path.join(os.getcwd(), 'preprocessor.pkl')

# Load model and preprocessor
try:
    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)
    logging.debug('Model loaded successfully')
except FileNotFoundError:
    logging.error(f'Model file not found at {MODEL_PATH}')
except Exception as e:
    logging.error(f'Error loading model: {str(e)}')

try:
    with open(PREPROCESSOR_PATH, 'rb') as preprocessor_file:
        preprocessor = pickle.load(preprocessor_file)
    logging.debug('Preprocessor loaded successfully')
except FileNotFoundError:
    logging.error(f'Preprocessor file not found at {PREPROCESSOR_PATH}')
except Exception as e:
    logging.error(f'Error loading preprocessor: {str(e)}')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input data
        features = ['BedroomAbvGr', 'FullBath', 'YearBuilt', 'Street', 'Neighborhood', 'Condition1']
        input_data = pd.DataFrame(columns=features)
        
        # Ensure all fields are provided
        input_data.at[0, 'BedroomAbvGr'] = int(request.form['BedroomAbvGr'])
        input_data.at[0, 'FullBath'] = int(request.form['FullBath'])
        input_data.at[0, 'YearBuilt'] = 2023 - int(request.form['HouseAge'])  # Convert HouseAge to YearBuilt
        input_data.at[0, 'Street'] = request.form['Street']
        input_data.at[0, 'Neighborhood'] = request.form['Neighborhood']
        input_data.at[0, 'Condition1'] = request.form['Condition1']

        logging.debug(f'Input data: {input_data}')

        # Process and predict
        processed_data = preprocessor.transform(input_data)
        logging.debug(f'Processed data: {processed_data}')

        prediction = model.predict(processed_data)
        logging.debug(f'Prediction: {prediction}')

        return render_template('index.html', prediction_text=f'Estimated House Price: ${prediction[0]:,.2f}')
    
    except Exception as e:
        # Handle errors and return an error message
        logging.error(f'Error occurred: {str(e)}', exc_info=True)
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
