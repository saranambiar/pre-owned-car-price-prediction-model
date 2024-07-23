from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

try:
    with open('model.pkl', 'rb') as file:
        model, feature_names = pickle.load(file)
    print("Model and feature names loaded successfully")
except Exception as e:
    print(f"Error loading model and feature names: {e}")
    model = None
    feature_names = None

# Dictionaries for encoding input values
brand_dic = {'Audi': '0', 'BMW': '1', 'Datsun': '2', 'Force': '3', 'Ford': '4', 'Honda': '5', 'Hyundai': '6', 'Isuzu': '7', 'Jaguar': '8', 'Jeep': '9', 'Kia': '10', 'Land Rover': '11', 'Lexus': '12', 'MG': '13', 'Mahindra': '14', 'Maruti': '15', 'Mercedes-Benz': '16', 'Mini': '17', 'Nissan': '18', 'Porsche': '19', 'Renault': '20', 'Skoda': '21', 'Tata': '22', 'Toyota': '23', 'Volkswagen': '24', 'Volvo': '25'}
model_dic = {'3': '0', '5': '1', '7': '2', 'A4': '3', 'A6': '4', 'A8': '5', 'Alto': '6', 'Altroz': '7', 'Alturas': '8', 'Amaze': '9', 'Aspire': '10', 'Aura': '11', 'Baleno': '12', 'Bolero': '13', 'C-Class': '14', 'CLS': '15', 'CR': '16', 'CR-V': '17', 'Camry': '18', 'Carnival': '19', 'Cayenne': '20', 'Celerio': '21', 'Ciaz': '22', 'City': '23', 'Civic': '24', 'Compass': '25', 'Cooper': '26', 'Creta': '27', 'D-Max': '28', 'Duster': '29', 'Dzire LXI': '30', 'Dzire VXI': '31', 'Dzire ZXI': '32', 'E-Class': '33', 'ES': '34', 'Ecosport': '35', 'Eeco': '36', 'Elantra': '37', 'Endeavour': '38', 'Ertiga': '39', 'F-PACE': '40', 'Figo': '41', 'Fortuner': '42', 'Freestyle': '43', 'GL-Class': '44', 'GO': '45', 'Glanza': '46', 'Grand': '47', 'Gurkha': '48', 'Harrier': '49', 'Hector': '50', 'Hexa': '51', 'Ignis': '52', 'Innova': '53', 'Jazz': '54', 'KUV': '55', 'KUV100': '56', 'KWID': '57', 'Kicks': '58', 'Marazzo': '59', 'NX': '60', 'Nexon': '61', 'Octavia': '62', 'Polo': '63', 'Q7': '64', 'Rapid': '65', 'RediGO': '66', 'Rover': '67', 'S-Presso': '68', 'S90': '69', 'Safari': '70', 'Santro': '71', 'Scorpio': '72', 'Seltos': '73', 'Superb': '74', 'Swift': '75', 'Swift Dzire': '76', 'Thar': '77', 'Tiago': '78', 'Tigor': '79', 'Triber': '80', 'Tucson': '81', 'Vento': '82', 'Venue': '83', 'Verna': '84', 'Vitara': '85', 'WR-V': '86', 'Wagon R': '87', 'X-Trail': '88', 'X1': '89', 'X3': '90', 'XC': '91', 'XC60': '92', 'XC90': '93', 'XE': '94', 'XF': '95', 'XL6': '96', 'XUV300': '97', 'XUV500': '98', 'Yaris': '99', 'Z4': '100', 'i10': '101', 'i20': '102', 'redi-GO': '103'}
seller_type_dic =  {"Dealer":0,"Individual":1,"Trustmark Dealer":3}
transmission_dic =  {"Automatic":0,"Manual":1}
fuel_dic = {"CNG":0,"Diesel":1,"Electric":2,"LPG":3,"Petrol":4}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    brand = request.form['title']
    model_name = request.form['model']
    seller_type = request.form['Seller']
    transmission_type = request.form['Transmission']
    fuel_type = request.form['Fuel']
    vehicle_age = int(request.form['age'])
    km_driven = int(request.form['km'])
    engine = int(request.form['engine'])
    seats = int(request.form['seats'])
    mileage = float(request.form['mileage'])

   # Encode the form values
    new_data_encoded = {
        'brand_encoded': int(brand_dic.get(brand, -1)),
        'model_encoded': int(model_dic.get(model_name, -1)),
        'seller_type_encoded': seller_type_dic.get(seller_type, -1),
        'transmission_type_encoded': transmission_dic.get(transmission_type, -1),
        'fuel_type_encoded': fuel_dic.get(fuel_type, -1),
        'vehicle_age': vehicle_age,
        'km_driven': km_driven,
        'engine': engine,
        'seats': seats,
        'mileage': mileage
    
    }

    # Convert the encoded input to a DataFrame
    new_data_encoded_df = pd.DataFrame([new_data_encoded])

    required_columns = ['vehicle_age', 'km_driven', 'mileage', 'engine', 'seats',
                        'brand_encoded', 'model_encoded',
                        'seller_type_encoded', 'transmission_type_encoded', 'fuel_type_encoded']

    for col in required_columns:
        if col not in new_data_encoded_df.columns:
            new_data_encoded_df[col] = 0  # or some default value

    # Reorder columns to match the specified order
    new_data_encoded_df = new_data_encoded_df[required_columns]

    # Predict the price and add 10%
    prediction = model.predict(new_data_encoded_df)
    adjusted_prediction = prediction[0] + prediction[0]*0.10

    return render_template('index.html', prediction_text=f'Estimated Price: ₹{adjusted_prediction:.2f}')


    
if __name__ == '__main__':
    app.run(debug=True)
