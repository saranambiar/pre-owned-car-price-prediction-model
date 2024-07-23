# importing all the required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

#reading csv file into variable cars

cars = pd.read_csv("cardekho_dataset.csv")

#dropping unnecessary columns from the dataset

cars.drop(labels=['car_name','max_power','Unnamed: 0'],axis=1,inplace=True)

#handling outliers

p = cars['selling_price'].quantile(0.99)
cars = cars[cars['selling_price']<p]

p = cars['mileage'].quantile(0.99)
cars = cars[cars['mileage']<p]

p = cars['km_driven'].quantile(0.99)
cars = cars[cars['km_driven']<p]
p = cars['engine'].quantile(0.99)
cars = cars[cars['engine']<p]
p = cars['vehicle_age'].quantile(0.99)
cars = cars[cars['vehicle_age']<p]

brand_dic = dict()
model_dic = dict()
seller_type_dic = dict()
transmission_dic = dict()
fuel_dic = dict()

#using LabelEncoder to encode columns of data

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

cars['brand_encoded'] = le.fit_transform(cars['brand'])
for num, brand in enumerate(le.classes_):
    brand_dic[brand] = num

cars['model_encoded'] = le.fit_transform(cars['model'])
for num, model in enumerate(le.classes_):
    model_dic[model] = num

cars['seller_type_encoded'] = le.fit_transform(cars['seller_type'])
for num, seller_type in enumerate(le.classes_):
    seller_type_dic[seller_type] = num

cars['transmission_type_encoded'] = le.fit_transform(cars['transmission_type'])
for num, transmission in enumerate(le.classes_):
    transmission_dic[transmission] = num

cars['fuel_type_encoded'] = le.fit_transform(cars['fuel_type'])
for num, fuel in enumerate(le.classes_):
    fuel_dic[fuel] = num


cars_new = cars.drop(['brand','model','seller_type','transmission_type','fuel_type'],axis=1)

X = cars_new.drop(['selling_price'],axis=1)
y = cars_new['selling_price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 25, random_state = 0)
regressor.fit(X_train, y_train)






# Save the model along with feature names
model_filename = 'model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump((regressor, X.columns.tolist()), file)

print("Model and feature names saved successfully")

