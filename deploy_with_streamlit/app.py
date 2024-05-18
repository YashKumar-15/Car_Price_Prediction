import numpy as np
import pandas as pd
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import pickle
import datetime



st.header('Welcome, To the Car Price Prediction ML Model')
st.sidebar.header('Select Car Features')
LR_model=pickle.load(open('LR_Model.pickle','rb'))
DT_model=pickle.load(open('DT_Model.pickle','rb'))
# DT_model=pickle.load(open('DT_Model.pickle','rb'))
df = pd.read_csv('Car_details.csv')


def extract_first_string(x):
  return x.split(' ')[0].strip()
df['name'] = df['name'].apply(extract_first_string,1)

name = st.sidebar.selectbox('Select Car', df['name'].unique())




year = st.selectbox("Select a year", range(2013, 2025))

st.markdown("Note: Select Year from 2013 to 2024")
options = ['']
fuel = st.sidebar.selectbox('Select Fuel Type', df['fuel'].unique())
seller_type = st.sidebar.selectbox('Select Seller Type', df['seller_type'].unique())
transmission = st.sidebar.radio('Select Transmission', df['transmission'].unique())

owner = st.sidebar.selectbox('Select Owner', df['owner'].unique())
default__engine_value = 2000
engine = st.slider('Select engine (CC)', 700, 3600,default__engine_value)
default_mileage_value = 12
mileage = st.slider('Select mileage', 2, 42,default_mileage_value)
default_km_driven_value = 70000
km_driven = st.slider('Select km Travelled',5000, 200000,default_km_driven_value)
# default_max_power_value = 250
# max_power = st.slider('Select max_power', 2, 400,default_max_power_value)

seats = st.slider('Select Seats', 2, 14, step=1)

# ml_model = st.radio('Select ML Model',['Linear Reagression','Decision Tree'])
if st.button('Predict'):
  df2=pd.DataFrame([[name, year, km_driven, fuel, seller_type, transmission,
       owner, mileage, engine, seats]], columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission',
       'owner', 'mileage', 'engine', 'seats'])
  
  df2['name'] = df2['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
        'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
        'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
        'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
        'Ambassador', 'Ashok', 'Isuzu', 'Opel'],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31])
  df2['owner'] = df2['owner'].replace(['First Owner', 'Second Owner', 'Third Owner',
        'Fourth & Above Owner', 'Test Drive Car'], [1,2,3,4,5])
  df2['seller_type'] = df2['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'],[1,2,3])
  df2['transmission'] = df2['transmission'].replace(['Manual', 'Automatic'],[1,2])
  df2['fuel'] = df2['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'],[1,2,3,4])

  # if ml_model == 'Linear Reagression':
  lr_prediction = LR_model.predict(df2)
  #   st.write(df2)
  st.write('Hi Predicted Price of Car is ',str(format(lr_prediction[0],'.2f')))
  # elif ml_model == 'Decision Tree':
  #   dt_prediction = DT_model.predict(df2)
  #   st.write(df2)
    # st.write('Hi Predicted Price of Car with Decision Tree is ',str(format(dt_prediction[0],'.2f')))
