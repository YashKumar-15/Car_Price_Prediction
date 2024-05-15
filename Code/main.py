# %%
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import pickle

# %%
df = pd.read_csv('Car_details.csv')
df

# %% [markdown]
# ### Data cleaning

# %%
df2 = df.drop(columns=['torque'])
df2

# %%
df2.describe()

# %%
df2.info()

# %%
df2.isnull().sum()

# %%
df2.dropna(inplace=True)
df2.shape

# %%
df2.duplicated().sum()

# %%
df2.drop_duplicates(inplace=True)
df2.shape

# %%
for col in df2.columns:
  print(f' column => {col}')
  print('===================')
  print(df2[col].unique())

# %%
df2.info()

# %%
def extract_first_string(x):
  return x.split(' ')[0].strip()
def float_extract_first_string(x):
  x=str(x).split(' ')[0]
  x=x.strip()
  if x=='':
    x=0
  return float(x)


df2['name'] = df2['name'].apply(extract_first_string,1)
df2['mileage'] = df2['mileage'].apply(float_extract_first_string,1)
df2['engine'] = df2['engine'].apply(float_extract_first_string,1)
df2['max_power'] = df2['max_power'].apply(float_extract_first_string,1)
df2.head(3)

# %%
df2['max_power'] = df2['max_power'].apply(float_extract_first_string,1)

# %%
df2['owner'].unique()

# %%
df2['seller_type'].unique()

# %%
df2['transmission'].unique()

# %%
df2['fuel'].unique()

# %%
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

# %%
df2

# %%
df2.info()

# %%
X = df2.drop(columns=['selling_price'])
y = df2['selling_price']

# %% [markdown]
# ### Training Machine Leaning Models

# %%
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

# %%
# linear regression Model
model = LinearRegression()

# %%
model.fit(Xtrain, ytrain)

# %%
pred = model.predict(Xtest)

# %%
pred

# %%
# Create and train the Decision Tree model
dt_model = DecisionTreeRegressor(random_state=40)
dt_model.fit(Xtrain, ytrain)

# Make predictions
y_pred_dt = dt_model.predict(Xtest)



# %% [markdown]
# ### Exporting Trained Models

# %%
pickle.dump(model,open('LR_Model.pickle','wb'))

# %%
pickle.dump(dt_model,open('DT_Model.pickle','wb'))


