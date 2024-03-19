# Machine-Learning-Model-for-Cinnamon-tree-flower-Prediction

A basic ML model in predicting the prices of cinnamon flowers in the Sri Lankan economy considering several factors which is vital for the  price prediction


from google.colab import drive
drive.mount('/content/drive/', force_remount=True)

import numpy as np
import pandas as pd

#For the model building itself I have considered the producer prices of Cinnamon tree flowers to predict its future values.For that I have used a linear regression kodel using some of the specific variables from the given data set.

For tha analysis based on the provided information I have assumed that the exchange rate (local currency units per USD) would have a higher impact on predicting the value column in the given dataset than the Local currency units per USD.

According to the deflator data set provided,it is possible to forecast the value column in the dataset using both the GDP deflator and the value added deflator for agriculture, forestry, and fishing. The value added deflator for the designated sectors focuses on inflation within the agriculture, forestry, and fishery industries while the GDP deflator measures inflation in the economy as a whole. The expected value may change as a result of changes in these inflation indicators, with higher predicted values perhaps resulting from rising inflation and lower predicted values possibly resulting from falling inflation. However, the actual impact will depend on the unique dynamics and relationships contained in the dataset, and more investigation is needed to pinpoint the consequences.

As the whole analysis related to the food-related businesses, variables like Consumer Prices, Food Indices (2015 = 100), and Food price inflation are likely to have a stronger impact on predicting the value column. To pinpoint the precise impacts and comprehend the precise linkages in the dataset, we need additional analysis.

As I have mentioned above the below code depicts a linear regression model for the prediction aanalysis for cinnamon tree flowers.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

cinnamon = r"data sheet"
# Load the data into a Pandas DataFrame
data = pd.read_csv("/content/drive/MyDrive/ML Assignment0032 - Sheet1.csv")

#print the data columns
print(data.columns)

# Extract the features (exchange rate and GDP deflator) and the target variable (cashew nuts value)
features = data[['exchange rate(LKR)', 'GDP Deflator', 'Deflator - Value Added Deflator (Agriculture, forestry and fishery)\n']]
target = data['Cinnamon and cinnamon-tree flowers, raw\r\n']

# Create an instance of MinMaxScaler
scaler = MinMaxScaler()

# Normalize the features
normalized_features = scaler.fit_transform(features)

# Split the normalized features and target variable into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(normalized_features, target, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the mean squared error (MSE) to evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

future_features = pd.DataFrame({
    'exchange rate(LKR)': [201.97, 202.34, 202.89],
    'GDP Deflator': [135.67, 135.89, 136.76],
    'Deflator - Value Added Deflator (Agriculture, forestry and fishery)': [165.45, 166.67, 167.26]
})


import pandas as pd
#To get the standard deviation
# Load the dataset
data = pd.read_csv("/content/drive/MyDrive/ML Assignment0032 - Sheet1.csv")  # Replace 'your_dataset.csv' with the actual filename


# Calculate the standard deviation
std_dev = data.std()


# Print the standard deviation
print("Standard Deviation:")
print(std_dev)

import matplotlib.pyplot as plt
import numpy as np

# Assuming you have actual_values and predicted_values arrays

plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.show()

![image](https://github.com/Pavanirt/Machine-Learning-Model-for-Cinnamon-tree-flower-Prediction/assets/160448544/ed1e9751-3e51-4d5c-a719-e6bc181951be)

#The outcome implies the predicted values for cinnamon tree flowers are clustered more closely around the average prediction. This can be seen as a positive outcome, as it suggests that the prediction model is capturing the underlying patterns and factors influencing the number of flowers accurately.

Therefore for a further evaluation we can use additional metrics, such as mean absolute error or coefficient of determination (R-squared), and assess the significance of the predictions in relation to the actual data and any applicable domain knowledge.

! pip install pandas-profiling
! pip install ipywidgets

from pandas_profiling import ProfileReport
report = ProfileReport(data)
report

#Report

![image](https://github.com/Pavanirt/Machine-Learning-Model-for-Cinnamon-tree-flower-Prediction/assets/160448544/8e846d58-a292-46f1-8bca-f8adc9402542)
![image](https://github.com/Pavanirt/Machine-Learning-Model-for-Cinnamon-tree-flower-Prediction/assets/160448544/3060f6db-adb7-440e-a119-3ad918e3b6a7)
![image](https://github.com/Pavanirt/Machine-Learning-Model-for-Cinnamon-tree-flower-Prediction/assets/160448544/dc26d049-8cb9-4e8e-ab26-df66e119e5eb)
![image](https://github.com/Pavanirt/Machine-Learning-Model-for-Cinnamon-tree-flower-Prediction/assets/160448544/b4464a62-4dda-4382-aa18-86d9f6ff1167)
![image](https://github.com/Pavanirt/Machine-Learning-Model-for-Cinnamon-tree-flower-Prediction/assets/160448544/195bab20-8840-4e66-9305-5ce7476de9f5)
![image](https://github.com/Pavanirt/Machine-Learning-Model-for-Cinnamon-tree-flower-Prediction/assets/160448544/7fc503ad-2778-47ee-8b92-28773200d3bf)
![image](https://github.com/Pavanirt/Machine-Learning-Model-for-Cinnamon-tree-flower-Prediction/assets/160448544/07a6c99c-59b4-49a1-a1fa-6a3d30f0240a)


#Interactions between variables

![image](https://github.com/Pavanirt/Machine-Learning-Model-for-Cinnamon-tree-flower-Prediction/assets/160448544/95364355-e277-4142-bc86-dc0682261048)
![image](https://github.com/Pavanirt/Machine-Learning-Model-for-Cinnamon-tree-flower-Prediction/assets/160448544/362a27d4-6985-4f59-a3e1-b8e2905165f7)
![image](https://github.com/Pavanirt/Machine-Learning-Model-for-Cinnamon-tree-flower-Prediction/assets/160448544/7e286480-62b3-4d5e-b384-f2b1586668d5)










