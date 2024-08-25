# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The problem statement for this experiment is to design and implement a neural network regression model to accurately predict a continuous target variable based on the input features from the provided dataset. The objective is to create a robust and reliable predictive model capable of capturing complex relationships within the data, ultimately yielding accurate and precise predictions. The model will undergo training, validation, and testing to ensure it generalizes well to unseen data. The focus will be on optimizing performance metrics such as mean squared error (MSE) or mean absolute error (MAE) to achieve the best possible predictive accuracy. By successfully implementing this regression model, valuable insights can be gained into the underlying patterns and trends within the dataset, thereby enhancing decision-making and understanding of the behavior of the target variable.

## Neural Network Model

![image](https://github.com/user-attachments/assets/e06fa686-9773-4b69-a6f9-6d9978c191ce)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

 Define the structure of your neural network and compile it with an appropriate optimizer and loss function.

### STEP 5:

Train the model with the training data.

### STEP 6:

Create plots to visualize how well the model is performing during training.

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:SHARMITHA V
### Register Number: 212223110048
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from google.colab import auth
import gspread
from google.auth import default

df=pd.read_csv('data 1 DL.csv')

auth.authenticate_user()
creds,_ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('exp 1').sheet1
rows = worksheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'INPUT':'float'})
df = df.astype({'OUTPUT':'float'})

df.head()
X = df[['input']].values
y = df[['output']].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential([
    Dense(8, activation='relu'),
    Dense(10,activation='relu'),
    Dense(1)
])
model.compile(optimizer = 'rmsprop', loss='mse')
model.fit(X_train1, y_train,epochs=2000)
loss=pd.DataFrame(model.history.history)
loss.plot()

X_test1 = Scaler.transform(X_test)

model.evaluate(X_test1,y_test)

num=[[3]]
num_1=Scaler.transform(num)
AI_Brain.predict(X_n1_1)
```
## Dataset Information

![image](https://github.com/user-attachments/assets/49c92ec7-502c-4aab-87bc-b078037a38e5)


## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/27b12ef9-6081-4877-b626-d157f3318fb8)


### Test Data Root Mean Squared Error

![image](https://github.com/user-attachments/assets/deda760c-244d-46d6-a416-354311e72447)


### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/1f910418-b623-4960-9bd4-c5f0260a2e64)


## RESULT

To develop a neural network regression model for the given dataset is created sucessfully.
