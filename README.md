# 🎓 Graduate Admission Prediction using Artificial Neural Networks 🧠
This project uses a neural network to predict the chances of admission into a graduate program based on academic features like GRE scores, CGPA, and research experience.

# 🗂️ Project Files
Graduate_admission_prediction_using_ANN.ipynb – Main Jupyter Notebook

abc.csv – Dataset used for training and prediction

# 🧰 Technologies Used
Python 🐍

Pandas & NumPy – Data handling 🧮

Seaborn & Matplotlib – Visualization 📊

Scikit-learn – Preprocessing and splitting 🧪

TensorFlow / Keras – ANN Model 🔧

📄 Dataset Overview
The dataset contains the following columns:

GRE Score

TOEFL Score

University Rating

SOP (Statement of Purpose strength)

LOR (Letter of Recommendation strength)

CGPA

Research (0 or 1)

Chance of Admit (target)

# 📊 Data Preprocessing
```python
df.drop(columns=['Serial No.'], inplace=True)
```
Checked for duplicates and dropped unnecessary columns.

Features and target split using:

```python
X = df.iloc[:, 0:-1]
Y = df.iloc[:, -1]
```
# 🔄 Train-Test Split
```python
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
```
# 🔍 Feature Scaling
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
# 🧠 ANN Model Architecture
```python
model = Sequential()
model.add(Dense(7, activation='relu', input_dim=7))
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='linear'))
```
Loss function: Mean Squared Error

Optimizer: Adam

```python
model.compile(loss='mean_squared_error', optimizer='adam')
```
# 🚀 Model Training
```python
history = model.fit(X_train_scaled, Y_train, epochs=100, validation_split=0.2)
```
# 📈 Evaluation
Training & Validation Loss plots

Model summary using .summary()

# 🧪 Results & Observations
The ANN accurately captures trends in graduate admissions.

Features like CGPA and Research Experience have strong impact.

