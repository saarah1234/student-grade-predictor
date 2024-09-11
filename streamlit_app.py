import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('Student_performance_data.csv')

# Process and clean the data as needed
data = data.drop('StudentID', axis=1)  # Drop irrelevant column

# Encode categorical variables
label_encoders = {}
for column in ['Gender', 'Ethnicity', 'ParentalEducation', 'Tutoring', 'ParentalSupport']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Display dataset in Streamlit for confirmation
st.write("Dataset Overview:")
st.write(data.head())

# Define features (X) and target (y)
X = data.drop('GradeClass', axis=1)  # Features
y = data['GradeClass']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the machine learning models
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

logreg_model = LogisticRegression(max_iter=1000)
logreg_model.fit(X_train, y_train)

knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

# Streamlit user interface
st.title("Student Grade Predictor")

# Let user select which model to use
model_choice = st.selectbox("Choose the model for prediction", 
                            ("Random Forest", "Logistic Regression", "K-Nearest Neighbors"))

# Let user input new data to make predictions
st.write("Please input the values for the following features:")
input_data = []
for feature in X.columns:
    value = st.number_input(f"Input {feature}", value=0)
    input_data.append(value)

# When the button is pressed, make a prediction
if st.button("Predict"):
    input_data = [input_data]  # Reshape input for prediction
    
    # Choose the selected model
    if model_choice == "Random Forest":
        prediction = rf_model.predict(input_data)
    elif model_choice == "Logistic Regression":
        prediction = logreg_model.predict(input_data)
    elif model_choice == "K-Nearest Neighbors":
        prediction = knn_model.predict(input_data)
    
    st.write(f"The predicted grade class is: {prediction[0]}")




