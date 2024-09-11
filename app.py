import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix

# Load and preprocess the data
@st.cache
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data = data.drop('StudentID', axis=1)  # Drop irrelevant column
    
    # Encode categorical variables
    label_encoders = {}
    for column in ['Gender', 'Ethnicity', 'ParentalEducation', 'Tutoring', 'ParentalSupport']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    # Define features (X) and target (y)
    X = data.drop('GradeClass', axis=1)  # Features
    y = data['GradeClass']  # Target variable

    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Apply SMOTE for balancing
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test, scaler, label_encoders, data.head()

def main():
    # Load and preprocess data
    file_path = 'Student_performance_data.csv'
    X_train, X_test, y_train, y_test, scaler, label_encoders, sample_data = load_and_preprocess_data(file_path)
    
    # Train the machine learning models
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    logreg_model = LogisticRegression(max_iter=1000)
    logreg_model.fit(X_train, y_train)

    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train, y_train)

    # Streamlit user interface
    st.title("Student Grade Predictor")

    # Display dataset sample
    st.write("Dataset Sample (First 5 Rows):")
    st.write(sample_data)

    # Model Evaluation
    st.subheader("Model Evaluation")
    
    # Evaluation metrics
    y_pred_rf = rf_model.predict(X_test)
    y_pred_logreg = logreg_model.predict(X_test)
    y_pred_knn = knn_model.predict(X_test)
    
    st.write("Random Forest Classification Report:")
    st.text(classification_report(y_test, y_pred_rf))

    st.write("Logistic Regression Classification Report:")
    st.text(classification_report(y_test, y_pred_logreg))

    st.write("K-Nearest Neighbors Classification Report:")
    st.text(classification_report(y_test, y_pred_knn))

    # Let user select which model to use
    model_choice = st.selectbox("Choose the model for prediction", 
                                ("Random Forest", "Logistic Regression", "K-Nearest Neighbors"))

    # Let user input new data to make predictions
    st.write("Please input the values for the following features:")
    feature_names = ['Age', 'Gender', 'Ethnicity', 'ParentalEducation', 'Tutoring', 'ParentalSupport', 
                     'StudyTimeWeekly', 'Absences', 'Extracurricular', 'Sports', 'Music', 'Volunteering', 'GPA']
    input_data = []

    for feature in feature_names:
        if feature == 'Gender':
            value = st.number_input(f"Input {feature} (0 or 1)", min_value=0, max_value=1)
        elif feature == 'Ethnicity':
            value = st.number_input(f"Input {feature} (0 to 3)", min_value=0, max_value=3)
        elif feature == 'ParentalEducation' or feature == 'ParentalSupport':
            value = st.number_input(f"Input {feature} (0 to 4)", min_value=0, max_value=4)
        elif feature in ['Tutoring', 'Extracurricular', 'Sports', 'Music', 'Volunteering']:
            value = st.number_input(f"Input {feature} (0 or 1)", min_value=0, max_value=1)
        elif feature in ['StudyTimeWeekly', 'GPA']:
            value = st.number_input(f"Input {feature} (float value)")
        else:
            value = st.number_input(f"Input {feature}", value=0)
        input_data.append(value)

    # Reshape and scale input data for prediction
    input_data = np.array(input_data).reshape(1, -1)
    input_data = scaler.transform(input_data)  # Apply the same scaling as training data

    # When the button is pressed, make a prediction
    if st.button("Predict"):
        # Choose the selected model
        if model_choice == "Random Forest":
            prediction = rf_model.predict(input_data)
        elif model_choice == "Logistic Regression":
            prediction = logreg_model.predict(input_data)
        elif model_choice == "K-Nearest Neighbors":
            prediction = knn_model.predict(input_data)
        
        st.write(f"The predicted grade class is: {prediction[0]}")

if __name__ == "__main__":
    main()
