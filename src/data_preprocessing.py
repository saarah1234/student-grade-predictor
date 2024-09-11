# src/data_preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess_data(file_path, training=True):
    data = pd.read_csv(file_path)
    data = data.drop('StudentID', axis=1)
    
    # Display the first few rows of the data for verification
    print("First few rows of the data:")
    print(data.head())
    
    # Encode categorical variables
    label_encoders = {}
    for column in ['Gender', 'Ethnicity', 'ParentalEducation', 'Tutoring', 'ParentalSupport']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    # Define features (X) and target (y)
    X = data.drop('GradeClass', axis=1)
    y = data['GradeClass']

    if training:
        # Scale the features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test, scaler, label_encoders
    else:
        # No scaling or splitting, just return features and target
        return X, y

if __name__ == "__main__":
    file_path = 'C:\student_grade_project\data\Student_performance_data.csv'  # Adjust path as needed
    X_train, X_test, y_train, y_test, scaler, label_encoders = load_and_preprocess_data(file_path)
