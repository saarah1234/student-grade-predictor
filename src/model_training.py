# src/model_training.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import joblib
from data_preprocessing import load_and_preprocess_data


def train_models():
    X_train, X_test, y_train, y_test, scaler, label_encoders = load_and_preprocess_data('../data/Student_performance_data.csv')

    # Train Random Forest model
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, '../models/rf_model.pkl')

    # Train Logistic Regression model
    logreg_model = LogisticRegression(max_iter=1000)
    logreg_model.fit(X_train, y_train)
    joblib.dump(logreg_model, '../models/logreg_model.pkl')

    # Train K-Nearest Neighbors model
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train, y_train)
    joblib.dump(knn_model, '../models/knn_model.pkl')

    return rf_model, logreg_model, knn_model, X_test, y_test

if __name__ == "__main__":
    train_models()
