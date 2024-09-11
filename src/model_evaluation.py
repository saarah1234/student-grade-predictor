# src/model_evaluation.py
from sklearn.metrics import classification_report
import joblib
from data_preprocessing import load_and_preprocess_data
from sklearn.model_selection import train_test_split

def evaluate_models():
    # Load full dataset for evaluation
    X, y = load_and_preprocess_data('../data/Student_performance_data.csv', training=False)
    
    # Split data into test set
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Load models
    rf_model = joblib.load('../models/rf_model.pkl')
    logreg_model = joblib.load('../models/logreg_model.pkl')
    knn_model = joblib.load('../models/knn_model.pkl')

    # Predictions
    rf_preds = rf_model.predict(X_test)
    logreg_preds = logreg_model.predict(X_test)
    knn_preds = knn_model.predict(X_test)

    # Evaluation metrics
    print("Random Forest Performance:")
    print(classification_report(y_test, rf_preds))

    print("Logistic Regression Performance:")
    print(classification_report(y_test, logreg_preds))

    print("K-Nearest Neighbors Performance:")
    print(classification_report(y_test, knn_preds))

if __name__ == "__main__":
    evaluate_models()
