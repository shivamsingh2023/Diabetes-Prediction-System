import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Check if model exists
MODEL_FILE = "diabetes_model.pkl"

# Train model if not exists
if not os.path.exists(MODEL_FILE):
    # Load dataset
    data = pd.read_csv("data.csv")
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model and scaler
    pickle.dump({"model": model, "scaler": scaler}, open(MODEL_FILE, "wb"))

# Load model and scaler
data = pickle.load(open(MODEL_FILE, "rb"))
model = data["model"]
scaler = data["scaler"]

def predict_diabetes(input_data):
    input_scaled = scaler.transform([input_data])
    return model.predict(input_scaled)[0]
