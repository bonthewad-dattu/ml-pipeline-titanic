import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

print("Testing ML Pipeline Components...")

# Check if data exists
if os.path.exists("data/raw/titanic.csv"):
    print("✅ Data file found")
    
    # Load and display data
    df = pd.read_csv("data/raw/titanic.csv")
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Simple data cleaning
    df_clean = df.copy()
    df_clean["Age"].fillna(df_clean["Age"].median(), inplace=True)
    df_clean["Embarked"].fillna("S", inplace=True)
    df_clean["Sex"] = df_clean["Sex"].map({"male": 0, "female": 1})
    
    # Prepare features
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
    X = df_clean[features]
    y = df_clean["Survived"]
    
    # Remove any remaining NaN values
    X = X.fillna(X.mean())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Model trained successfully!")
    print(f"📊 Accuracy: {accuracy:.4f}")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/simple_model.pkl")
    joblib.dump(features, "models/simple_features.pkl")
    
    print("💾 Model saved as models/simple_model.pkl")
    
else:
    print("❌ Data file not found at data/raw/titanic.csv")

print("Test completed!")
