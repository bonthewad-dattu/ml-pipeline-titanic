# test_api_simple.py
import requests
import json

print("🚀 Testing ML Pipeline API...")
print("=" * 50)

# Test 1: Health Check
print("1. Testing Health Endpoint...")
try:
    response = requests.get("http://localhost:3000/health")
    print(f"✅ Status: {response.status_code}")
    print(f"📊 Response: {response.json()}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 2: API Info
print("\n2. Testing API Info...")
try:
    response = requests.get("http://localhost:3000/api/info")
    print(f"✅ Status: {response.status_code}")
    info = response.json()
    print(f"🤖 Model: {info.get('model_type', 'N/A')}")
    print(f"📋 Features: {info.get('feature_count', 'N/A')}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 3: Make Prediction
print("\n3. Testing Prediction...")
test_data = {
    "pclass": 1,
    "sex": "female", 
    "age": 25,
    "sibsp": 0,
    "parch": 0,
    "fare": 100,
    "embarked": "C"
}

try:
    response = requests.post("http://localhost:3000/api/predict", json=test_data)
    print(f"✅ Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"🎯 Prediction: {result.get('prediction')}")
        print(f"📊 Probability: {result.get('probability'):.4f}")
        print(f"🔮 Status: {result.get('survival_status')}")
        print(f"💯 Confidence: {result.get('confidence')}")
    else:
        print(f"❌ Error: {response.text}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 50)
print("🎉 Testing completed!")