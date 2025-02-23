import joblib

# Save the trained model
joblib.dump(model, 'disease_outbreak_model.pkl')

# Load the model (when you need to use it later)
loaded_model = joblib.load('disease_outbreak_model.pkl')

# Use the loaded model for prediction
loaded_prediction = loaded_model.predict(new_data)
print(loaded_prediction)
