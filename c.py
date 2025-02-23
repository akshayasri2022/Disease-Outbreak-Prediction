# Example new data for prediction (replace with real data for prediction)
new_data = pd.DataFrame({
    'year': [2025],
    'month': [2],
    'temperature': [30],  # Example temperature
    'population_density': [5000]  # Example population density
})

# Standardize the new data using the same scaler
new_data = scaler.transform(new_data)

# Predict the outbreak (0 = no outbreak, 1 = outbreak)
prediction = model.predict(new_data)

if prediction[0] == 1:
    print("Potential Disease Outbreak Predicted!")
else:
    print("No Outbreak Predicted.")
