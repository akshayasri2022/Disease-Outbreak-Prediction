# main.py

import os
from src.preprocess import preprocess_data
from src.train_model import train_model
from src.predict import make_predictions

def main():
    # Step 1: Preprocess the data
    print("Starting data preprocessing...")
    data_file = os.path.join("data", "outbreak_data.csv")  # Replace with your actual data file
    processed_data = preprocess_data(data_file)
    print("Data preprocessing complete.")

    # Step 2: Train the model
    print("Training the model...")
    model = train_model(processed_data)
    print("Model training complete.")

    # Step 3: Make predictions using the trained model
    print("Making predictions...")
    predictions = make_predictions(model, processed_data)
    
    print(f"Predictions: {predictions}")
    
if __name__ == "__main__":
    main()
