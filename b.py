import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load your dataset
df = pd.read_csv('outbreak_data.csv')  # Replace with your data file path

# Preview the dataset
print(df.head())

# Handle missing values (example: forward fill method)
df.fillna(method='ffill', inplace=True)

# Feature engineering: Let's say you have a 'date' column
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# Feature selection (choose the relevant features you want for training)
X = df[['year', 'month', 'temperature', 'population_density']]  # Example features
y = df['outbreak']  # The target variable (0: no outbreak, 1: outbreak)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (if required for some models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
