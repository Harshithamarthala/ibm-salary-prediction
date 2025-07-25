import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load and clean data
df = pd.read_csv('data.csv')
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Encode all categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
X = df.drop('income', axis=1)
y = df['income']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open('income_prediction_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save all encoders
for col, le in label_encoders.items():
    with open(f'{col}_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

# ✅ Save the column names used during training
with open('model_columns.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("\n✅ Model and encoders saved successfully!")
