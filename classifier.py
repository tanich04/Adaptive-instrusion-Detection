import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
import joblib

# Function to optimize loading of large datasets
def load_data(filepath, chunk_size=100000):
    # Define dtypes for known columns
    dtype_dict = {
        "frame.number": "int32",
        "frame.len": "int32",
        "frame.protocols": "category",
        "eth.type": "category",
        "ip.len": "float32",
        "ip.ttl": "float32",
        "tcp.srcport": "float32",
        "tcp.dstport": "float32",
        "tcp.flags.syn": "float32",
        "tcp.flags.ack": "float32",
        "alert": "category",  # Target column
    }

    chunks = []
    try:
        for chunk in pd.read_csv(filepath, dtype=dtype_dict, low_memory=False, chunksize=chunk_size):
            chunks.append(chunk)
    except Exception as e:
        print(f"Error while loading data: {e}")
        exit()

    # Combine chunks into a single DataFrame
    return pd.concat(chunks, ignore_index=True)

# Function to preprocess the data
def preprocess_data(data):
    # Drop unnecessary columns
    unused_columns = [
        "frame.time", "frame.time_epoch", "eth.src", "eth.dst", "ip.src", "ip.dst",
        "http.request.method", "http.request.uri", "http.user_agent", "http.host",
        "dns.qry.name", "alert"  # Keep `alert` for the target
    ]
    features = data.drop(columns=unused_columns, errors='ignore')

    # Encode target column
    label_encoder = LabelEncoder()
    data['alert'] = label_encoder.fit_transform(data['alert'])  # benign = 0, malicious = 1
    target = data['alert']

    # Handle missing values
    features.fillna(0, inplace=True)

    # Sparse one-hot encoding for categorical features
    categorical_cols = features.select_dtypes(include=['category']).columns
    features = pd.get_dummies(features, columns=categorical_cols, sparse=True)

    return features, target, label_encoder

# Function to train the model
def train_model(X_train, y_train):
    # Train Random Forest Classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf

# Main script
if __name__ == "__main__":
    # Step 1: Load the dataset
    filepath = "data.csv"  # Change to the path of your dataset
    print("Loading dataset...")
    data = load_data(filepath)
    print(f"Dataset loaded successfully. Shape: {data.shape}")

    # Step 2: Preprocess the data
    print("Preprocessing dataset...")
    features, target, label_encoder = preprocess_data(data)
    print(f"Features shape: {features.shape}, Target shape: {target.shape}")

    # Step 3: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Step 4: Train the model
    print("Training model...")
    classifier = train_model(X_train, y_train)
    print("Model trained successfully.")

    # Step 5: Evaluate the model
    print("Evaluating model...")
    y_pred = classifier.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Step 6: Save the trained model
    model_filename = "packet_classifier.pkl"
    joblib.dump(classifier, model_filename)
    print(f"Model saved as '{model_filename}'.")

    # Test with new data (optional)
    new_packet = X_test.iloc[0].values.reshape(1, -1)  # Use the first test row as an example
    prediction = classifier.predict(new_packet)
    print(f"New Packet Prediction: {label_encoder.inverse_transform(prediction)}")
