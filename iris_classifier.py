import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load the Iris dataset
print("Loading Iris dataset...")
iris = load_iris()
X = iris.data
y = iris.target

# Create a DataFrame for better visualization
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(y, iris.target_names)

print("\nDataset Information:")
print(f"Number of samples: {len(X)}")
print(f"Number of features: {X.shape[1]}")
print(f"Feature names: {iris.feature_names}")
print(f"Target classes: {iris.target_names}")
print("\nFirst few rows:")
print(df.head())

# Split the data into training and testing sets (90% training, 10% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print("\n" + "="*50)
print("DATA SPLIT INFORMATION")
print("="*50)
print(f"Total samples: {len(X)}")
print(f"Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Testing samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# Create and train the Random Forest model with optimized parameters
print("\n" + "="*50)
print("TRAINING RANDOM FOREST CLASSIFIER")
print("="*50)
print("Training in progress...")
print(f"Number of trees: 200")
print(f"Max depth: 10")
print(f"Min samples split: 2")
print(f"Min samples leaf: 1")
print(f"Training samples: {len(X_train)}")

# Optimized hyperparameters for higher accuracy
rf_model = RandomForestClassifier(
    n_estimators=200,        # More trees for better accuracy
    max_depth=10,            # Deeper trees
    min_samples_split=2,     # Minimum samples to split a node
    min_samples_leaf=1,      # Minimum samples at leaf node
    max_features='sqrt',     # Number of features to consider
    bootstrap=True,          # Bootstrap samples
    random_state=42,
    verbose=1
)
rf_model.fit(X_train, y_train)

print("\n✓ Training completed successfully!")

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Save the model
with open('iris_rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("\nModel saved as 'iris_rf_model.pkl'")

# Simple interface for prediction with random values
print("\n" + "="*50)
print("PREDICTION INTERFACE WITH RANDOM VALUES")
print("="*50)

def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    """
    Predict iris species based on flower measurements
    
    Parameters:
    - sepal_length: Length of sepal in cm
    - sepal_width: Width of sepal in cm
    - petal_length: Length of petal in cm
    - petal_width: Width of petal in cm
    """
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = rf_model.predict(features)
    probabilities = rf_model.predict_proba(features)
    
    species = iris.target_names[prediction[0]]
    
    print(f"\nInput Features:")
    print(f"  Sepal Length: {sepal_length} cm")
    print(f"  Sepal Width:  {sepal_width} cm")
    print(f"  Petal Length: {petal_length} cm")
    print(f"  Petal Width:  {petal_width} cm")
    print(f"\nPredicted Species: {species.upper()}")
    print(f"\nPrediction Probabilities:")
    for i, prob in enumerate(probabilities[0]):
        print(f"  {iris.target_names[i]}: {prob*100:.2f}%")
    
    return species

# Generate random values within realistic ranges for iris flowers
print("\nGenerating random test cases...\n")

for i in range(3):
    print(f"\n--- Random Test Case {i+1} ---")
    # Random values based on typical iris flower measurements
    sepal_l = round(np.random.uniform(4.0, 8.0), 1)
    sepal_w = round(np.random.uniform(2.0, 4.5), 1)
    petal_l = round(np.random.uniform(1.0, 7.0), 1)
    petal_w = round(np.random.uniform(0.1, 2.5), 1)
    
    predict_species(sepal_l, sepal_w, petal_l, petal_w)

# Interactive prediction option
print("\n" + "="*50)
print("INTERACTIVE PREDICTION")
print("="*50)
print("\nYou can now make your own predictions!")
print("Enter values or press Enter to use random values\n")

try:
    sepal_l = input("Sepal Length (cm) [Press Enter for random]: ")
    sepal_l = float(sepal_l) if sepal_l else round(np.random.uniform(4.0, 8.0), 1)
    
    sepal_w = input("Sepal Width (cm) [Press Enter for random]: ")
    sepal_w = float(sepal_w) if sepal_w else round(np.random.uniform(2.0, 4.5), 1)
    
    petal_l = input("Petal Length (cm) [Press Enter for random]: ")
    petal_l = float(petal_l) if petal_l else round(np.random.uniform(1.0, 7.0), 1)
    
    petal_w = input("Petal Width (cm) [Press Enter for random]: ")
    petal_w = float(petal_w) if petal_w else round(np.random.uniform(0.1, 2.5), 1)
    
    predict_species(sepal_l, sepal_w, petal_l, petal_w)
    
except KeyboardInterrupt:
    print("\n\nProgram terminated by user.")
except Exception as e:
    print(f"\nError: {e}")

print("\n" + "="*50)
print("Activity completed successfully!")
print("="*50)