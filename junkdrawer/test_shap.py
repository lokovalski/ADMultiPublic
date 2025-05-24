import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier

# Create a small test dataset
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y = np.array([0, 1, 0])

# Train a simple model
model = RandomForestClassifier(n_estimators=10)
model.fit(X, y)

# Create explainer and get SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

print("Testing different SHAP force plot parameter combinations...")

# Print shapes for debugging
print("\nShapes:")
print(f"Expected value shape: {explainer.expected_value.shape if hasattr(explainer.expected_value, 'shape') else 'scalar'}")
print(f"SHAP values shape: {np.array(shap_values).shape}")
print(f"X shape: {X.shape}")

# Try different combinations
combinations = [
    ("base_value only", lambda: shap.plots.force(explainer.expected_value)),
    ("base_value and shap_values", lambda: shap.plots.force(explainer.expected_value, shap_values)),
    ("base_value[0] and shap_values[0]", lambda: shap.plots.force(explainer.expected_value[0], shap_values[0])),
    ("base_value[1] and shap_values[1]", lambda: shap.plots.force(explainer.expected_value[1], shap_values[1])),
    ("base_value and shap_values[..., 0]", lambda: shap.plots.force(explainer.expected_value, shap_values[..., 0])),
]

for name, func in combinations:
    try:
        func()
        print(f"✓ Success with {name}")
    except Exception as e:
        print(f"✗ Error with {name}: {str(e)}") 