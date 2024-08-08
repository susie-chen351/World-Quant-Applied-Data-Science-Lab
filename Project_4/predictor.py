import pandas as pd
import pickle

def make_predictions(data_filepath, model_filepath):
    """Generate predictions from the provided model and data file."""
    # Load data
    X_test = pd.read_csv(data_filepath)
    # Load model
    with open(model_filepath, "rb") as f:
        model = pickle.load(f)
    # Generate predictions
    y_test_pred = model.predict(X_test.values)
    # Put predictions into Series with same index as X_test ("Bankrupt?")
    y_test_pred = pd.Series(y_test_pred, index=X_test.index, name="Bankrupt?")
    return y_test_pred