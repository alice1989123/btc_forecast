import numpy as np
from scipy.optimize import minimize

# Define the hyperparameters to optimize
hyperparameters = np.array([1.0, 1.0, 1.0])

# Define the objective function to optimize
def objective(hyperparameters, X_train, y_train, X_val, y_val , model):
    # Train the model with the current hyperparameters
    history = compile_and_fit(model, X_train, y_train, X_val, y_val)
    
    # Compute the validation loss
    val_loss = model.evaluate(X_val, y_val, verbose=0)
    
    return val_loss

# Define the bounds for the hyperparameters
bounds = ((0.0, 10.0), (0.0, 10.0), (0.0, 10.0))

# Define the training and validation data
X_train, y_train = ...
X_val, y_val = ...

# Perform the gradient search over the hyperparameters
result = minimize(objective, hyperparameters, args=(X_train, y_train, X_val, y_val), bounds=bounds, method='L-BFGS-B')

# Get the optimized hyperparameters
optimized_hyperparameters = result.x