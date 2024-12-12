from sklearn.metrics import log_loss
from qiskit.quantum_info import SparsePauliOp
from tqdm import tqdm
import numpy as np

def cost_func_classification(parameters_values, ansatz, observable, estimator, x_train, y_train):
    """
    Cost function for binary classification using a quantum circuit.

    Args:
        parameters_values: Parameter values for the ansatz (TwoLocal circuit).
        ansatz: Combined circuit (feature map + ansatz).
        estimator: Qiskit StatevectorEstimator or similar.
        x_train: Training features (must match the number of qubits).
        y_train: Binary labels (0 or 1).
        observable: SparsePauliOp observable for classification.

    Returns:
        Binary cross-entropy loss.
    """
    losses = []

    for i, (x, y) in tqdm(enumerate(zip(x_train, y_train))):
        # Bind feature values to the feature map
        feature_param_dict = {param: val for param, val in zip(ansatz.parameters[:len(x)], x)}
        
        # Bind model parameters to the ansatz
        param_dict = {param: val for param, val in zip(ansatz.parameters[len(x):], parameters_values)}
        
        # Combine both parameter sets
        all_params = {**feature_param_dict, **param_dict}
        
        # Assign all parameters to the circuit
        bound_circuit = ansatz.assign_parameters(all_params)
        
        # Estimate the expectation value
        result = estimator.run([(bound_circuit,observable)]).result()[0].data.evs.item() # Corrected call
        prob = (1 + result) / 2  # Convert expectation value to probability in [0, 1]
        # Compute binary cross-entropy loss
        loss = log_loss([y], [prob], labels=[0, 1])
        losses.append(loss)
    
    return np.mean(losses)


def predict(parameters_values, ansatz, observable, estimator, testing_x):
    """
    Predict binary labels using a quantum circuit.

    Args:
        parameters_values: Optimized parameter values for the ansatz.
        ansatz: Combined circuit (feature map + ansatz).
        sampler: Qiskit's Sampler primitive.
        x_test: Test features (must match the number of qubits).

    Returns:
        List of predicted probabilities.
    """
    predictions = []

    for x in testing_x:
        # Bind feature values to the feature map
        feature_param_dict = {param: val for param, val in zip(ansatz.parameters[:len(x)], x)}
        
        # Bind model parameters to the ansatz
        param_dict = {param: val for param, val in zip(ansatz.parameters[len(x):], parameters_values)}
        
        # Combine both parameter sets
        all_params = {**feature_param_dict, **param_dict}
        
        # Assign all parameters to the circuit
        bound_circuit = ansatz.assign_parameters(all_params)
        
        # Estimate the expectation value
        result = estimator.run([(bound_circuit,observable)]).result()[0].data.evs.item() # Corrected call
        prob = (1 + result) / 2  # Convert expectation value to probability in [0, 1]
        
        predictions.append(prob)
    
    return predictions
