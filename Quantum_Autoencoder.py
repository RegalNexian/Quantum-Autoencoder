# -*- coding: utf-8 -*-
"""
Quantum Autoencoder for Robust Data Compression - ENHANCED VERSION V2 (with expanded datasets)

This script implements a Quantum Autoencoder with an improved training loop.
The key improvements to increase success probability and smooth training are:
1.  Statevector-Based Cost Function: Eliminates shot noise by using a statevector simulator, leading to a smooth cost landscape and faster convergence.
2.  Deterministic Optimizer: Switched back to COBYLA, which is highly effective for the now smooth and deterministic cost function.
3.  More Expressive Ansatz: Increased the number of layers and used a linear entanglement strategy in the encoder for better learning capacity.
4.  Expanded Datasets: Includes a library of 5 distinct quantum states.
"""

# Helper function to install packages if they are not present
import subprocess
import sys

# ### 1. Imports and Setup ###
import numpy as np
import matplotlib.pyplot as plt

# Qiskit Imports
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import state_fidelity, Statevector
from qiskit.visualization import plot_bloch_multivector, plot_histogram
from qiskit.circuit.library import RealAmplitudes # For random circuit

# Qiskit Aer for simulation
from qiskit_aer import AerSimulator

# Qiskit Algorithms for optimization
from qiskit_algorithms.optimizers import COBYLA

# ### 2. Configuration ###
N_QUBITS = 6
N_LATENT = 2
N_TRASH = N_QUBITS - N_LATENT
N_LAYERS = 4
MAX_ITERATIONS = 2000

print("--- Configuration ---")
print(f"Total Qubits: {N_QUBITS}")
print(f"Latent Qubits: {N_LATENT}")
print(f"Trash Qubits: {N_TRASH}")
print(f"Encoder Layers: {N_LAYERS}\n")


# ### 3. Quantum Dataset Preparation ###
def create_w_state_circuit(num_qubits):
    qc = QuantumCircuit(num_qubits)
    qc.x(0)
    for i in range(1, num_qubits):
        theta = np.arccos(np.sqrt(1 / (num_qubits - i + 1)))
        qc.ry(-theta, i); qc.cz(i - 1, i); qc.ry(theta, i)
    for i in range(num_qubits - 1, 0, -1):
        qc.cx(i, i - 1)
    return qc

def load_quantum_dataset(dataset_name, num_qubits):
    """Loads different quantum state datasets based on the number of qubits."""
    qc = QuantumCircuit(num_qubits, name='data_circuit')
    
    if dataset_name == "GHZ_State":
        qc.h(0)
        for i in range(1, num_qubits): qc.cx(0, i)
    elif dataset_name == "W_State":
        w_circuit = create_w_state_circuit(num_qubits)
        qc.compose(w_circuit, inplace=True)
    elif dataset_name == "Alternating_Layered_State":
        for i in range(num_qubits): qc.h(i)
        for i in range(0, num_qubits - 1, 2): qc.cx(i, i + 1)
        for i in range(1, num_qubits - 1, 2): qc.cx(i, i + 1)
    elif dataset_name == "Bell_State_Tensor_Product":
        if num_qubits % 2 != 0: raise ValueError("Requires an even number of qubits.")
        for i in range(0, num_qubits, 2):
            qc.h(i)
            qc.cx(i, i+1)
    elif dataset_name == "Randomized_Circuit_State":
        rand_circuit = RealAmplitudes(num_qubits, reps=3)
        random_params = np.random.rand(rand_circuit.num_parameters) * 2 * np.pi
        qc.compose(rand_circuit.assign_parameters(random_params), inplace=True)
    else:
        raise ValueError("Unknown dataset!")
    
    sim_qc = qc.copy()
    sim_qc.save_statevector()
    simulator = AerSimulator(method="statevector")
    transpiled_qc = transpile(sim_qc, simulator)
    result = simulator.run(transpiled_qc).result()
    quantum_state = result.get_statevector()
    print(f"Loaded {dataset_name} dataset for {num_qubits} qubits.")
    return quantum_state, qc

# Ask user for dataset choice
print("--- Dataset Selection ---")
print("1: GHZ State"); print("2: W State"); print("3: Alternating Layered State")
print("4: Bell State Tensor Product"); print("5: Randomized Circuit State")
choice = input("Enter the dataset number (1-5): ")
dataset_mapping = { 
    "1": "GHZ_State", "2": "W_State", "3": "Alternating_Layered_State", 
    "4": "Bell_State_Tensor_Product", "5": "Randomized_Circuit_State" 
}
dataset_name = dataset_mapping.get(choice, "GHZ_State")
original_state, data_circuit = load_quantum_dataset(dataset_name, N_QUBITS)

print("\nOriginal State Circuit:"); data_circuit.draw('mpl').show()
print("\nOriginal State Bloch Spheres:"); plot_bloch_multivector(original_state).show()


# ### 4. Quantum Autoencoder Circuit Architecture ###
def create_encoder(num_qubits, num_layers):
    params = ParameterVector('θ', length=2 * num_qubits * num_layers)
    qc = QuantumCircuit(num_qubits, name='Encoder')
    for layer in range(num_layers):
        for i in range(num_qubits):
            param_idx = layer * 2 * num_qubits + 2 * i
            qc.ry(params[param_idx], i); qc.rz(params[param_idx + 1], i)
        qc.barrier()
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)
        qc.barrier()
    return qc

total_training_qubits = N_QUBITS + 1
qae_circuit = QuantumCircuit(total_training_qubits, 1)
qae_circuit.compose(data_circuit, qubits=range(N_QUBITS), inplace=True)
qae_circuit.barrier()
encoder = create_encoder(N_QUBITS, N_LAYERS)
qae_circuit.compose(encoder, qubits=range(N_QUBITS), inplace=True)
qae_circuit.barrier()
trash_qubits = list(range(N_LATENT, N_QUBITS))
ancilla_qubit = N_QUBITS
qae_circuit.x(trash_qubits)
qae_circuit.mcx(control_qubits=trash_qubits, target_qubit=ancilla_qubit)
qae_circuit.x(trash_qubits)
qae_circuit.barrier()
decoder = encoder.inverse().copy(name='Decoder')
qae_circuit.compose(decoder, qubits=range(N_QUBITS), inplace=True)
qae_circuit.barrier()
qae_circuit.measure(ancilla_qubit, 0)

print("\nQuantum Autoencoder Training Circuit:"); qae_circuit.draw('mpl').show()


# ### 5. Training the Autoencoder with a Statevector Cost Function ###
sv_simulator = AerSimulator(method='statevector')

# Create a version of the training circuit without the final measurement
qae_circuit_sv = qae_circuit.remove_final_measurements(inplace=False)
qae_circuit_sv.save_statevector()

cost_history = []
iteration_count = 0

def cost_function_sv(params):
    """A noiseless cost function based on the exact final statevector."""
    global iteration_count
    
    bound_circuit = qae_circuit_sv.assign_parameters(params)
    
    transpiled_circuit = transpile(bound_circuit, sv_simulator)
    result = sv_simulator.run(transpiled_circuit).result()
    final_statevector = result.get_statevector()
    
    probs = final_statevector.probabilities_dict()
    prob_1 = sum(p for bitstring, p in probs.items() if bitstring[-1] == '1')
    
    cost = 1 - prob_1
    
    cost_history.append(cost)
    if iteration_count % 10 == 0:
        print(f"Iteration {iteration_count}: Cost = {cost:.4f}, Success Probability = {prob_1:.4f}")
    
    iteration_count += 1
    return cost

optimizer = COBYLA(maxiter=MAX_ITERATIONS)
initial_params = np.random.rand(qae_circuit.num_parameters) * 2 * np.pi

print("\n--- Training (Statevector Method) ---")
print("Starting training...")
optimal_result = optimizer.minimize(cost_function_sv, initial_params)
optimal_params = optimal_result.x

print("\nTraining complete!")
print(f"Final Cost: {optimal_result.fun:.4f}\n")


# ### 6. Analysis and Visualization ###
print("--- Analysis ---")
plt.figure(figsize=(10, 5))
plt.plot(range(len(cost_history)), cost_history, marker='o', linestyle='-')
plt.xlabel("Iteration")
plt.ylabel("Cost (1 - Success Probability)")
plt.title("Quantum Autoencoder Training Progress (Statevector Method)")
plt.grid(True)
plt.show()

reconstruction_circuit = QuantumCircuit(N_QUBITS)
reconstruction_circuit.compose(encoder, inplace=True)
reconstruction_circuit.compose(decoder, inplace=True)
bound_reconstruction = reconstruction_circuit.assign_parameters(optimal_params)

initial_circuit = QuantumCircuit(N_QUBITS)
initial_circuit.initialize(original_state.data, range(N_QUBITS))
full_circuit = initial_circuit.compose(bound_reconstruction)
full_circuit.save_statevector()

transpiled_full = transpile(full_circuit, sv_simulator)
result = sv_simulator.run(transpiled_full).result()
reconstructed_state = result.get_statevector()

final_fidelity = state_fidelity(original_state, reconstructed_state)
print(f"\nFinal Reconstruction Fidelity for {dataset_name}: {final_fidelity:.4f}\n")

print("Reconstructed State Bloch Spheres:")
plot_bloch_multivector(reconstructed_state).show()

original_probs = np.abs(original_state.data)**2
reconstructed_probs = np.abs(reconstructed_state.data)**2
indices = np.arange(len(original_probs))
width = 0.4

plt.figure(figsize=(12, 6))
plt.bar(indices - width/2, original_probs, width=width, alpha=0.7, label="Original State Probabilities", color='royalblue')
plt.bar(indices + width/2, reconstructed_probs, width=width, alpha=0.7, label="Reconstructed State Probabilities", color='orangered')
plt.xlabel("Computational Basis State Index")
plt.ylabel("Probability (|Amplitude|^2)")
plt.title("Original vs. Reconstructed State Probability Distribution")
plt.xticks(indices, [f'|{i:0{N_QUBITS}b}⟩' for i in indices], rotation=90)
plt.legend()
plt.tight_layout()
plt.show()