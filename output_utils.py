import torch
import math
import sympy as sp

def print_statevector_symbolic(statevector: torch.Tensor):
    """
    Prints a PyTorch statevector using Dirac notation and symbolic fractions/square roots.
    """
    num_qubits = int(torch.log2(torch.tensor(statevector.shape)).item())
    output_terms = []
    
    for i, amplitude in enumerate(statevector):
        val = amplitude.item()
        
        if abs(val) > 1e-6:
            binary_state = f"{i:0{num_qubits}b}"
            
            # Convert float to a SymPy rational or expression, rounding slightly to catch sqrt fractions
            real_sym = sp.nsimplify(val.real, tolerance=1e-5, rational=False)
            imag_sym = sp.nsimplify(val.imag, tolerance=1e-5, rational=False)
            
            # Combine into a single symbolic complex number
            # SymPy uses 'I' for the imaginary unit
            complex_sym = real_sym + imag_sym * sp.I
            
            # Clean up formatting brackets for compound terms
            amp_str = str(complex_sym).replace('**', '^').replace('*', '')
            if '+' in amp_str or '-' in amp_str and not amp_str.startswith('-'):
                amp_str = f"({amp_str})"
                
            output_terms.append(f"{amp_str}|{binary_state}⟩")
            
    print(" + ".join(output_terms))

def print_probabilities(statevector: torch.Tensor):
    """
    Prints the probality of each vector
    """

    states = []

    for i in range(statevector.shape[0]):
        states.append(f"{i:0{int(math.sqrt(statevector.shape[0]))}b}")

    ket_vectors = [f"|{str(state).strip('|>')}⟩" for state in states]
    
    # Dynamically find the longest ket to ensure perfect alignment
    max_ket_len = max(len(ket) for ket in ket_vectors)
    col1_width = max(max_ket_len, 13) # Minimum width of 13 for "Quantum State"
    
    # Print Table Headers
    print(f"{'Quantum State':<{col1_width}} | {'Probability':>12}")
    print("-" * (col1_width + 15))
    
    # Print Rows
    for ket, prob in zip(ket_vectors, (torch.abs(statevector) ** 2).tolist()):
        # :>11.2% formats the float as a percentage with 4 decimal places, right-aligned
        print(f"{ket:<{col1_width}} | {prob:>11.4%}")

