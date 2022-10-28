"""Functions for the code's encoding.

Defines functions related to computing the encodings for the code.
"""

import numpy as np
import qiskit.quantum_info as qi
from qiskit import QuantumCircuit, QuantumRegister

def get_random_encoding(n,
                        k,
                        num_gates,
                        num_local_qubits = 2,
                        qr = None,
                        stabilizer_format = 'matrix',
                        gate_error_list = None,
                        return_logical_operators = False,
                        return_circuit = False,
                        return_encoding = False,
                        return_gate_errors = False):
    """
    Creates a random encoding for the stabilizer code.

    Parameters
    ----------
    n : int
        Total number of encoding qubits.
    k : int
        Number of original data qubits.
    num_gates : int
        Number of Clifford basis gates composing the encoding.
    num_local_qubits : int, default 2
        Number of qubits the Clifford basis gates affect.
    qr : QuantumRegister, optional
        Optional quantum register, to use to create the circuit.
    stabilizer_format: {'matrix','str','both'}
        In what format to return the stabilizers. If ``'matrix'``, returns 
        them as a ((n-k) x 2n) parity check matrix. If ``'str'``, they are 
        returned as a list of strings, each indicating a stabilizer. If 
        ``'both'``, returns both.
    return_logical_operators : bool, default False
        Also return the 2k minimal logical operators associated with the
        encoding.
    return_circuit : bool, default False
        Also return the circuit implementing the encoding.
    return_encoding : bool, default False
        Also return encoding as an instruction. For certain uses, it is more 
        efficient to use it as an instruction than as an actual circuit.

    Returns
    -------
    stabilizers : 2D array, list, tuple
        Either a ((n-k) x 2n) 2D array, or a list of strings, or a tuple of 
        both, encoding the stabilizers, depending on `stabilizer_format`.
    logical_operators : tuple
        Tuple ``(logical_Xs, logical_Zs)`` containing a list of strings 
        indicating the k Pauli strings encoding the X logical gates, and a 
        list of k Pauli strings encoding the Z logical gates.
    circuit : QuantumCircuit
        Quantum circuit implementing the encoding.
    encoding : Instruction
        Encoding given as an instruction.

    Notes
    -----
    It is assumed that the stabilizer order that Qiskit uses is preserved, so 
    that, among the n minimal stabilizers of the encoding, the first k are 
    actually logical operators (encoding the Z gate) and the next (n-k) 
    stabilizers are the ones actually considered stabilizers in the context of
    stabilizer codes. 
    """

    qubit_sets_to_apply_to = [
        list(
            np.random.choice(n, num_local_qubits, replace=False)
        ) for _ in range(num_gates)
    ]

    gate_list = [
        qi.random_clifford(
            num_local_qubits).to_circuit() for _ in range(num_gates)
    ]

    if qr is None:
        qr = QuantumRegister(n, 'q')

    circ = QuantumCircuit(qr)
    for gate, mapping in zip(gate_list, qubit_sets_to_apply_to):
        circ.append(gate, [qr[i] for i in mapping])

    total_stabilizers = qi.Clifford(circ).stabilizer
    stabilizers = total_stabilizers[k:]

    # Here we remove the first k stabilizers
    # (I don't have guarantees that they are the right stabilizers to remove,
    # but my tests suggest that Qiskit already orders them)    

    if stabilizer_format == 'str':
        stabilizers = stabilizers.to_labels()
    
    elif stabilizer_format == 'matrix':
        # Stabilizers in parity check matrix format
        stabilizers = stabilizers.array.astype(int)

    elif stabilizer_format == 'both':
        stabilizers_str = stabilizers.to_labels()
        parity_check_matrix = stabilizers.array.astype(int)
        stabilizers = (parity_check_matrix, stabilizers_str)

    if return_logical_operators:
        h_circ = QuantumCircuit(qr)
        h_circ.h([qr[i] for i in range(k)])
        result_X = qi.Clifford(h_circ + circ).stabilizer.to_labels()

        result_Z = total_stabilizers.to_labels()
        logical_Zs = result_Z[:k]
        logical_Xs = result_X[:k]
        #assert set(stabilizers) == set(result_X[k:]), "The stabilizers don't match."

    if return_circuit:
        encoding = circ.to_instruction()
        circuit = QuantumCircuit(qr)
        circuit.append(encoding, qr)
    elif return_encoding:
        encoding = circ.to_instruction()

    if return_gate_errors:
        gate_errors = get_gate_errors(n,
                                    gate_list,
                                    qubit_sets_to_apply_to,
                                    gate_error_list=gate_error_list)

    # Implementing all the return possibilities...
    if return_gate_errors:
        if return_logical_operators:
            if return_circuit:
                if return_encoding:
                    return stabilizers, (logical_Xs, logical_Zs), circuit, encoding, gate_errors
                else:
                    return stabilizers, (logical_Xs, logical_Zs), circuit, gate_errors
            else:
                if return_encoding:
                    return stabilizers, (logical_Xs, logical_Zs), encoding, gate_errors
                else:
                    return stabilizers, (logical_Xs, logical_Zs), gate_errors
        else:
            if return_circuit:
                if return_encoding:
                    return stabilizers, circuit, encoding, gate_errors
                else:
                    return stabilizers, circuit, gate_errors
            else:
                if return_encoding:
                    return stabilizers, encoding, gate_errors
                else:
                    return stabilizers, gate_errors
    else:
        if return_logical_operators:
            if return_circuit:
                if return_encoding:
                    return stabilizers, (logical_Xs, logical_Zs), circuit, encoding
                else:
                    return stabilizers, (logical_Xs, logical_Zs), circuit
            else:
                if return_encoding:
                    return stabilizers, (logical_Xs, logical_Zs), encoding
                else:
                    return stabilizers, (logical_Xs, logical_Zs)
        else:
            if return_circuit:
                if return_encoding:
                    return stabilizers, circuit, encoding
                else:
                    return stabilizers, circuit
            else:
                if return_encoding:
                    return stabilizers, encoding
                else:
                    return stabilizers

def get_gate_errors(n, gate_list, qubit_sets_to_apply_to, gate_error_list=None):

    if gate_error_list is None:
        gate_error_list = ['IX','IY','IZ',
                            'XI','XX','XY','XZ',
                            'YI','YX','YY','YZ',
                            'ZI','ZX','ZY','ZZ']

    cliff_gate_list = [qi.Clifford(gate) for gate in gate_list]

    qr = QuantumRegister(n, 'q')
    circ = QuantumCircuit(qr)

    mid_encoding = qi.Clifford(circ)
    mid_gate = qi.Clifford(circ)

    new_error_list = []
    for i, (gate, mapping) in enumerate(zip(reversed(cliff_gate_list), reversed(qubit_sets_to_apply_to))):
        new_error_list.append([])
        for error in gate_error_list:
            exp_error = _error_to_qiskit(error, mapping, n)
            new_error = exp_error.evolve(mid_gate, frame='s')
            print(exp_error, new_error)
            new_error_list[i].append(new_error.to_label())
        mid_gate = mid_gate.compose(gate, qargs=mapping, front=True)
        #print(mapping)
        #print(gate.to_circuit())
        #print(mid_gate.to_circuit())

    return new_error_list[::-1]

def _error_to_qiskit(error, mapping, n):

    pauli = ['I'] * n
    list_error = list(error)
    for op, qb in zip(list_error, mapping):
        pauli[n - qb - 1] = op

    new_error = ''.join(pauli)
    return qi.Pauli(new_error)

def get_random_parity_check_matrix(n, s):
    """
    Creates a random 2n x s parity check matrix. Note that s = n - k.

    Parameters
    ----------
    n : int
        Number of qubits.
    s : int
        Number of stabilizers.

    Returns
    -------
    2D array
        2n x s parity check matrix.

    Notes
    -----
    Since the stabilizers need to commute with each other, and they must 
    generate the stabilizer set, it is easier to just use the Qiskit methods, 
    instead of doing something like in `get_random_error_matrix()`.
    """

    k = n - s
    
    parity_check_matrix = qi.random_clifford(int(n)).stabilizer[k:].array.astype(int)

    return parity_check_matrix

def get_statistics(n, k, num_gates, num_local_qubits=2):

    line_stats = {}
    for i in range(n):
        if i < k:
            line_stats[i] = set()
        else:
            line_stats[i] = set([i])

    count_stats = np.zeros(n-k+1, dtype=int)

    for _ in range(num_gates):
        mapping = np.random.choice(n, num_local_qubits, replace=False)
        s = set().union(*[line_stats[i] for i in mapping])
        for i in mapping:
            line_stats[i] = s
        count_stats[len(s)] += 1

    return count_stats
