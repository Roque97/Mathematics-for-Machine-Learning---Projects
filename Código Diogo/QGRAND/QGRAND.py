import numpy as np

# Import Qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer#, transpile
#from qiskit.tools.visualization import plot_histogram, plot_state_city
import qiskit.quantum_info as qi
import qiskit as qk

#from IPython.display import clear_output

#import matplotlib.pyplot as plt
#plt.rc('text',usetex=True)
#plt.rc('font', family='serif', size=16)
#plt.rcParams['figure.figsize'] = [10, 6]
#from datetime import datetime
#from time import time

from scipy.interpolate import interp1d
from scipy.special import comb
from itertools import combinations, product

from .hamming import hamming_bound, max_distance_hamming
from .encoding import get_random_encoding
from .pauli import pauli_generator, get_numbered_noise

class QGRAND:

    def __init__(self,
                n=None,
                k=None,
                num_gates=None,
                num_local_qubits=2,
                noise_statistics=None,
                max_iterations=None,
                compress_circuit=False,
                backend=None):

        self.n, self.k, self.num_gates = n, k, num_gates
        self.num_local_qubits = num_local_qubits

        if noise_statistics is not None:
            if isinstance(noise_statistics[0], str):
                self.noise = noise_statistics
                self.noise_probabilities = [1/len(self.noise) for _ in self.noise]
                self.noise_statistics = [(prob, error) for prob, error in zip(self.noise_probabilities, self.noise)]

            else:
                self.noise_statistics = noise_statistics
                self.noise = [error for prob, error in noise_statistics]
                self.noise_probabilities = [prob for prob, error in noise_statistics]

        self.qb = QuantumRegister(n, 'q')
        self.anc = QuantumRegister(n-k, 'a')
        self.cb = ClassicalRegister(n-k, 'c')
        self.max_iterations = max_iterations if max_iterations is not None else 3*n+2
        self.compress_circuit = compress_circuit

        if backend is None:
            self.backend = Aer.get_backend('aer_simulator_stabilizer')
        else:
            self.backend = Aer.get_backend(backend)

        self.syndrome_circuit = None
        self.initial_circuit = None

        self._hamming_f = self._get_hamming_bound_interp()

    @staticmethod
    def _get_hamming_bound_interp():
        code_array = np.linspace(0,1,101)
        h_list = hamming_bound(code_array)
        _hamming_f = interp1d(h_list, code_array)
        return _hamming_f


    def get_encoding(self, fast=False):

        if fast:
            stabs, self.encoding = get_random_encoding(self.n,
                                            self.k,
                                            self.num_gates,
                                            qr = self.qb,
                                            stabilizer_format = 'both',
                                            return_logical_operators = False,
                                            return_circuit = False,
                                            return_encoding = False)
            self.parity_check_matrix, self.stabilizers = stabs

        else:
            stabs, los, self.circuit, self.encoding = get_random_encoding(
                                            self.n,
                                            self.k,
                                            self.num_gates,
                                            qr = self.qb,
                                            stabilizer_format = 'both',
                                            return_logical_operators = True,
                                            return_circuit = True,
                                            return_encoding = True)
            self.parity_check_matrix, self.stabilizers = stabs
            self.logical_Xs, self.logical_Zs = los

        # n, k = self.n, self.k
        # num_stabs, num_local_qubits = self.num_gates, self.num_local_qubits
        # qbs_to_apply = [list(np.random.choice(n, num_local_qubits, replace=False)) for stab in range(num_stabs)]
        # stabs = [qi.random_clifford(num_local_qubits).to_circuit() for stab in range(num_stabs)]

        # if not fast:
        #     h_circ = QuantumCircuit(self.qb)
        #     h_circ.h([self.qb[i] for i in range(k)])

        # circ = QuantumCircuit(self.qb)
        # for stab, mapping in zip(stabs, qbs_to_apply):
        #     circ.append(stab, [self.qb[i] for i in mapping])

        # self.encoding = circ.to_instruction()

        # if not fast:
        #     self.circuit = QuantumCircuit(self.qb)
        #     self.circuit.append(self.encoding, self.qb)

        # # Here we remove the first k stabilizers
        # # (I don't have guarantees that they are the right stabilizers to remove,
        # # but my tests suggest that Qiskit already orders them)
        # result_Z = qi.Clifford(circ).stabilizer

        # self.parity_check_matrix = result_Z[k:].array.astype(int)

        # result_Z = result_Z.to_labels()

        # stabilizers = result_Z[k:]
        # self.stabilizers = stabilizers

        # if not fast:

        #     result_X = qi.Clifford(h_circ + circ).stabilizer.to_labels()

        #     logical_Zs = result_Z[:k]
        #     logical_Xs = result_X[:k]
        #     assert set(stabilizers) == set(result_X[k:]), "The stabilizers don't match."
        #     self.logical_Zs = logical_Zs
        #     self.logical_Xs = logical_Xs

    def get_distance_from_stabilizers(self, rerun=False, by_matrix=False):
        n, k = self.n, self.k

        try:
            hamm = self.hamming_distance
        except AttributeError:
            self.hamming_distance = max_distance_hamming(n,k, self._hamming_f)
            hamm = self.hamming_distance

        distance_Z = min([n - op.count('I') for op in self.logical_Zs])
        distance_X = min([n - op.count('I') for op in self.logical_Xs])
        
        distance_max = min(distance_Z, distance_X, hamm)
        # print("Hamming bound is d = {}.".format(hamm))
        if distance_max == 1:
            return distance_max
        else:
            if by_matrix:
                if not rerun:
                    try:
                        parity_check_matrix = self.parity_check_matrix
                    except AttributeError:
                        self.get_parity_check_matrix()
                        parity_check_matrix = self.parity_check_matrix
                else:
                    self.get_parity_check_matrix()
                    parity_check_matrix = self.parity_check_matrix

                for dd, error in pauli_generator(n, distance_max):
                    error_vector = np.zeros((1, 2*n), dtype=int)
                    for j, op in enumerate(list(error)):
                        error_vector[0, j] = 1 if op == 'Z' or op == 'Y' else 0
                        error_vector[0, n+j] = 1 if op == 'X' or op == 'Y' else 0
                    if np.sum(error_vector @ parity_check_matrix.T % 2) == 0:
                        return dd
                return distance_max

            else:
                stabilizers = qi.StabilizerTable.from_labels(self.stabilizers)
                for dd, error in pauli_generator(n, distance_max):
                    if len(stabilizers.anticommutes_with_all(error)) == 0:
                        distance = dd
                        return distance
                return distance_max

    def get_distance(self, rerun=False, by_matrix=False):
        if rerun:
            self.get_encoding()
        return self.get_distance_from_stabilizers(rerun=rerun, by_matrix=by_matrix)

    def apply_error(self, error=None):
        if error is None:
            errors = [''.join([np.random.choice(['X','Y','Z']), str(np.random.randint(self.qb.size))])]
        else:
            errors = error.split()
        self.errors = errors
        circuit = QuantumCircuit(self.qb)
        for err in errors:
            if err[0] == 'X':
                circuit.x(self.qb[int(err[1:])])
            elif err[0] == 'Z':
                circuit.z(self.qb[int(err[1:])])
            elif err[0] == 'Y':
                circuit.y(self.qb[int(err[1:])])
        self.error_circuit = circuit.to_instruction()
        self.circuit.barrier(self.qb)
        self.circuit.append(self.error_circuit, self.qb)

    def apply_stabilizers(self):
        anc, qb = self.anc, self.qb
        
        self.circuit.barrier(anc, qb)

        if self.syndrome_circuit is None:
            syndrome_circuit = QuantumCircuit(anc,qb)
            syndrome_circuit.h(anc)
            syndrome_circuit.barrier(anc)
            
            for i, stab in enumerate(self.stabilizers):
                syndrome_circuit.barrier(anc, qb) if i!=0 else None
                for j, pauli in enumerate(reversed(list(stab)[1:])):
                    if pauli == 'X':
                        syndrome_circuit.cx(anc[i], qb[j])
                    elif pauli == 'Y':
                        syndrome_circuit.cy(anc[i], qb[j])
                    elif pauli == 'Z':
                        syndrome_circuit.cz(anc[i], qb[j])
            syndrome_circuit.barrier(anc)
            syndrome_circuit.h(anc)
            for i, stab in enumerate(self.stabilizers):
                if stab[0] == '-':
                    syndrome_circuit.x(anc[i])
            syndrome_circuit.barrier(anc)

            self.syndrome_circuit = syndrome_circuit.to_instruction()

        self.circuit.append(self.syndrome_circuit, (*anc,*qb))


    def apply_error_correction(self, previous_error, next_error):
        qb, cb = self.qb, self.cb
        n, k = self.n, self.k
        correction_circuit = QuantumCircuit(qb, cb)

        if previous_error is not None:
            if previous_error[0] == 'X':
                correction_circuit.x(qb[int(previous_error[1:])])
            elif previous_error[0] == 'Z':
                correction_circuit.z(qb[int(previous_error[1:])])
            elif previous_error[0] == 'Y':
                correction_circuit.y(qb[int(previous_error[1:])])

        if next_error is not None:
            for res in range(1, 2**(n-k)):
                if next_error[0] == 'X':
                    correction_circuit.x(qb[int(next_error[1:])]).c_if(cb, res)
                elif next_error[0] == 'Z':
                    correction_circuit.z(qb[int(next_error[1:])]).c_if(cb, res)
                elif next_error[0] == 'Y':
                    correction_circuit.y(qb[int(next_error[1:])]).c_if(cb, res)

        self.correction_circuit = correction_circuit.to_instruction()

        self.circuit.append(self.correction_circuit, qb, cb)

    def undo_correction_attempt(self, previous_error):
        qb, cb = self.qb, self.cb
        n, k = self.n, self.k
        correction_circuit = QuantumCircuit(qb, cb)

        if previous_error[0] == 'X':
            correction_circuit.x(qb[int(previous_error[1:])])
        elif previous_error[0] == 'Z':
            correction_circuit.z(qb[int(previous_error[1:])])
        elif previous_error[0] == 'Y':
            correction_circuit.y(qb[int(previous_error[1:])])

        self.correction_circuit = correction_circuit.to_instruction()

        self.circuit.append(self.correction_circuit, qb, cb)

    def try_correction(self, next_error, conditional=True):
        qb, cb = self.qb, self.cb
        n, k = self.n, self.k
        correction_circuit = QuantumCircuit(qb, cb)

        if conditional:

            for res in range(1, 2**(n-k)):
                if next_error[0] == 'X':
                    correction_circuit.x(qb[int(next_error[1:])]).c_if(cb, res)
                elif next_error[0] == 'Z':
                    correction_circuit.z(qb[int(next_error[1:])]).c_if(cb, res)
                elif next_error[0] == 'Y':
                    correction_circuit.y(qb[int(next_error[1:])]).c_if(cb, res)

        else:
            if next_error[0] == 'X':
                correction_circuit.x(qb[int(next_error[1:])])
            elif next_error[0] == 'Z':
                correction_circuit.z(qb[int(next_error[1:])])
            elif next_error[0] == 'Y':
                correction_circuit.y(qb[int(next_error[1:])])

        self.correction_circuit = correction_circuit.to_instruction()

        self.circuit.append(self.correction_circuit, qb, cb)

    def set_circuit(self, force=False):

        if self.initial_circuit is None:
            circuit = QuantumCircuit(self.qb, self.anc, self.cb)
            circuit.append(self.encoding, self.qb)
            circuit.append(self.error_circuit, self.qb)
            self.initial_circuit = circuit

        if force:
            self.circuit = self.initial_circuit.copy()


    def apply_QGRAND(self):
        #n, k = self.n, self.k
        self.results = []
        anc, cb, qb = self.anc, self.cb, self.qb
        n, k = self.n, self.k
        noise = ''.join(['I' for _ in range(n)]) + self.noise
        self.set_circuit()
        success = False
        ind = 0
        
        for ii in range(self.max_iterations):

            self.apply_stabilizers()

            self.circuit.measure(anc, cb)
            
            job = qk.execute(self.circuit, shots=1, backend=self.backend)
            last_result = list(job.result().get_counts().keys())[0]
            numbered_noise = get_numbered_noise(noise[ind])
            print("Iteration: {}\t Testing: {}\t Syndrome: {}".format(ii, numbered_noise, last_result))
            self.results.append((ii, last_result))
            
            if last_result == '0'*(n-k):
                print('QGRAND has corrected the error after {} iterations.'.format(ii+1))
                print('The corrected error was '+numbered_noise+'.')
                success = True
                break
                
            self.circuit.reset(anc)

            #self.apply_error_correction(noise[ind] if ind>0 else None, noise[ind+1] if ind<len(noise) else None)
            self.undo_correction_attempt(noise[ind]) if ind>0 else None
            self.try_correction(noise[ind+1]) if ind<len(noise)-1 else None
            ind += 1
            if ind == len(noise):
                break
         
        if not success:        
            print('QGRAND could not correct the error in {} iterations.'.format(self.max_iterations))

    def apply_QGRAND_fast(self):
        #n, k = self.n, self.k
        self.results = []
        anc, cb, qb = self.anc, self.cb, self.qb
        n, k = self.n, self.k
        noise = ['I'] + self.noise
        success = False
        ind = 0

        # self.apply_stabilizers()
        # clifford = qi.Clifford(self.circuit)
        self.set_circuit()
        self.circuit = self.initial_circuit.copy()

        for ii in range(self.max_iterations-1):

            self.apply_stabilizers()

            self.circuit.measure(anc, cb)
            
            job = qk.execute(self.circuit, shots=1, backend=self.backend)
            last_result = list(job.result().get_counts().keys())[0]
            print("Iteration: {}\t Testing: {}\t Syndrome: {}".format(ii, noise[ind], last_result))
            self.results.append((ii, last_result))
            
            if last_result == '0'*(n-k):
                print('QGRAND has corrected the error after {} iterations.'.format(ii+1))
                print('The corrected error was '+noise[ind]+'.')
                success = True
                break

            self.circuit = self.initial_circuit.copy()
                
            self.try_correction(noise[ind+1], conditional=False) if ind<len(noise)-1 else None
            ind += 1
            if ind == len(noise):
                break
         
        if not success:        
            print('QGRAND could not correct the error in {} iterations.'.format(self.max_iterations))

    def get_parity_check_matrix(self):

        n = self.n
        parity_check_matrix = np.zeros((len(self.stabilizers), 2*n), dtype=int)

        for i, stabilizer in enumerate(self.stabilizers):
            # The sign is only necessary when implementing the stabilizers. The math is indifferent to it.
            for j, pauli in enumerate(list(stabilizer)[1:]):
                if pauli == 'I':
                    continue
                else:
                    if pauli != 'Z':
                        parity_check_matrix[i, j] = 1
                    if pauli != 'X':
                        parity_check_matrix[i, n+j] = 1

        self.parity_check_matrix = parity_check_matrix

    def get_syndrome_table(self, error_matrix=None):

        if error_matrix is None:
            try:
                error_matrix = self.error_matrix
            except AttributeError:
                self.get_error_matrix()
                error_matrix = self.error_matrix
        
        try:
            self.syndrome_table = error_matrix @ self.parity_check_matrix.T % 2
        except AttributeError:
            self.get_parity_check_matrix()
            self.syndrome_table = error_matrix @ self.parity_check_matrix.T % 2

    def get_int_syndrome_vector(self):

        int_syndrome_vector = np.zeros(self.syndrome_table.shape[0], dtype=int)

        for i, syndrome in enumerate(self.syndrome_table):
            out = 0
            for bit in syndrome:
                out = (out << 1) | bit
            int_syndrome_vector[i] = out

        self.int_syndrome_vector = int_syndrome_vector

    def get_error_to_syndrome_mapping(self):

        self.error_to_syndrome_mapping = [(syndrome, get_numbered_noise(error)) for syndrome, error in zip(self.int_syndrome_vector, self.noise)]

    def get_error_to_syndrome_prob_mapping(self):

        self.error_to_syndrome_prob_mapping = [
        (syndrome, get_numbered_noise(error), prob) for syndrome, error, prob in zip(self.int_syndrome_vector, self.noise, self.noise_probabilities)]

    def get_syndrome_table_with_leaders(self):

        sorted_table = sorted(self.error_to_syndrome_prob_mapping, key=lambda x: x[0])

        table_with_leaders = []
        syndrome = None
        for error in sorted_table:
            error_syndrome, error_str, prob = error
            if error_syndrome != syndrome:
                syndrome = error_syndrome
                table_with_leaders.append([error_syndrome, []])
            table_with_leaders[-1][-1].append((error_str, prob))

        # print(table_with_leaders)
        
        final_table = []
        for syndrome, errors in table_with_leaders:
            sorted_errors = sorted(errors, key=lambda x: -x[1])
            # prob = sum(error[1] for error in errors)
            prob = sorted_errors[0][1]
            only_errors = [error for error, _ in sorted_errors]
            final_table.append([syndrome, prob, only_errors])

        self.syndrome_table_with_leaders = sorted(final_table, key=lambda x: -x[1])

    def get_error_rate(self, p=None):

        failure_rate = 0.
        for _, _, errors in self.syndrome_table_with_leaders:
            missed_errors = errors[1:]
            for error in missed_errors:
                n_errors = len(error.split())
                failure_rate += p**n_errors

        self.failure_rate = failure_rate

    def get_decision_tree(self, measured_syndromes=None, measurements=None):

        if measured_syndromes is None:
            self.get_syndrome_table()

        good_rows = self.syndrome_table[:, measured_syndromes] == measurements

        working_table = self.syndrome_table[good_rows]

        
        decision_tree = []
        measured_stabilizers = []
        n_layers = 1
        #unique_syndrome_table = np.unique(self.syndrome_table, axis=0)

        next_stabilizer = np.argmax(shannon_entropy(self.syndrome_table))

        decision_tree.append([next_stabilizer, []])
        measured_stabilizers.append(next_stabilizer)

        next_stabilizer = np.argmax(shannon_entropy(working_table * self.noise_probabilities[good_rows,None]))
        measured_syndromes += [next_stabilizer]

        #self.get_decision_tree(measured_syndromes, measurements+[0]), self.get_decision_tree(measured_syndromes, measurements+[1])


def shannon_entropy(p):
    return -np.sum(np.where(p==0., 0., p * np.log2(p)), axis=0)
        

def apply_iteration_old(circuit, n, k, qb, stabilizers, max_iter):
    anc, cb = [], []
    for ind in range(max_iter):
        anc.append(QuantumRegister(n-k, 'a[{}]'.format(ind)))
        cb.append(ClassicalRegister(n-k, 'c[{}]'.format(ind)))
        circuit.add_register(anc[ind])
        circuit.add_register(cb[ind])        

        circuit = QGRAND.apply_stabilizers(circuit, qb, anc, stabilizers, ind)

        circuit.measure(anc[ind], cb[ind])

        if ind != max_iter-1:
            circuit.barrier(*anc, qb)
            for res in range(1, 2**(n-k)):
                circuit.x(qb[0]).c_if(cb[ind], res)
        
    return circuit







       
    
