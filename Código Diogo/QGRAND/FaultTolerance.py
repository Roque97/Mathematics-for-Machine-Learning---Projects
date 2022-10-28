from QGRAND.encoding import get_random_encoding, _error_to_qiskit
from qiskit import QuantumRegister, QuantumCircuit
import qiskit.quantum_info as qi
import re
import numpy as np
from scipy.special import comb
from itertools import combinations, product
import warnings

class FaultTolerance:

    def __init__(self,
                n,
                k,
                num_gates,
                num_local_qubits = 2,
                gate_error_list = None,
                apply_encoding_errors = False,
                apply_stabilizer_errors = True):

        self.n, self.k, self.num_gates = n, k, num_gates
        self.num_local_qubits = num_local_qubits
        self.qr = QuantumRegister(n, 'q')
        self.anc = QuantumRegister(n-k, 'a')
        self._set_gate_error_list(gate_error_list)
        self.full_circuit = None
        self.apply_encoding_errors = apply_encoding_errors
        self.apply_stabilizer_errors = apply_stabilizer_errors

    def _set_gate_error_list(self, gate_error_list):

        if gate_error_list is None:
            self.gate_error_list = ['IX','IZ','XI','ZI']
        else:
            self.gate_error_list = gate_error_list

    def get_encoding(self):
        self.stabilizers, self.encoding = get_random_encoding(self.n,
                                        self.k,
                                        self.num_gates,
                                        self.num_local_qubits,
                                        self.qr,
                                        stabilizer_format='str',
                                        return_circuit=True)
        
        self.encoding = self.encoding.decompose().decompose()

    def get_syndrome_circuit(self, apply_barriers=False):
        anc, qr = self.anc, self.qr
        
        syndrome_circuit = QuantumCircuit(anc,qr)
        syndrome_circuit.h(anc)
        syndrome_circuit.barrier(anc) if apply_barriers else None
        
        if True:
            for i, stab in enumerate(self.stabilizers):
                syndrome_circuit.barrier(anc, qr) if i!=0 and apply_barriers else None
                for j, pauli in enumerate(reversed(list(stab)[1:])):
                    if pauli == 'X':
                        syndrome_circuit.cx(anc[i], qr[j])
                    elif pauli == 'Y':
                        syndrome_circuit.sdg(qr[j])
                        syndrome_circuit.cx(anc[i], qr[j])
                        syndrome_circuit.s(qr[j])
                    elif pauli == 'Z':
                        syndrome_circuit.h(qr[j])
                        syndrome_circuit.cx(anc[i], qr[j])
                        syndrome_circuit.h(qr[j])
        else:
            for i, stab in enumerate(self.stabilizers):
                syndrome_circuit.barrier(anc, qr) if i!=0 and apply_barriers else None
                for j, pauli in enumerate(reversed(list(stab)[1:])):
                    if pauli == 'X':
                        syndrome_circuit.h(qr[j])
                        syndrome_circuit.cz(anc[i], qr[j])
                        syndrome_circuit.h(qr[j])
                    elif pauli == 'Y':
                        syndrome_circuit.cy(anc[i], qr[j])
                    elif pauli == 'Z':
                        syndrome_circuit.cz(anc[i], qr[j])
        syndrome_circuit.barrier(anc) if apply_barriers else None
        syndrome_circuit.h(anc)
        for i, stab in enumerate(self.stabilizers):
            if stab[0] == '-':
                syndrome_circuit.x(anc[i])
        syndrome_circuit.barrier(anc) if apply_barriers else None

        self.syndrome_circuit = syndrome_circuit.to_instruction()

    def get_full_circuit(self, n_stab_runs=1):
        anc, qr = self.anc, self.qr

        ancs = list(reversed([QuantumRegister(self.n-self.k, 'a'+str(i)) for i in range(n_stab_runs)]))
        regs = ancs + [qr]
        full_circuit = QuantumCircuit(*regs)
        full_circuit.append(self.encoding, qr)

        # syndrome_circuit = QuantumCircuit(2*self.n-self.k)
        # syndrome_circuit.compose(self.syndrome_circuit, qubits=(*anc, *qr), inplace=True)
        # Test again
        for reg in reversed(ancs):
            full_circuit.append(self.syndrome_circuit, (*reg,*qr))
            #full_circuit.compose(self.syndrome_circuit, qubits=(*reg,*qr), inplace=True)
        full_circuit = full_circuit.decompose()
        # full_circuit = QuantumCircuit(anc,qr)
        # full_circuit.append(self.encoding, qr)
        # full_circuit.append(self.syndrome_circuit, (*anc,*qr))
        # full_circuit = full_circuit.decompose()

        # display(full_circuit.draw())

        self.full_qr = QuantumRegister(self.n + n_stab_runs*(self.n-self.k), 'f')
        self.full_circuit = QuantumCircuit(self.full_qr)
        self.full_circuit.compose(full_circuit, inplace=True)
        self.n_total = self.full_circuit.num_qubits


    def evolve_gate_errors(self, ignore_extra_stabs=0):
        full_circuit_qasm = self.full_circuit.qasm()

        split_qasm = full_circuit_qasm.split('\n')
        initial_qasm = split_qasm[:3]

        ev_error_list = []
        mid_qasm = []
        for line in reversed(split_qasm[3:]):
            if 'cx' in line:
                loc = re.findall('f\[(\d+)\]', line)
                mapping = [int(d) for d in loc]
                
                add_cx = True
                if not self.apply_encoding_errors:
                    if min(mapping) >= self.n_total - self.n:
                        add_cx = False
                if not self.apply_stabilizer_errors:
                    if min(mapping) < self.n_total - self.n:
                        add_cx = False
                else:
                    if min(mapping) < ignore_extra_stabs*(self.n-self.k):
                        add_cx = False
                if add_cx:
                    full_qasm = initial_qasm + list(reversed(mid_qasm))
                    mid_circuit = QuantumCircuit.from_qasm_str('\n'.join(full_qasm))
                    #print(mapping)
                    #print(mid_circuit)
                    mid_gate = qi.Clifford(mid_circuit)
                    ev_errors = self._get_evolved_errors(mapping, mid_gate)
                    ev_error_list.append(ev_errors)
            mid_qasm.append(line)

        self.ev_error_list = ev_error_list
        self.n_cx = len(ev_error_list)

    def _get_evolved_errors(self, mapping, mid_gate):

        ev_errors = []
        for error in self.gate_error_list:
            exp_error = _error_to_qiskit(error, mapping, self.n_total)
            new_error = exp_error.evolve(mid_gate, frame='s')
            #print(exp_error, new_error)
            ev_errors.append(new_error.to_label())

        return ev_errors

    def _get_syndrome(self, error_str):

        error = list(error_str)
        if error[0] == '-':
            error = error[1:]
        error = error[self.n:]
        syndrome = np.zeros(self.n_total - self.n, dtype=int)
        for i, op in enumerate(error):
            if op in ['X','Y']:
                syndrome[i] = 1
        return syndrome

    def get_base_syndromes(self, show=False):

        self.base_syndromes = np.zeros((self.n_cx, len(self.gate_error_list), self.n_total-self.n), dtype=int)
        for i, ev_errors in enumerate(self.ev_error_list):
            for j, error in enumerate(ev_errors):
                self.base_syndromes[i,j] = self._get_syndrome(error)
                print(i, j, error, self.base_syndromes[i,j]) if show else None

    def syndrome_generator(self, max_t, n_errors):

        n, k = self.n, self.k
        n_cx = self.n_cx
        n_total = self.n_total

        array = np.zeros(n_total-n, dtype=int)
        # paulis = ['X', 'Y', 'Z']
        for dd in range(1, max_t+1):
            combos = combinations(range(n_cx), dd)
            all_ops = product(range(1, n_errors+1), repeat=dd)
            case = product(all_ops, combos)
            for error_nums, inds in case:
                for ind_err, ind_cx in zip(error_nums, inds):
                    syndrome = self.syndrome_power_set(ind_err, ind_cx)
                    array = (array + syndrome) % 2
                yield dd, array
                array[:] = 0

    def syndrome_power_set(self, ind_err, ind_cx):
        syndromes = self.base_syndromes[ind_cx]
        array = np.zeros_like(syndromes[0])
        bin_rep = np.array(list(np.binary_repr(ind_err).zfill(len(self.gate_error_list)))).astype(int)
        for i, ind in enumerate(bin_rep):
            if ind==1:
                array = (array + syndromes[i]) % 2
        return array

    def get_weight_statistics(self,
                            prob=None,
                            max_t=None,
                            fill_syndromes=False,
                            n_cx_errors=15,
                            truncate=True):

        warnings.warn("""This function is a lower bound for the performance 
                        of the code. It assumes that errors are distinct. 
                        Therefore, it not only disregards possible degeneracies
                        in the main n qubits, but also the highly degenerate 
                        errors in the ancilla qubits.""")

        n, k, n_cx, n_total = self.n, self.k, self.n_cx, self.n_total

        if max_t is None:
            max_t = n_cx

        if truncate:
            n_show = max_t
        else:
            n_show = n_cx

        # Converting prob to numpy array
        if isinstance(prob, float):
            prob = [prob]

        if not isinstance(prob, np.ndarray):
            prob = np.array(prob)

        if fill_syndromes:
            syndrome_list = np.zeros(2**(n_total-n), dtype=bool)
            syndrome_list[0] = True
        else:
            seen_syndromes = set([0])

        accounted_syndromes = 1
        error_type = np.zeros(n_show, dtype=int)
        probabilities = np.zeros((n_show, prob.shape[0]))

        probability_no_error = np.array([(1-p)**n_cx for p in prob])

        for dd, syndrome in self.syndrome_generator(max_t, n_cx_errors):

            out = 0
            for bit in syndrome:
                out = (out << 1) | bit
            int_syndrome = out

            if fill_syndromes:
                if not syndrome_list[int_syndrome]:
                    syndrome_list[int_syndrome] = True
                    accounted_syndromes += 1

                    error_type[dd-1] += 1
                    probabilities[dd-1] += (prob/n_cx_errors)**dd * (1-prob)**(n_cx-dd)

                    if accounted_syndromes == 2**(n_total-n):
                        break
            else:
                if int_syndrome not in seen_syndromes:
                    seen_syndromes.add(int_syndrome)
                    accounted_syndromes += 1

                    error_type[dd-1] += 1

                    probabilities[dd-1] += (prob/n_cx_errors)**dd * (1-prob)**(n_cx-dd)

                    if accounted_syndromes == 2**(n_total-n):
                        break


        failure_rate = 1 - probability_no_error - np.cumsum(probabilities, axis=0)

        n_errors = np.array([n_cx_errors**i * comb(n_cx, i) for i in range(1, n_show+1)])
        correctable_fraction = error_type / n_errors

        mean_iterations = np.array([
            sum((prob/n_cx_errors)**(dd+1) * (1-prob)**(n_cx-dd-1) * nn * (
                    sum(error_type[:dd]) + (nn+1)/2
                ) for dd, nn in enumerate(error_type[:i])
            ) for i in range(1,n_show+1)
        ])

        return failure_rate, correctable_fraction, mean_iterations
