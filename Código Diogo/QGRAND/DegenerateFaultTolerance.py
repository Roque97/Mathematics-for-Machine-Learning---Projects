from QGRAND.FaultTolerance import FaultTolerance
from qiskit import QuantumRegister, QuantumCircuit
import qiskit.quantum_info as qi
import re
import numpy as np
from scipy.special import comb
from itertools import combinations, product
from copy import deepcopy
import warnings

class DegenerateFaultTolerance(FaultTolerance):

    # def _get_syndrome(self, error_str):

    #     error = list(error_str)
    #     if error[0] == '-':
    #         error = error[1:]
    #     error = error[self.n:]
    #     syndrome = np.zeros(self.n_total - self.n, dtype=int)
    #     for i, op in enumerate(error):
    #         if op in ['X','Y']:
    #             syndrome[i] = 1
    #     return syndrome

    # def get_base_syndromes(self, show=False):

    #     self.base_syndromes = np.zeros((self.n_cx, len(self.gate_error_list), self.n_total-self.n), dtype=int)
    #     for i, ev_errors in enumerate(self.ev_error_list):
    #         for j, error in enumerate(ev_errors):
    #             self.base_syndromes[i,j] = self._get_syndrome(error)
    #             print(i, j, error, self.base_syndromes[i,j]) if show else None

    def get_base_errors(self):

        self.base_errors = np.zeros((self.n_cx, len(self.gate_error_list), 2*self.n), dtype=int)
        for i, _ in enumerate(self.base_errors):
            for j, _ in enumerate(self.base_errors[0]):
                error = self.ev_error_list[i][j]
                if error[0] == '-':
                    error = error[1:]
                error = error[:self.n]

                # Convert error to int array
                array = np.zeros(2*self.n, dtype=int)
                for ind, op in enumerate(reversed(error)):
                    if op == 'I':
                        continue
                    if op != 'X':
                        array[ind] = 1
                    if op != 'Z':
                        array[self.n+ind] = 1

                # # Convert int array to int
                # out = 0
                # for bit in array:
                #     out = (out << 1) | bit
                # int_error = out

                self.base_errors[i,j] = array


    def syndrome_generator(self, max_t, n_errors):

        n, k = self.n, self.k
        n_cx = self.n_cx
        n_total = self.n_total

        array = np.zeros(n_total-n, dtype=int)
        err_array = np.zeros(2*n, dtype=int)
        # paulis = ['X', 'Y', 'Z']
        for dd in range(1, max_t+1):
            combos = combinations(range(n_cx), dd)
            all_ops = product(range(1, n_errors+1), repeat=dd)
            case = product(all_ops, combos)
            for error_nums, inds in case:
                for ind_err, ind_cx in zip(error_nums, inds):
                    syndrome = self.syndrome_power_set(ind_err, ind_cx)
                    array = (array + syndrome) % 2
                    error = self.error_power_set(ind_err, ind_cx)
                    err_array = (err_array + error) % 2
                yield dd, array, err_array
                array[:] = 0
                err_array[:] = 0

    # def syndrome_power_set(self, ind_err, ind_cx):
    #     syndromes = self.base_syndromes[ind_cx]
    #     array = np.zeros_like(syndromes[0])
    #     bin_rep = np.array(list(np.binary_repr(ind_err).zfill(len(self.gate_error_list)))).astype(int)
    #     for i, ind in enumerate(bin_rep):
    #         if ind==1:
    #             array = (array + syndromes[i]) % 2
    #     return array

    def error_power_set(self, ind_err, ind_cx):
        syndromes = self.base_errors[ind_cx]
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

        # if fill_syndromes:
        #     syndrome_list = np.zeros(2**(n_total-n), dtype=bool)
        #     syndrome_list[0] = True
        # else:
        seen_syndromes = set([0])
        syndrome_error_dict = {0:0}

        accounted_syndromes = 1
        error_type = np.zeros(n_show, dtype=int)
        probabilities = np.zeros((n_show, prob.shape[0]))

        probability_no_error = np.array([(1-p)**n_cx for p in prob])

        for dd, syndrome, error in self.syndrome_generator(max_t, n_cx_errors):

            out = 0
            for bit in syndrome:
                out = (out << 1) | bit
            int_syndrome = out

            out = 0
            for bit in error:
                out = (out << 1) | bit
            int_error = out

            # if fill_syndromes:
            #     if not syndrome_list[int_syndrome]:
            #         syndrome_list[int_syndrome] = True
            #         accounted_syndromes += 1

            #         error_type[dd-1] += 1
            #         probabilities[dd-1] += (prob/n_cx_errors)**dd * (1-prob)**(n_cx-dd)

            #         if accounted_syndromes == 2**(n_total-n):
            #             break
            # else:
            if int_syndrome not in seen_syndromes:
                seen_syndromes.add(int_syndrome)
                syndrome_error_dict[int_syndrome] = int_error
                accounted_syndromes += 1

                error_type[dd-1] += 1

                probabilities[dd-1] += (prob/n_cx_errors)**dd * (1-prob)**(n_cx-dd)

                if accounted_syndromes == 2**(n_total-n):
                    break
                
            elif int_error == syndrome_error_dict[int_syndrome]:

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
