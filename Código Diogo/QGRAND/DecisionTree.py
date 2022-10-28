# Work in progress

import numpy as np
from .pauli import pauli_generator
from QGRAND import QGRAND
import sys
import matplotlib.pyplot as plt   # now can import graphic package
#sys.stdout = open('Code_QIA.txt',"w")
plt.rc('text',usetex=True)
plt.rc('font', family='serif', size=24)
plt.rcParams['figure.figsize'] = [10, 6]

class DecisionTree:

    def __init__(self, x, logN, n, max_t):

        self.x, self.logN = x, logN
        self.n = n
        self.max_t = max_t

        self.syndrome_table = None

        self._set_default_error()
        self._set_min_iterations()

    def _set_default_error(self):
        errors = [''.join(['I' for j in range(self.n)])]
        errors += [error for _, error in pauli_generator(self.n,self.max_t+1)]
        self.errors = errors

    def _set_min_iterations(self):
        n_errors = len(self.errors)
        min_iters = self.lower_bound_iters(n_errors, self.n-self.k)
        min_iters_sim, q_p, q_m = self.lower_bound_sim(n_errors, self.n-self.k, runs=10_001)


    def get_number_of_iterations(self):
        self.n_iters = [[self.lower_bound_iters(float(10**j), i) for i in 2**(2**self.x)] for j in self.logN]

    def plot_number_of_iterations(self, save = False, filename = 'decision_tree_savings'):
        fig = plt.figure()
        for i, plot in enumerate(self.n_iters):
            plt.plot(self.x, plot, label='$N=10^{}$'.format(self.logN[i]))
        plt.legend()
        plt.ylabel('Number of iterations')
        plt.xlabel('$\log_2 \log_2 s$')
        plt.show()

        filename = 'Plots/'+filename+'.pdf'
        fig.savefig(filename, bbox_inches='tight') if save else None

    def get_fit(self):
        data = np.array(self.n_iters).T
        N_array = np.log2(10**self.logN)
        soln = self.polyfit2d(N_array, self.x, data, kx=1, ky=1, order=1)
        return soln

    @staticmethod
    def lower_bound_iters(N, S):
        N -= 1
        f = lambda N, S: N/2 - np.sqrt(N * np.log(S) / 2)
        n_iters = 0
        while N > 0.5:
            N = f(N, S-n_iters)
            n_iters += 1
            #print(N)
        return n_iters

    @staticmethod
    def lower_bound_sim(N, S, runs=31, conf=0.2):
        N -= 1
        iter_array = np.zeros(runs, dtype=int)
        for run in range(runs):
            min_data_sum = 0
            while min_data_sum == 0:
                data = np.random.randint(2, size=(N, S))
                min_data_sum = np.min(np.sum(data, axis=1))
            iters = 0
            while data.shape[0] > 0:
                #print(data)
                next_stabilizer = np.argmax(np.sum(data, axis=0))
                #print(next_stabilizer)
                good_rows = data[:, next_stabilizer] == 0
                #print(good_rows)
                data = data[good_rows]
                iters += 1
                #print(data.shape[0])
            iter_array[run] = iters
        return np.median(iter_array), np.quantile(iter_array, 1-conf), np.quantile(iter_array, conf)

    @staticmethod
    def shannon_entropy(p, probs):
        probs /= np.sum(probs)
        prob_1 = np.sum(p * probs[:,None], axis=0)
        entropy = -np.nan_to_num(prob_1*np.log2(prob_1)) - np.nan_to_num((1-prob_1)*np.log2(1-prob_1))
        return entropy, prob_1


    def get_decision_tree(self, measured_stabilizers=None, measurements=None, prob=1.):
        
        # print("--------------------------------------------------")
            
        if measured_stabilizers is None:
            measured_stabilizers = np.array([], dtype=int)
            measurements = np.array([], dtype=int)
            good_rows = np.full((syndrome_table.shape[0]), True, dtype=bool)
        else:
            #measured_syndromes, measurements = np.array(measured_syndromes), np.array(measurements)
            good_rows = np.squeeze(np.all(syndrome_table[:, measured_stabilizers] == measurements, axis=1))
        #print(measured_syndromes, measurements)
        #print(good_rows)

        working_table = syndrome_table[good_rows]
        assert len(measured_stabilizers) <= syndrome_table.shape[1], 'Stuck in loop.'

        if working_table.shape[0] <= 1:
            index = np.argwhere(good_rows)[0,0] if working_table.shape[0] == 1 else -1
            #print(prob, measured_stabilizers, measurements, index)
            return [(index, measured_stabilizers.shape[0])]
        
        entropy, prob_1 = self.shannon_entropy(working_table, noise_probabilities[good_rows])
        mask = np.zeros(entropy.size, dtype=bool)
        mask[measured_stabilizers] = True
        masked_entropy = np.ma.array(entropy, mask=mask)
        next_stabilizer = np.argmax(masked_entropy)
        if next_stabilizer in measured_stabilizers:
            print(working_table, working_table.shape[0], measured_stabilizers, measurements)
            print(good_rows)
            print(entropy, prob_1)
            print(next_stabilizer)
            print(noise_probabilities, noise_probabilities[good_rows])
            raise AssertionError("Repeated stabilizer.")
        measured_stabilizers = np.append(measured_stabilizers,[next_stabilizer])
        
        prob_1 = prob_1[next_stabilizer]
        
        #print(next_stabilizer, measured_syndromes, measurements)

        data_0 = self.get_decision_tree(measured_stabilizers, np.append(measurements,[0]), prob*(1-prob_1))
        data_1 = self.get_decision_tree(measured_stabilizers, np.append(measurements,[1]), prob*prob_1)
        
        return data_0 + data_1

    def shannon(alpha, N, eps=None):
        if eps is None:
            probs = np.exp(-alpha*np.arange(N))
            probs /= np.sum(probs)
        else:
            probs = np.full(N, eps/(N-1))
            probs[0] = 1 - eps 
        
        entropy = np.sum(-probs*np.log2(probs))   
        
        return entropy, probs

    def entropy_dependence(self, n, k, ty, alpha, eps, runs=3):
        
        global syndrome_table
        global noise_probabilities
        
        #n = 30
        #k = 1
        ty += 1
        errors = [''.join(['I' for j in range(n)])]
    #     errors += [''.join([('I' if j!=i else 'Y') for j in range(n)]) for i in range(n)]
    #     errors += [''.join([('I' if j!=i else 'Z') for j in range(n)]) for i in range(n)]
        
        errors += [error for d, error in pauli_generator(n,ty)]

        n_errors = len(errors)
        min_iters = self.lower_bound_iters(n_errors, n-k)
        min_iters_sim, q_p, q_m = self.lower_bound_sim(n_errors, n-k, runs=10_001)
        
        _, probs = self.shannon(alpha, n_errors)
        entropy, max_iters = -np.sum(np.nan_to_num(probs*np.log2(probs))), np.log2(probs.shape[0])

        noise = [(prob, error) for prob, error in zip(probs, errors)]

        iter_array = np.zeros(runs)
        for run in range(runs):
            trial = QGRAND(n = n, k = k, num_gates = 5000, noise_statistics=noise)
            trial.get_encoding()
            trial.get_syndrome_table()
            syndrome_table = trial.syndrome_table
            noise_probabilities = np.array(trial.noise_probabilities)

            syndrome_table, index = np.unique(syndrome_table, axis=0, return_index=True)
            noise_probabilities = noise_probabilities[index]
            
            #print(run, probs.shape[0], noise_probabilities.shape[0])

            iter_data = self.get_decision_tree()
            iter_array[run] = sum([iters * noise_probabilities[i] for i, iters in iter_data])
            #print(iter_data[0], noise_probabilities[0])
        
        # Uncomment to implement constant probabilities
        _, probs = self.shannon(alpha, n_errors, eps=eps)
        entropy_eps = -np.sum(np.nan_to_num(probs*np.log2(probs)))

        noise = [(prob, error) for prob, error in zip(probs, errors)]

        iter_array_eps = np.zeros(runs)
        for run in range(runs):
            trial = QGRAND(n = n, k = k, num_gates = 5000, noise_statistics=noise)
            trial.get_encoding()
            trial.get_syndrome_table()
            syndrome_table = trial.syndrome_table
            noise_probabilities = np.array(trial.noise_probabilities)

            syndrome_table, index = np.unique(syndrome_table, axis=0, return_index=True)
            noise_probabilities = noise_probabilities[index]
            
            #print(run, probs.shape[0], noise_probabilities.shape[0])

            iter_data = self.get_decision_tree()
            iter_array_eps[run] = sum([iters * noise_probabilities[i] for i, iters in iter_data])
            #print(iter_data[0], noise_probabilities[0])
        
        return entropy, entropy_eps, max_iters, min_iters, min_iters_sim, q_p, q_m, iter_array, iter_array_eps

    @staticmethod
    def polyfit2d(x, y, z, kx=3, ky=3, order=None):
        '''
        Two dimensional polynomial fitting by least squares.
        Fits the functional form f(x,y) = z.

        Notes
        -----
        Resultant fit can be plotted with:
        np.polynomial.polynomial.polygrid2d(x, y, soln.reshape((kx+1, ky+1)))

        Parameters
        ----------
        x, y: array-like, 1d
            x and y coordinates.
        z: np.ndarray, 2d
            Surface to fit.
        kx, ky: int, default is 3
            Polynomial order in x and y, respectively.
        order: int or None, default is None
            If None, all coefficients up to maxiumum kx, ky, ie. up to and including x^kx*y^ky, are considered.
            If int, coefficients up to a maximum of kx+ky <= order are considered.

        Returns
        -------
        Return paramters from np.linalg.lstsq.

        soln: np.ndarray
            Array of polynomial coefficients.
        residuals: np.ndarray
        rank: int
        s: np.ndarray

        '''

        # grid coords
        x, y = np.meshgrid(x, y)
        # coefficient array, up to x^kx, y^ky
        coeffs = np.ones((kx+1, ky+1))

        # solve array
        a = np.zeros((coeffs.size, x.size))

        # for each coefficient produce array x^i, y^j
        for index, (j, i) in enumerate(np.ndindex(coeffs.shape)):
            # do not include powers greater than order
            if order is not None and i + j > order:
                arr = np.zeros_like(x)
            else:
                arr = coeffs[i, j] * x**i * y**j
            a[index] = arr.ravel()

        # do leastsq fitting and return leastsq result
        return np.linalg.lstsq(a.T, np.ravel(z), rcond=None)