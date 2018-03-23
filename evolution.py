# Author: Andrew Malta 2018
import numpy as np

class Weighted_Average_ES():
    def __init__(self, kwargs):
        self.theta_dim = kwargs["theta_dim"]
        self.gen_size = kwargs["gen_size"]
        self.sigma = kwargs["sigma"]
        self.alpha = kwargs["alpha"]
        self.n = kwargs["n"]
        self.theta = np.zeros(self.theta_dim)

    def ask(self):
        self.epsilon = np.random.randn(self.gen_size, self.theta_dim)
        self.solutions = self.theta.reshape(1, self.theta_dim) + self.epsilon * self.sigma

        return self.solutions

    def tell(self, fitnesses):
        argsort = list(np.argsort([-val for val in fitnesses]))
        fitness_update_sum = np.zeros(self.theta_dim)

        for i in argsort[0:n]:
            fitness_update_sum += (self.epsilon[i] * fitnesses[i])

        self.theta = self.theta + (self.alpha / self.n * self.sigma) * fitness_update_sum

    def result(self):
        return self.theta

class Winner_ES():
    def __init__(self, kwargs):
        self.theta_dim = kwargs["theta_dim"]
        self.gen_size = kwargs["gen_size"]
        self.sigma = kwargs["sigma"]
        self.theta = np.zeros(self.theta_dim)

    def ask(self):
        self.epsilon = np.random.randn(self.gen_size, self.theta_dim)
        self.solutions = self.theta.reshape(1, self.theta_dim) + self.epsilon * self.sigma

        return self.solutions

    def tell(self, fitnesses):
        argsort = list(np.argsort([-val for val in fitnesses]))
        self.theta = self.solutions[argsort[0]]

    def result(self):
        return self.theta

class CMA_ES():
    def __init__(self, kwargs):
        self.theta_dim = kwargs["theta_dim"]
        self.gen_size = kwargs["gen_size"]
        self.start_sigma = kwargs["start_sigma"]
        self.mu = kwargs["mu"]

        self.theta = np.zeros(self.theta_dim)

        self.n = self.theta_dim  # parameter for ease of readability
        self.lamb = self.gen_size # use same notation as wikipedia

        # initialization of the 5 major parameters that we update
        self.mean = np.zeros(self.n)
        self.sigma = self.start_sigma
        self.C = np.identity(self.n)
        self.p_sigma = np.zeros(self.n)
        self.p_c = np.zeros(self.n)

    #TODO: Vectorize this code for perf
    def ask(self):
        self.solutions = []
        for _ in xrange(self.gen_size):
            self.solutions.append(np.random.multivariate_normal(
                self.mean, self.sigma * self.sigma * self.C))

        return self.solutions

    def tell(self, fitnesses):
        tmp = self.mean
        sorted_indices = np.argsort(fitnesses)

         # update m
        self.mean = 0
        # using equal weighting 
        for i in sorted_indices[0:self.mu]:
            self.mean += self.solutions[i] / self.mu

        # update p_sigma
        c_sigma = 3. / self.n
        df = 1 - c_sigma # discount factor
        C_tmp = np.sqrt(np.linalg.inv(self.C))
        displacement = (self.mean - tmp) / float(self.sigma)

        self.p_sigma = (df * self.p_sigma + 
            np.sqrt(1 - np.power(df,2)) * np.sqrt(self.mu) * C_tmp * displacement)

        # update p_c
        c_c = 4. / self.n
        df2 = (1 - c_c)
        alpha = 1.5

        norm = np.linalg.norm(self.p_sigma)
        indicator = norm >= 0 and norm <= alpha * np.sqrt(self.n)
        complements = np.sqrt(1 - np.power((1 - c_c), 2))
        neutral_selection = np.sqrt(self.mu) * displacement
        self.p_c = df2 * self.p_c + indicator * (complements * neutral_selection)

        # update Covariance Matrix
        c_1 = 2. / np.power(self.n, 2)
        c_mu = self.mu / np.power(self.n, 2)
        c_s = (1 - indicator)* c_1 * c_c * (2 - c_c)

        tmp_pc = self.p_c.reshape(-1, 1)
        rank_1_mat = c_1 * tmp_pc.dot(tmp_pc.transpose())

        mat_sum = np.zeros(self.C.shape)
        for i in xrange(self.mu):
            vec = ((self.solutions[sorted_indices[i]] - tmp) / self.sigma).reshape(-1, 1)
            mat_sum += (1. / self.mu) * vec.dot(vec.transpose())

        self.C = (1 - c_1 - c_mu + c_s) * self.C + rank_1_mat + c_mu * mat_sum 

        # update sigma
        d_sigma = 1  # dampening parameter
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        tmp = (1 - 1. / (4 * self.n) + 1. / (21 * self.n * self.n))
        nse = np.sqrt(self.n) * tmp  # neutral selection expectation
        self.sigma = self.sigma * np.exp((c_sigma / d_sigma) * ((p_sigma_norm / nse) - 1))

    
    def result(self):
        return self.mean



