# Author: Andrew Malta 2018
import numpy as np

class Weighted_Average_ES:
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

    def set_theta(self, theta):
        self.theta = np.array(theta)

class Winner_ES:
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

    def set_theta(self, theta):
        self.theta = np.array(theta)

class CMA_ES:
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

    def set_theta(self, theta):
        self.mean = np.array(theta)

    def result(self):
        return self.mean

''' helper functions taken from OpenAI's implementation of their algorithm which
can be found here:
(https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
'''
def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks

def compute_centered_ranks(x):
    """
    https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
    """
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y

def compute_weight_decay(weight_decay_coef, theta):
    """
    compute the weight decay for the model parameters. Used to encourage the 
    mean model parameter squared to not get too large. 
    """
    flattened = np.array(theta)
    mean_sq_theta = np.mean(flattened * flattened, axis=1)
    return -weight_decay_coef * mean_sq_theta

class OpenES:
    ''' Basic Version of OpenAI Evolution Strategy.'''
    def __init__(self, args):
        self.theta_dim = args["theta_dim"]
        self.sigma = args["sigma_start"]
        self.sigma_init = args["sigma_start"]
        self.sigma_lower_bound = args["sigma_lower_bound"]
        self.sigma_mult = args["sigma_mult"]
        self.alpha = args["alpha"]
        self.alpha_mult = args["alpha_mult"]
        self.alpha_lower_bound = args["alpha_lower_bound"]
        self.gen_size = args["gen_size"]
        self.rewards = np.zeros(self.gen_size)
        self.theta = np.zeros(self.theta_dim)
        self.best_theta = np.zeros(self.theta_dim)
        self.best_reward = -float("inf") # lowest possible value
        self.forget_best = args["forget_best"]
        self.weight_decay = args["weight_decay"]
        self.rank_fitness = args["rank_fitness"]
        if self.rank_fitness: # use rank rather than fitness numbers
            self.forget_best = True # forget the best one if we use rank fitness

    def ask(self):
        '''returns a collection of model weights'''    
        self.epsilon = np.random.randn(self.gen_size, self.theta_dim)
        self.solutions = self.theta.reshape(1, self.theta_dim) + self.epsilon * self.sigma

        return self.solutions

    def tell(self, simulation_results):
        '''updates internal variables based on the results from simulating the results from
           self.ask '''

        # input must be a numpy float array of the correct length
        if len(simulation_results) != self.gen_size:
            print "incorrect length of input"
            return None

        rewards = np.array(simulation_results)
        
        if self.rank_fitness:
            rewards = compute_centered_ranks(rewards)
        
        # decay the weights of our neural network using
        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
            rewards += l2_decay

        index_ordering = np.argsort(rewards)[::-1]

        best_reward = rewards[index_ordering[0]]
        best_theta = self.solutions[index_ordering[0]]

        self.curr_best_reward = best_reward
        self.curr_best_theta = best_theta

        # update our best parameters we are storing
        if self.forget_best or (self.curr_best_reward > self.best_reward):
            self.best_theta = best_theta
            self.best_reward = self.curr_best_reward

        # standardize the rewards to have a gaussian distribution
        normalized_rewards = (rewards - np.mean(rewards)) / np.std(rewards)
        change_theta = 1./(self.gen_size*self.sigma)*np.dot(self.epsilon.transpose(), normalized_rewards)
        
        self.theta += self.alpha * change_theta

        # adapt sigma if not already too small
        if (self.sigma > self.sigma_lower_bound):
            self.sigma *= self.sigma_mult

        # adapt learning rate if not already too small
        if (self.alpha > self.alpha_lower_bound):
            self.alpha *= self.alpha_mult

    def current_param(self):
        return self.curr_best_theta

    def set_theta(self, theta):
        self.theta = np.array(theta)

    def result(self):
        return self.best_theta