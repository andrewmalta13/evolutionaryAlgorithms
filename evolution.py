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

    
    def result(self):
        return self.mean

def compute_ranks(x):
  """
  Returns ranks in [0, len(x))
  Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
  (https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
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

def compute_weight_decay(weight_decay, model_param_list):
  model_param_grid = np.array(model_param_list)
  return - weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)

# base class for an optimizer
class Optimizer(object):
  def __init__(self, pi, epsilon=1e-08):
    self.pi = pi
    self.dim = pi.num_params
    self.epsilon = epsilon
    self.t = 0

  def update(self, globalg):
    self.t += 1
    step = self._compute_step(globalg)
    theta = self.pi.mu
    ratio = np.linalg.norm(step) / (np.linalg.norm(theta) + self.epsilon)
    self.pi.mu = theta + step
    return ratio

  def _compute_step(self, globalg):
    raise NotImplementedError

# Adam optimizer for use in PEPG
class Adam(Optimizer):
  def __init__(self, pi, stepsize, beta1=0.99, beta2=0.999):
    Optimizer.__init__(self, pi)
    self.stepsize = stepsize
    self.beta1 = beta1
    self.beta2 = beta2
    self.m = np.zeros(self.dim, dtype=np.float32)
    self.v = np.zeros(self.dim, dtype=np.float32)

  def _compute_step(self, globalg):
    a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
    self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
    self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
    step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
    return step

# PEPG implementation from estool
class PEPG:
  '''Extension of PEPG with bells and whistles.'''
  def __init__(self, num_params,             # number of model parameters
               sigma_init=0.10,              # initial standard deviation
               sigma_alpha=0.20,             # learning rate for standard deviation
               sigma_decay=0.999,            # anneal standard deviation
               sigma_limit=0.01,             # stop annealing if less than this
               sigma_max_change=0.2,         # clips adaptive sigma to 20%
               learning_rate=0.01,           # learning rate for standard deviation
               learning_rate_decay = 0.9999, # annealing the learning rate
               learning_rate_limit = 0.01,   # stop annealing learning rate
               elite_ratio = 0,              # if > 0, then ignore learning_rate
               popsize=256,                  # population size
               average_baseline=True,        # set baseline to average of batch
               weight_decay=0.01,            # weight decay coefficient
               rank_fitness=True,            # use rank rather than fitness numbers
               forget_best=True):            # don't keep the historical best solution

    self.num_params = num_params
    self.sigma_init = sigma_init
    self.sigma_alpha = sigma_alpha
    self.sigma_decay = sigma_decay
    self.sigma_limit = sigma_limit
    self.sigma_max_change = sigma_max_change
    self.learning_rate = learning_rate
    self.learning_rate_decay = learning_rate_decay
    self.learning_rate_limit = learning_rate_limit
    self.popsize = popsize
    self.average_baseline = average_baseline
    if self.average_baseline:
      assert (self.popsize % 2 == 0), "Population size must be even"
      self.batch_size = int(self.popsize / 2)
    else:
      assert (self.popsize & 1), "Population size must be odd"
      self.batch_size = int((self.popsize - 1) / 2)

    # option to use greedy es method to select next mu, rather than using drift param
    self.elite_ratio = elite_ratio
    self.elite_popsize = int(self.popsize * self.elite_ratio)
    self.use_elite = False
    if self.elite_popsize > 0:
      self.use_elite = True

    self.forget_best = forget_best
    self.batch_reward = np.zeros(self.batch_size * 2)
    self.mu = np.zeros(self.num_params)
    self.sigma = np.ones(self.num_params) * self.sigma_init
    self.curr_best_mu = np.zeros(self.num_params)
    self.best_mu = np.zeros(self.num_params)
    self.best_reward = 0
    self.first_interation = True
    self.weight_decay = weight_decay
    self.rank_fitness = rank_fitness
    if self.rank_fitness:
      self.forget_best = True # always forget the best one if we rank
    # choose optimizer
    self.optimizer = Adam(self, learning_rate)

    self.gen_size = self.popsize

  def rms_stdev(self):
    sigma = self.sigma
    return np.mean(np.sqrt(sigma*sigma))

  def ask(self):
    '''returns a list of parameters'''
    # antithetic sampling
    self.epsilon = np.random.randn(self.batch_size, self.num_params) * self.sigma.reshape(1, self.num_params)
    self.epsilon_full = np.concatenate([self.epsilon, - self.epsilon])
    if self.average_baseline:
      epsilon = self.epsilon_full
    else:
      # first population is mu, then positive epsilon, then negative epsilon
      epsilon = np.concatenate([np.zeros((1, self.num_params)), self.epsilon_full])
    solutions = self.mu.reshape(1, self.num_params) + epsilon
    self.solutions = solutions
    return solutions

  def tell(self, reward_table_result):
    # input must be a numpy float array
    assert(len(reward_table_result) == self.popsize), "Inconsistent reward_table size reported."

    reward_table = np.array(reward_table_result)
    
    if self.rank_fitness:
      reward_table = compute_centered_ranks(reward_table)
    
    if self.weight_decay > 0:
      l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
      reward_table += l2_decay

    reward_offset = 1
    if self.average_baseline:
      b = np.mean(reward_table)
      reward_offset = 0
    else:
      b = reward_table[0] # baseline
      
    reward = reward_table[reward_offset:]
    if self.use_elite:
      idx = np.argsort(reward)[::-1][0:self.elite_popsize]
    else:
      idx = np.argsort(reward)[::-1]

    best_reward = reward[idx[0]]
    if (best_reward > b or self.average_baseline):
      best_mu = self.mu + self.epsilon_full[idx[0]]
      best_reward = reward[idx[0]]
    else:
      best_mu = self.mu
      best_reward = b

    self.curr_best_reward = best_reward
    self.curr_best_mu = best_mu

    if self.first_interation:
      self.sigma = np.ones(self.num_params) * self.sigma_init
      self.first_interation = False
      self.best_reward = self.curr_best_reward
      self.best_mu = best_mu
    else:
      if self.forget_best or (self.curr_best_reward > self.best_reward):
        self.best_mu = best_mu
        self.best_reward = self.curr_best_reward

    # short hand
    epsilon = self.epsilon
    sigma = self.sigma

    # update the mean

    # move mean to the average of the best idx means
    if self.use_elite:
      self.mu += self.epsilon_full[idx].mean(axis=0)
    else:
      rT = (reward[:self.batch_size] - reward[self.batch_size:])
      change_mu = np.dot(rT, epsilon)
      self.optimizer.stepsize = self.learning_rate
      update_ratio = self.optimizer.update(-change_mu) # adam, rmsprop, momentum, etc.
      #self.mu += (change_mu * self.learning_rate) # normal SGD method

    # adaptive sigma
    # normalization
    if (self.sigma_alpha > 0):
      stdev_reward = 1.0
      if not self.rank_fitness:
        stdev_reward = reward.std()
      S = ((epsilon * epsilon - (sigma * sigma).reshape(1, self.num_params)) / sigma.reshape(1, self.num_params))
      reward_avg = (reward[:self.batch_size] + reward[self.batch_size:]) / 2.0
      rS = reward_avg - b
      delta_sigma = (np.dot(rS, S)) / (2 * self.batch_size * stdev_reward)

      # adjust sigma according to the adaptive sigma calculation
      # for stability, don't let sigma move more than 10% of orig value
      change_sigma = self.sigma_alpha * delta_sigma
      change_sigma = np.minimum(change_sigma, self.sigma_max_change * self.sigma)
      change_sigma = np.maximum(change_sigma, - self.sigma_max_change * self.sigma)
      self.sigma += change_sigma

    if (self.sigma_decay < 1):
      self.sigma[self.sigma > self.sigma_limit] *= self.sigma_decay
    
    if (self.learning_rate_decay < 1 and self.learning_rate > self.learning_rate_limit):
      self.learning_rate *= self.learning_rate_decay

  def current_param(self):
    return self.curr_best_mu

  def set_mu(self, mu):
    self.mu = np.array(mu)
  
  def best_param(self):
    return self.best_mu

  def result(self): # return best params so far, along with historically best reward, curr reward, sigma
    return self.best_mu




