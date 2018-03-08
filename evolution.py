# Author: Andrew Malta 2018
import numpy as np

class evolve():
    def __init__(self, theta_dim, num_generations, gen_size, evaluate, num_trials):
        self.theta_dim = theta_dim
        self.num_generations = num_generations
        self.gen_size = gen_size
        self.evaluate = evaluate
        self.seed_gen = xrange(gen_size)
        self.num_trials = num_trials

    def weighted_average_es(self, n, alpha=.2, sigma=.05):
        theta = np.zeros(self.theta_dim)
        for t in xrange(self.num_generations):
            fitnesses = []
            epsilons = []
            for s in self.seed_gen:
                # np.random.seed(s)
                epsilon = np.random.normal(0, 1, self.theta_dim)

                new_theta = theta + sigma * epsilon
                fitness = self.evaluate(new_theta, self.num_trials, t == self.num_generations - 1)
                fitnesses.append(fitness)
                epsilons.append(epsilon)

            argsort = list(np.argsort([-val for val in fitnesses]))
            fitness_update_sum = np.zeros(self.theta_dim)
            for i in argsort[0:n]:
                fitness_update_sum += (epsilons[i] * fitnesses[i])

            theta = theta + (alpha / n * sigma) * fitness_update_sum
            print "end of generation {} with a best fitness of {}".format(t + 1, fitnesses[argsort[0]])

        return theta

    def winner_es(self, alpha=.25, sigma=.02):
        theta = np.zeros(self.theta_dim)
        for t in xrange(self.num_generations):
            best_fitness = -float("inf")
            best_weights = None
            for s in self.seed_gen:
                # np.random.seed(s)
                epsilon = np.random.normal(0, 1, self.theta_dim)
                new_theta = theta + sigma * epsilon
                fitness = self.evaluate(new_theta,
                                        self.num_trials,
                                        t == self.num_generations - 1)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_weights = theta + sigma * epsilon

            print "end of generation {} with a best fitness of {}".format(t + 1, best_fitness)
            theta = best_weights

        return theta


    def cma_es(self, mu=5, start_sigma=.02):
        n = self.theta_dim
        lamb = self.gen_size

        mean = np.zeros(n)
        sigma = start_sigma
        C = np.identity(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        for t in xrange(self.num_generations):
            x = []
            fitnesses = []
            for s in self.seed_gen:
                # np.random.seed(s)
                theta = np.random.multivariate_normal(mean, sigma * sigma * C)
                x.append(theta)
                # negate the objective function to convert to allow for minimization
                fitnesses.append(-self.evaluate(theta, self.num_trials))

            tmp = mean
            sorted_indices = np.argsort(fitnesses)

            # update m
            mean = 0
            # using equal weighting 
            for i in sorted_indices[0:mu]:
                mean += x[i] / mu

            # update p_sigma
            c_sigma = 3. / n
            df = 1 - c_sigma # discount factor
            C_tmp = np.sqrt(np.linalg.inv(C))
            displacement = (mean - tmp) / float(sigma)

            p_sigma = (df * p_sigma + 
                np.sqrt(1 - np.power(df,2)) * np.sqrt(mu) * C_tmp * displacement)

            # update p_c
            c_c = 4. / n
            df2 = (1 - c_c)
            alpha = 1.5

            norm = np.linalg.norm(p_sigma)
            indicator = norm >= 0 and norm <= alpha * np.sqrt(n)
            complements = np.sqrt(1 - np.power((1 - c_c), 2))
            neutral_selection = np.sqrt(mu) * displacement
            p_c = df2 * p_c + indicator * (complements * neutral_selection)

            # update Covariance Matrix
            c_1 = 2. / np.power(n, 2)
            c_mu = mu / np.power(n, 2)
            c_s = (1 - indicator)* c_1 * c_c * (2 - c_c)

            tmp_pc = p_c.reshape(-1, 1)
            rank_1_mat = c_1 * tmp_pc.dot(tmp_pc.transpose())

            mat_sum = np.zeros(C.shape)
            for i in xrange(mu):
                vec = ((x[sorted_indices[i]] - tmp) / sigma).reshape(-1, 1)
                mat_sum += (1. / mu) * vec.dot(vec.transpose())

            C = (1 - c_1 - c_mu + c_s) * C + rank_1_mat + c_mu * mat_sum 

            # update sigma
            d_sigma = 1  # dampening parameter
            p_sigma_norm = np.linalg.norm(p_sigma)
            tmp = (1 - 1. / (4 * n) + 1. / (21 * n * n))
            nse = np.sqrt(n) * tmp  # neutral selection expectation
            simga = sigma * np.exp((c_sigma / d_sigma) * ((p_sigma_norm / nse) - 1))

            best_fitness = fitnesses[sorted_indices[0]]
            best_weights = x[sorted_indices[0]]

            print "end of generation {} with a best fitness of {}".format(t + 1, -best_fitness)

        return best_weights







