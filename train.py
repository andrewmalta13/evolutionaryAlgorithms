from mpi4py import MPI
import numpy as np
import json
import os
import subprocess
import sys
from simulate import simulate
from models import Net
from evolution import Weighted_Average_ES, Winner_ES, CMA_ES, OpenES
import argparse
import time
import gym


### ES related code
num_episode = 1
eval_steps = 25 # evaluate every N_eval steps

num_worker = 8
num_worker_trial = 16

population = num_worker * num_worker_trial

gamename = 'BipedalWalkerHardcore-v2'
cap_time_mode = True
antithetic = False

# parameter to determine how to calculate the meta-fitness score:
# run the simulate function a number of times and calculate a statistic
# that measures the robustness
batch_mode = 'mean'

# seed for reproducibility
seed_start = 0

# name of the file (can override):
filebase = None

model = None
num_params = -1

es = None

### MPI related code
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

PRECISION = 10000
SOLUTION_PACKET_SIZE = (5+num_params)*num_worker_trial
RESULT_PACKET_SIZE = 4*num_worker_trial
###

def initialize_settings():
  global population, filebase, model, num_params, es, PRECISION, SOLUTION_PACKET_SIZE, RESULT_PACKET_SIZE
  population = num_worker * num_worker_trial
  filebase = 'log/'+gamename+'.'+optimizer+'.'+str(num_episode)+'.'+str(population)
  model = Net(None) #
  num_params = model.num_flat_features()

  print("size of model", num_params)

  gen_size = num_worker * num_worker_trial

  if optimizer == 'weighted_average_es':
    args = {"theta_dim": model.num_flat_features(),
            "gen_size": gen_size,
            "sigma": .05,
            "alpha": .2,
            "n": 15}

    es = Weighted_Average_ES(args)
  elif optimizer == 'winner_es':
    args = {"theta_dim": model.num_flat_features(),
            "gen_size": gen_size,
            "sigma": .05,
            "alpha": .2,
            "n": 15}

    es = Winner_ES(args)
  elif optimizer == 'cma':
    args = {"theta_dim": model.num_flat_features(),
            "gen_size": gen_size,
            "start_sigma": .05,
            "mu": 5
           }

    es = CMA_ES(args)
  elif optimizer == "openes":
    args = {"theta_dim": model.num_flat_features(),
            "sigma_start": .1,
            "sigma_mult": .999,
            "sigma_lower_bound": .02,
            "alpha": .01,
            "alpha_mult": 1,
            "alpha_lower_bound": .01,
            "weight_decay": .005,
            "start_sigma": .05,
            "rank_fitness": True,            # use rank rather than fitness numbers
            "forget_best": True,             # should we forget the best fitness at each iteration
            "gen_size": gen_size
           }

    es = OpenES(args)

  else:
    print("invalid evolutionary algorithm")

  PRECISION = 10000
  SOLUTION_PACKET_SIZE = (5+num_params)*num_worker_trial
  RESULT_PACKET_SIZE = 4*num_worker_trial
###

def sprint(*args):
  print(args) # if python3, can do print(*args)
  sys.stdout.flush()

class Seeder:
  def __init__(self, init_seed=0):
    np.random.seed(init_seed)
    self.limit = np.int32(2**31-1)
  def next_seed(self):
    result = np.random.randint(self.limit)
    return result
  def next_batch(self, batch_size):
    result = np.random.randint(self.limit, size=batch_size).tolist()
    return result

def encode_solution_packets(seeds, solutions, train_mode=1, max_len=-1):
  n = len(seeds)
  result = []
  worker_num = 0
  for i in range(n):
    worker_num = int(i / float(num_worker_trial)) + 1
    result.append([worker_num, i, seeds[i], train_mode, max_len])
    result.append(np.round(np.array(solutions[i])*PRECISION,0))
  result = np.concatenate(result).astype(np.int32)
  result = np.split(result, num_worker)
  return result

def decode_solution_packet(packet):
  packets = np.split(packet, num_worker_trial)
  result = []
  for p in packets:
    result.append([p[0], p[1], p[2], p[3], p[4], p[5:].astype(np.float)/PRECISION])
  return result

def encode_result_packet(results):
  r = np.array(results)
  r[:, 2:4] *= PRECISION
  return r.flatten().astype(np.int32)

def decode_result_packet(packet):
  r = packet.reshape(num_worker_trial, 4)
  workers = r[:, 0].tolist()
  jobs = r[:, 1].tolist()
  fits = r[:, 2].astype(np.float)/PRECISION
  fits = fits.tolist()
  times = r[:, 3].astype(np.float)/PRECISION
  times = times.tolist()
  result = []
  n = len(jobs)
  for i in range(n):
    result.append([workers[i], jobs[i], fits[i], times[i]])
  return result

def worker(env, weights, seed, train_mode_int=1, max_len=-1):
  train_mode = (train_mode_int == 1)
  model.set_model_params(weights)
  reward_list, t_list = simulate(env, Net, weights, num_worker_trial, seed)
  if batch_mode == 'min':
    reward = np.min(reward_list)
  else:
    reward = np.mean(reward_list)
  t = np.mean(t_list)
  return reward, t

def slave():
  env = gym.make(gamename)
  packet = np.empty(SOLUTION_PACKET_SIZE, dtype=np.int32)
  while 1:
    comm.Recv(packet, source=0)
    assert(len(packet) == SOLUTION_PACKET_SIZE)
    solutions = decode_solution_packet(packet)
    results = []
    for solution in solutions:
      worker_id, jobidx, seed, train_mode, max_len, weights = solution
      assert (train_mode == 1 or train_mode == 0), str(train_mode)
      worker_id = int(worker_id)
      possible_error = "work_id = " + str(worker_id) + " rank = " + str(rank)
      assert worker_id == rank, possible_error
      jobidx = int(jobidx)
      seed = int(seed)
      fitness, timesteps = worker(env, weights, seed, train_mode, max_len)
      results.append([worker_id, jobidx, fitness, timesteps])
    result_packet = encode_result_packet(results)
    assert len(result_packet) == RESULT_PACKET_SIZE
    comm.Send(result_packet, dest=0)

def send_packets_to_slaves(packet_list):
  num_worker = comm.Get_size()
  assert len(packet_list) == num_worker-1
  for i in range(1, num_worker):
    packet = packet_list[i-1]
    assert(len(packet) == SOLUTION_PACKET_SIZE)
    comm.Send(packet, dest=i)

def receive_packets_from_slaves():
  result_packet = np.empty(RESULT_PACKET_SIZE, dtype=np.int32)

  reward_list_total = np.zeros((population, 2))

  check_results = np.ones(population, dtype=np.int)
  for i in range(1, num_worker+1):
    comm.Recv(result_packet, source=i)
    results = decode_result_packet(result_packet)
    for result in results:
      worker_id = int(result[0])
      possible_error = "work_id = " + str(worker_id) + " source = " + str(i)
      assert worker_id == i, possible_error
      idx = int(result[1])
      reward_list_total[idx, 0] = result[2]
      reward_list_total[idx, 1] = result[3]
      check_results[idx] = 0

  check_sum = check_results.sum()

  return reward_list_total

def evaluate_batch(model_params, max_len=-1):
  # duplicate model_params
  solutions = []
  for i in range(es.gen_size):
    solutions.append(np.copy(model_params))

  seeds = np.arange(es.gen_size)

  packet_list = encode_solution_packets(seeds, solutions, train_mode=0, max_len=max_len)

  send_packets_to_slaves(packet_list)
  reward_list_total = receive_packets_from_slaves()

  reward_list = reward_list_total[:, 0] # get rewards
  return np.mean(reward_list)

def master(starting_file):
  start_time = int(time.time())
  sprint("training", gamename)
  sprint("population", es.gen_size)
  sprint("num_worker", num_worker)
  sprint("num_worker_trial", num_worker_trial)
  sys.stdout.flush()

  seeder = Seeder(seed_start)

  filename = filebase+'.json'
  filename_log = filebase+'.log.json'
  filename_hist = filebase+'.hist.json'
  filename_best = filebase+'.best.json'

  # if the user specified a place to start from
  if starting_file:
    print starting_file
    with open(starting_file, "r") as f:
      sprint("Loading File to Start From:", starting_file)
      data = json.load(f)
      weights = data[0]

    es.set_theta(weights)

  t = 0

  history = []
  eval_log = []
  best_reward_eval = 0
  best_model_params_eval = None

  max_len = -1 # max time steps (-1 means ignore)
  restart_counter = 0

  while True:
    t += 1

    # go back to your best solution if after 15 generations you
    # don't see an improvement in your fitness. 
    if restart_counter == 15:
      sprint("---resest theta to previous best---")
      restart_counter = 0
      es.set_theta(best_model_params_eval)

    solutions = es.ask()

    if antithetic:
      seeds = seeder.next_batch(int(es.gen_size/2))
      seeds = seeds+seeds
    else:
      seeds = seeder.next_batch(es.gen_size)

    packet_list = encode_solution_packets(seeds, solutions, max_len=max_len)

    send_packets_to_slaves(packet_list)
    reward_list_total = receive_packets_from_slaves()

    reward_list = reward_list_total[:, 0] # get rewards

    mean_time_step = int(np.mean(reward_list_total[:, 1])*100)/100. # get average time step
    max_time_step = int(np.max(reward_list_total[:, 1])*100)/100. # get average time step
    avg_reward = int(np.mean(reward_list)*100)/100. # get average time step
    std_reward = int(np.std(reward_list)*100)/100. # get average time step

    es.tell(reward_list)

    es_solution = es.result()
    model_params = es_solution # best historical solution

    model.set_model_params(np.array(model_params).round(4))

    r_max = int(np.max(reward_list)*100)/100.
    r_min = int(np.min(reward_list)*100)/100.

    curr_time = int(time.time()) - start_time

    h = (t, curr_time, avg_reward, r_min, r_max, std_reward, mean_time_step+1., int(max_time_step)+1)

    if cap_time_mode:
      max_len = 2*int(mean_time_step+1.0)
    else:
      max_len = -1

    history.append(h)

    with open(filename, 'wt') as out:
      res = json.dump([np.array(es.result()).round(4).tolist()], out, sort_keys=True, indent=2, separators=(',', ': '))

    with open(filename_hist, 'wt') as out:
      res = json.dump(history, out, sort_keys=False, indent=0, separators=(',', ':'))

    sprint(gamename, h)

    if (t == 1):
      best_reward_eval = avg_reward
    if (t % eval_steps == 0): # evaluate on actual task at hand

      prev_best_reward_eval = best_reward_eval
      model_params_quantized = np.array(es.result()).round(4)
      reward_eval = evaluate_batch(model_params_quantized, max_len=-1)
      model_params_quantized = model_params_quantized.tolist()
      improvement = reward_eval - best_reward_eval
      eval_log.append([t, reward_eval, model_params_quantized])

      with open(filename_log, 'wt') as out:
        res = json.dump(eval_log, out)

      if (len(eval_log) == 1 or reward_eval > best_reward_eval):
        best_reward_eval = reward_eval
        best_model_params_eval = model_params_quantized
      else:
        restart_counter += 1
        sprint("---restart counter: ", restart_counter, "---")
  
      with open(filename_best, 'wt') as out:
        res = json.dump([best_model_params_eval, best_reward_eval], out, sort_keys=True, indent=0, separators=(',', ': '))
      sprint("improvement", t, improvement, "curr", reward_eval, "prev", prev_best_reward_eval, "best", best_reward_eval)


def main(args):
  global optimizer, num_episode, eval_steps, num_worker, num_worker_trial, seed_start, retrain_mode, cap_time_mode
  optimizer = args.optimizer
  num_episode = args.num_episode
  eval_steps = args.eval_steps
  num_worker = args.num_worker
  num_worker_trial = args.num_worker_trial
  seed_start = args.seed_start
  start_file = args.start_file

  initialize_settings()

  sprint("process", rank, "out of total ", comm.Get_size(), "started")
  if (rank == 0):
    master(start_file)
  else:
    slave()

def mpi_fork(n):
  """Re-launches the current script with workers
  Returns "parent" for original parent, "child" for MPI children
  (from https://github.com/garymcintire/mpi_util/)
  """
  if n<=1:
    return "child"
  if os.getenv("IN_MPI") is None:
    env = os.environ.copy()
    env.update(
      MKL_NUM_THREADS="1",
      OMP_NUM_THREADS="1",
      IN_MPI="1"
    )
    print( ["mpirun", "-np", str(n), sys.executable] + sys.argv)
    subprocess.check_call(["mpirun", "-np", str(n), sys.executable] +['-u']+ sys.argv, env=env)
    return "parent"
  else:
    global nworkers, rank
    nworkers = comm.Get_size()
    rank = comm.Get_rank()
    print('assigning the rank and nworkers', nworkers, rank)
    return "child"

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=('Train policy for OpenAI gym env: BipedalWalker-v2'
                                                'using openes, winner_es, weighted_average_es, cma-es'))

  parser.add_argument('-o', '--optimizer', type=str, help='ses, pepg, openes, ga, cma.', default='openes')
  parser.add_argument('-e', '--num_episode', type=int, default=1, help='num episodes per trial')
  parser.add_argument('--eval_steps', type=int, default=25, help='evaluate every eval_steps step')
  parser.add_argument('-n', '--num_worker', type=int, default=8)
  parser.add_argument('-t', '--num_worker_trial', type=int, help='trials per worker', default=4)
  parser.add_argument('-s', '--seed_start', type=int, default=111, help='initial seed')
  parser.add_argument('--start_file', type=str, default=None, help='')

  args = parser.parse_args()
  if "parent" == mpi_fork(args.num_worker+1): os.exit()
  main(args)
