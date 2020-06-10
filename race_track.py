import numpy as np
import pygame
import pickle
import time
import sys
from tqdm import tqdm

rows, cols = 32, 17

# visualize the race track
class Visualizer:
    
    def __init__(self, racetrack):

      self.data = racetrack
      self.cell_edge = 15
      self.width = 100*self.cell_edge
      self.height = 100*self.cell_edge
      self.display = pygame.display.set_mode((self.width, self.height))
      pygame.display.set_caption("Racetrack")

      self.window = True

    def draw(self, state=[]):
        self.display.fill(0)
        for i in range(rows):
            for j in range(cols):
                if self.data[i,j] == 0:
                    color = (255,0,0)
                elif self.data[i,j] == 3:
                    color = (255,255,0)
                elif self.data[i,j] == 2:
                    color = (0,255,0)
                else:
                  color = (0,0,0)
                pygame.draw.rect(self.display,color,((j*self.cell_edge,i*self.cell_edge),(self.cell_edge,self.cell_edge)),1)
        
        if len(state)>0:
            pygame.draw.rect(self.display,(0,0,255),((state[1]*self.cell_edge,state[0]*self.cell_edge),(self.cell_edge,self.cell_edge)),1)
        
        pygame.display.update()

gamma = 1

n_actions = 9
n_velocity = 5
# set this flag to True to load previously saved training information
load_prev = False

# origin is at 0,0 top left
# race track is a matrix of 32X17, with 0 as the allowed area and 1 as the off-track area 
race_track = np.zeros((rows, cols))
# marks the borders of right of race track
race_track[6:, 9:] = 1
race_track[6, 9] = 0
# marks the borders of left of race track
race_track[29:, 2] = 1
race_track[22:, 1] = 1
race_track[14:, 0] = 1

race_track[:4, 0] = 1
race_track[:3, 1] = 1
race_track[:1, 2] = 1

# mark the finish cells with 2
race_track[:6, 16] = 2
# mark the start cells with 3
race_track[31, 3:9] = 3

# define start line columns
start_line_cols = np.arange(3, 9)
# define finish cells as a set of (row, col) tuples
finish_cells = [(r, cols-1) for r in np.arange(0, 6)]

# state = (row, col, vel_x, vel_y)
# initialize Q to be large (negative) random state-action values. This is an optimistic initialization
Q = np.random.rand(rows, cols, n_velocity, n_velocity, n_actions) * 400 - 500
# C denotes the sum of weights
C = np.zeros((rows, cols, n_velocity, n_velocity, n_actions))
# define target policy as greedy wrt Q
pi = np.argmax(Q, axis=-1)

if load_prev:
  with open('Q.pkl', 'rb') as f:
    Q = pickle.load(f)

  with open('C.pkl', 'rb') as f:
    C = pickle.load(f)

  with open('pi.pkl', 'rb') as f:
    pi = pickle.load(f)

assert pi.shape == (rows, cols, n_velocity, n_velocity)

# action list
actions = [(0,0),
       (0, 1),
       (0, -1),
       (1, 0),
       (1, 1),
       (1, -1),
       (-1, 0),
       (-1, 1),
       (-1, -1)
      ]

# define valid actions for each pair (vel_x, vel_y)
valid_actions = {}
for vel_x in range(n_velocity):
  for vel_y in range(n_velocity):
    valid_actions[(vel_x, vel_y)] = []
    for i, act in enumerate(actions):
      if act[0] == -1 and vel_x == 1 and vel_y <= 1:
        continue
      elif act[1] == -1 and vel_y == 1 and vel_x <=1:
        continue
      if vel_x + act[0] >=0  and vel_x + act[0] <=4 and vel_y + act[1] >= 0 and vel_y + act[1] <=4:
        valid_actions[(vel_x, vel_y)].append(i)


def sample_episode(epsilon = 0.1, noise=False):
  '''
  :param epsilon: epsilon of greedy policy
  :param noise: whether to add noise to control process
  :returns:
  Episode information - S (sequence of states)
            - A (sequence of actions)
            - B (sequence of importance sampling ratios)
            - R (total reward from this episode)
  '''
  trajectory = []
  S = []
  A = []
  B = []
  R = 0
  s = (31, np.random.choice(start_line_cols), 0, 0)
  while True:
    S.append(s)
    valid_acts = valid_actions[(s[2], s[3])]
    n_valid_acts = len(valid_acts)
    action = pi[s[0], s[1], s[2], s[3]]
    is_valid = action in valid_acts
    if np.random.random() >= epsilon:
      if is_valid:
        a = action
        b = 1 - epsilon
      else:
        a = np.random.choice(valid_acts)
        b = 1 / n_valid_acts
    else:
      a = np.random.choice(valid_acts)
      if is_valid:
        b = epsilon / n_valid_acts
      else:
        b = 1 / n_valid_acts

    if noise and np.random.random() < 0.1:
      a = 0
      b = 0.1

    A.append(a)
    B.append(b)
    R += (-1)
    act = actions[a]
 
    next_velocity = (s[2]+act[0], s[3]+act[1])
    next_s = (s[0] - s[3], s[1] + s[2], next_velocity[0], next_velocity[1])

    # if crosses finish line, complete the episode
    if (next_s[0] <= finish_cells[-1][0] and next_s[1] >= (cols-1)) or (next_s[0] in range(finish_cells[-1][0], finish_cells[-1][0] + 5) and next_s[1] >= cols):
      assert len(S) == len(A)
      assert len(A) == len(B)
      return S, A, B, R

    # if hits boundary, return to start state, else go to next state
    if race_track[next_s[0], next_s[1]] == 1 or next_s[0] >= rows or next_s[1] < 0 or next_s[0] < 0:
      s = (31, np.random.choice(start_line_cols), 0, 0)
    else:
      s = next_s

def mc_off_policy_control(S, A, B):
  '''
  :param S: sequence of states
  :param A: sequence of actions
  :param B: sequence of importance-sampling ratios
  '''
  G = 0
  W = 1
  R = -1
  for s, a, b in zip(reversed(S), reversed(A), reversed(B)):
    G = gamma * G + R
    s1, s2, s3, s4 = s[0], s[1], s[2], s[3]
    C[s1, s2, s3, s4, a] += W
    Q[s1, s2, s3, s4, a] += W * (G - Q[s1, s2, s3, s4, a]) / C[s1, s2, s3, s4, a]
    pi[s1, s2, s3, s4] = np.argmax(Q[s1, s2, s3, s4], axis=-1)
    if a != pi[s1, s2, s3, s4]:
      break
    W /= b

def q_learning_control(alpha=0.1, epsilon=0.1):
  '''
  :param alpha: learning rate/step size
  :param epsilon: epsilon of greedy policy
  :returns:
  Episode information - S (sequence of states)
            - A (sequence of actions)
            - B (sequence of importance sampling ratios)
            - R (total reward from this episode)
  '''
  trajectory = []
  S = []
  A = []
  B = []
  R = 0
  s = (31, np.random.choice(start_line_cols), 0, 0)
  while True:
    S.append(s)
    s1, s2, s3, s4 = s[0], s[1], s[2], s[3]
    valid_acts = valid_actions[(s3, s4)]
    n_valid_acts = len(valid_acts)
    # action = pi[s[0], s[1], s[2], s[3]]
    action = np.argmax(Q[s1, s2, s3, s4], axis=-1)
    is_valid = action in valid_acts
    if np.random.random() >= epsilon:
      if is_valid:
        a = action
        b = 1 - epsilon
      else:
        a = np.random.choice(valid_acts)
        b = 1 / n_valid_acts
    else:
      a = np.random.choice(valid_acts)
      if is_valid:
        b = epsilon / n_valid_acts
      else:
        b = 1 / n_valid_acts

    A.append(a)
    B.append(b)
    R += (-1)
    act = actions[a]

    next_velocity = (s[2]+act[0], s[3]+act[1])
    next_s = (s[0] - s[3], s[1] + s[2], next_velocity[0], next_velocity[1])
    # if crosses finish line, complete the episode
    if (next_s[0] <= finish_cells[-1][0] and next_s[1] >= (cols-1)) or (next_s[0] in range(finish_cells[-1][0], finish_cells[-1][0] + 5) and next_s[1] >= cols):
      assert len(S) == len(A)
      assert len(A) == len(B)
      # Q(terminal, .) = 0
      Q[s1, s2, s3, s4, a] = Q[s1, s2, s3, s4, a] + alpha * (-1 + gamma * 0 - Q[s1, s2, s3, s4, a]) 
      return S, A, B, R

    # if hits boundary, return to start state, else go to next state
    if race_track[next_s[0], next_s[1]] == 1 or (next_s[0] >= rows) or next_s[1] < 0 or next_s[0] < 0:
      next_s = (31, np.random.choice(start_line_cols), 0, 0)

    # Q-Learning update
    Q[s1, s2, s3, s4, a] = Q[s1, s2, s3, s4, a] + alpha * (-1 + gamma * np.max(Q[next_s[0], next_s[1], next_s[2], next_s[3]], axis=-1) - Q[s1, s2, s3, s4, a])
    s = next_s

def plot(R):
  '''
  plot the graph of reward vs. no. of episodes
  '''
  plt.plot(R)
  plt.xlabel('No. of episodes')
  plt.ylabel('Reward')
  plt.xticks(ticks=np.arange(0, 120000, 20000), labels=[str(i/1000) + 'k' for i in np.arange(0, 120000, 20000)])
  plt.show()

def main():
  '''
  Run large no. of episodes to optimize the state-action value function
  '''
  rewards = []
  if len(sys.argv) < 2:
    print('Expected one of the following: mc or ql.')
    return
  elif sys.argv[1] == 'mc':
    # run monte-carlo off policy algo
    # monte-carlo requires more episodes to converge for all (s,a)
    n_episodes = 200000
    for i in tqdm(range(n_episodes)): 
      S, A, B, R = sample_episode()
      rewards.append(R)
      mc_off_policy_control(S, A, B)
  elif sys.argv[1] == 'ql':
    # run q-learning off policy algo
    n_episodes = 100000
    for i in tqdm(range(n_episodes)): 
      S, A, B, R = q_learning_control()
      rewards.append(R)
  else:
    print('Invalid combination of arguments.')
    return

  plot(rewards)
  with open('Q.pkl', 'wb+') as f:
    pickle.dump(Q, f)

  with open('C.pkl', 'wb+') as f:
    pickle.dump(C, f)

  with open('pi.pkl', 'wb+') as f:
    pickle.dump(pi, f)

  # now sample an episode using optimal policy
  if sys.argv[1] == 'mc':
    S, A, B, _ = sample_episode(epsilon=0.0)
  else:
    S, A, B, _ = q_learning_control(epsilon=0.0)

  print('Episode length: ', len(S))
  vis = Visualizer(race_track)
  for i, s in enumerate(S):
    # interrupt the script or close pygame window to stop the visualization
    vis.draw((s[0],s[1]))
    time.sleep(0.2)

if __name__ == '__main__':
  main()
