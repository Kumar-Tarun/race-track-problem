# Race Track Problem
* This repository contains code for the Race track problem solved via Monte-Carlo off-policy algorithm and Q-learning algorithm.

## Results
* Q-Learning converges quickly than Monte-Carlo for all state-action pairs.
* Following are the sample policies found by both the algorithms.
### Monte-Carlo
![](/images/mc.gif) 
### Q-Learning
![](/images/ql.gif)

## Dependencies
* numpy
* tqdm
* pygame

## Usage
* Run the ```python3 race_track.py option```, where ```option``` is **ql** for Q-learning or **mc** for Monte-Carlo.
* After training is finished, automatically the pygame window opens to display the optimal policy found.
* If the window doesn't open, that means for some of the starting states, the algorithm has still not converged.
* For both Monte-Carlo and Q-learning, the running-time is under 5 minutes.
