import numpy as np


# encoding of the higher stages

S_A = 0
S_B = 1

# just as info
#possible_states  = [0,1]
#possible_actions = [0,1]    # [left lever, right lever pull]

hstage = {0: "Stage A", 1: "Stage B"}



# ### Definition of our environment: the two-step task
class rsc_two_step():
  def __init__(self, high_reward_prob = 0.9):

    self.r_prob = high_reward_prob
    self.verbose = False;
    self.reset();

    common_prob = 0.8
    self.xtransitions = np.array([ [common_prob, 1-common_prob],[1-common_prob, common_prob]])

    # indices as follows: [reward_received? yes->1, after_common_transition->1, action same as in last trial->1 ]
    self.stats = np.zeros((2,2,2))

    # both S_A and S_B can be rewarded
    # state with high probability yields reward with r_prob probability
    # the other only with (1-r_prob)
    self.state_w_higher_reward_probability = np.random.choice([S_A,S_B]);
    #self.state_w_higher_reward_probability = init_reward_stage

  
  def reset(self):
    self.numtrials_played = 0;
    # scalar value in the range of [0 1 2]
    self.last_action = None
    self.numswitches = 0;


  def change_high_reward_probability(self, high_reward_prob):
    self.high_reward_prob = high_reward_prob;


  def conduct_action(self, action):
    # actions should be in the range [0,1] 
    if self.verbose: print("Action: ", action)

    # update stats about (non-)repetition of actions
    if self.last_action != None:
        self.stats[self.last_reward,1 if self.last_was_common else 0,1 if self.last_action == action else 0] += 1;
        if self.verbose: print("Same action as in last trial: ", self.last_action == action)


    # with some rather low probability (2.5%) swith the high reward probability to the other lever (S_A or S_B)
    #if (np.random.uniform() < 0.025):
    if (np.random.uniform() < 0.025):
      self.state_w_higher_reward_probability = S_A if (self.state_w_higher_reward_probability == S_B) else S_B
      if self.verbose: 
        print("--------------------------------------------")
        print("Switched highly rewarded second stage to: ", hstage[self.state_w_higher_reward_probability]);
        self.numswitches = self.numswitches +1;

    ## First Step
    # is common means that the action leads to the desired second state

    second_level = S_A if (np.random.uniform() < self.xtransitions[action][0]) else S_B
    is_common = self.xtransitions[action][second_level] >= 0.5

    if self.verbose: print(("Common" if is_common else " Uncommon"), " transition to ", hstage[second_level])

    ## Second Step
    # find reward probability at current state:    
    trial_r_prop = self.r_prob if second_level == self.state_w_higher_reward_probability else (1.0-self.r_prob)
    if self.verbose: print("This stage has a reward probability of ", trial_r_prop)

    reward = 1 if np.random.uniform() < trial_r_prop else 0
    if self.verbose: print("Reward received: ", reward)

    # book-keeping for stats
    self.numtrials_played += 1
    self.last_action = action;
    self.last_was_common = is_common;
    self.last_reward = reward;

    # return which second level was reached, and to which reward it led
    return [second_level, reward];



  def print_stats(self):
    for x in range(len(self.stats)):
      for y in range(len(self.stats[x])):
        for z in range(len(self.stats[x][y])):
          print("Reward received: ", x, ", After common transition?: ", y, " Action repeated afterwards: ", z, "\t - ", self.stats[x,y,z])

  
  def play_interactively(self, verbose = False):
    cached_verb = self.verbose;
    self.verbose = verbose;
    print("Interactive game started, input 'x' to end game. Actions: 0 or 1")
    self.reset()
    x = None;
    while True:
      x = input(str(self.numtrials_played) + "> ");
      if x == 'x': break
      [sl, reward] = self.conduct_action(int(x));
      print("Observed/reached second state: ", hstage[sl], "\nReward: ", reward)
    print("Interactive session aborted, stats:")
    self.print_stats();
    self.verbose = cached_verb;
