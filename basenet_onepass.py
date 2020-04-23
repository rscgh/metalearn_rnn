'''

This is a pytorch implementation of the agent described in [Wang et al., 2018](https://www.biorxiv.org/content/biorxiv/early/2018/04/13/295964.full.pdf). 

This largely mirros and is based on on a tensorflow implementation of the same agent by [Michaël Trazzi and Yasmine Hamdani, under the supervision of Olivier Sigaud](https://github.com/mtrazzi/two-step-task), who themselves made use of [awjuliani's Meta-RL implementation](https://github.com/awjuliani/Meta-RL)(a single-threaded A2C LSTM implementation). For conceptual understanding you can also read [Michaels article on Medium](https://blog.floydhub.com/meta-rl/) and [Arthurs post](https://medium.com/hackernoon/learning-policies-for-learning-policies-meta-reinforcement-learning-rl%C2%B2-in-tensorflow-b15b592a2ddf).

Additionally to previous implementations, I added more logging and put a focus on code readibility/understanding of the implementation.

File from the repository: https://github.com/rscgh/metalearn_rnn

This implementation just uses one pass for each trial through the network instead of two which was done in the original tensorflow implementation (so it combines episode rollout accumulation and gradient tagging of the variables). This does not nessesarily result in better performance / shorter runtime.

'''


## Imports

import os


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.tensor as ts

import numpy as np
from scipy import signal as scysignal


from games.rsc_two_step import rsc_two_step

# logging stuff 

import time
from time import gmtime, strftime

import matplotlib.pyplot as plt
from collections import namedtuple

from torch.utils.tensorboard import SummaryWriter



## Helper functions

discount = lambda x, gamma: scysignal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out



## own network class; basically does everything
# stated by the main function

class OwnRnn(nn.Module):
  def __init__(self, log_dir = "out/rnn", num_actions=2, num_states=2, optimizer_str = None, learning_rate = 7e-4):
    super(OwnRnn, self).__init__();

    # episode buffer, will hold info for later gradient calulation of a single episode
    # usually contains 200 trials, one trial per row
    self.epbuffer = []
    
    # 2 is one value for the reward, and one fot the timestamp
    self.input_size = num_actions + num_states + 2;
    self.num_actions  = num_actions;
    self.num_states  = num_actions;

    self.num_rnn_units = 48;

    # input of the rnn has shape [sequence length, batch, inputs]
    # output of rnn has shape: [sequence length / num trials per episode, batch (number of episodes), num_hidden_unit_actiontions]
    # hence output[1,2,30] represents the value of the activation of the 30th unit in the RNN in response to the first input of the 2nd sample (there is multiple samples in one batch; and each sample presents the network with a number of consecutive inputs; here only 1)

    # we dont use RNN but LSTM now
    #self.reccurrent_layer = torch.nn.RNN(input_size=self.input_size, hidden_size= self.num_rnn_units, num_layers =1, nonlinearity = 'relu')

    # do here new https://github.com/ikostrikov/pytorch-a3c/blob/master/model.py
    # self.lstm = nn.LSTMCell(self.input_size, self.num_rnn_units)  -> is wrong, we dont use cell but the LSTM
    self.lstm = nn.LSTM(self.input_size, self.num_rnn_units)
    
    self.lstm.bias_ih_l0.data.fill_(0)
    self.lstm.bias_hh_l0.data.fill_(0)

    self.action_outp_layer = nn.Linear(in_features= self.num_rnn_units, out_features= num_actions)
    self.value_outp_layer  = nn.Linear(in_features= self.num_rnn_units, out_features= 1)

    self.action_outp_layer.weight.data = normalized_columns_initializer(self.action_outp_layer.weight.data, 0.01)    
    self.value_outp_layer.weight.data = normalized_columns_initializer(self.value_outp_layer.weight.data, 0.01)


    # softmax function for getting the action distributions later on
    # using the second dimension because this one includes the hidden unit activations
    self.act_smx = nn.Softmax(dim=2)

    # hyperparameter
    self.gamma = .9

    # set the optimzer
    if optimizer_str == 'RMS':
      self.optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate)
    elif optimizer_str == 'Adam':
      self.optimizer_str = torch.optim.Adam(self.parameters(), lr=learning_rate)
    else:
      self.optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate)
    
    # define the logfolders
    # used for the matplotlib images
    self.output_pref = log_dir + "/ownrnn_ep-"
    self.wr = SummaryWriter(log_dir=log_dir + '/tb')
    self.log_dir = log_dir

    return


  def __del__(self):
    self.wr.close();

  def do_training(self, taskenv, num_episodes = 20000, single_episode_length = 200, stats_every_x_episodes = 500):
    
    start_time = time.time()

    min_acc_episode_reward =0;

    print("Starting...")

    for x in range(num_episodes+1):

      taskenv.reset()

      # each episdoe has 200 game steps / trials
      episode_buffer = self.run_x_times(single_episode_length, taskenv)
      
      # estimation of loss and weight update
      info = self.calc_loss_and_update_weights(episode_buffer, t=x);


      ### from here on, we only have logging functions, can be edited as pleased
      new_highscore =  info['acc_ep_reward'] > min_acc_episode_reward;

      # this is the progress report info line that updates every 20 trials to the console
      if x % 20 == 0 or new_highscore: 

        if new_highscore: min_acc_episode_reward = info['acc_ep_reward'];
        minstr = "\t(New Highscore)" if new_highscore else ""

        print(int(x*100/num_episodes),"% - ", "Episode: ", x,  " \tLoss = ", info['loss'], " \tAccRew = ", info['acc_ep_reward'], minstr )


      if x % stats_every_x_episodes == 0:

        # output to console at every x episodes the stats for the most recent episode / gradient update
        print("###### Beg Netw-Starts ######")
        print("Name\t\t\t\t\tavgp\tmedp\tstdp\tminp\tmaxp\tsump")
        print_tensor_param_stats(self.value_outp_layer)
        print_tensor_param_stats(self.action_outp_layer)
        print_tensor_param_stats(self.lstm)
        print("...")
        print_tensor_param_stats(self.value_outp_layer, grad=True)
        print_tensor_param_stats(self.action_outp_layer, grad=True)
        print_tensor_param_stats(self.lstm, grad=True)
        print("###### End Netw-Starts ######")


        # console output of passed and estimated remaining time for all the episodes to complete
        elapsed_time = time.time() - start_time
        exp_total_duration = (elapsed_time/(x+1e-5)) * num_episodes
        remain_time = exp_total_duration - elapsed_time;

        print("-------------------------------")
        print("Total Runtime so far    : ",time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        print("Expected reminaing time : ",time.strftime("%H:%M:%S", time.gmtime(remain_time)))
        print("Expected total duration : ",time.strftime("%H:%M:%S", time.gmtime(exp_total_duration)))
        print("-------------------------------")

        self.plot(x, taskenv.stats)
        taskenv.stats = np.zeros((2,2,2));

        self.save_model(x);

      # log more detailed states every 2000 episodes to tensorboard (do this more rarely as it may be somewhat resource intensive)
      if x % 2000 == 0:

        log_step = int( x / stats_every_x_episodes )+1;
        add_tb_param_histograms(self.wr, self.action_outp_layer, log_step)
        add_tb_param_histograms(self.wr, self.value_outp_layer, log_step)
        self.wr.flush()
        add_tb_param_histograms(self.wr, self.lstm, log_step)        
        self.wr.flush()
        add_tb_param_histograms(self.wr, self.action_outp_layer, log_step, grad=True)
        add_tb_param_histograms(self.wr, self.value_outp_layer, log_step, grad=True)
        self.wr.flush()
        add_tb_param_histograms(self.wr, self.lstm, log_step, grad=True)
        self.wr.flush()

        print("addied histograms: ", 1)
        print("-------------------------------")



      ### end if episode % 100 = 0
    ### End for;
  ### end do training

  # this is basically our episode
  def run_x_times(self, number_of_feedforward_steps, taskenv):
    
    # we never are really done, because our automaton 
    # equals one episode with one feedforward thing, so instead
    # we define an episode as enough draws:

    # number_of_feedforward_steps ~ our batch size ~ episode length

    oh_prev_action        = F.one_hot(ts(0), self.num_actions)
    oh_prev_reached_state = F.one_hot(ts(0), self.num_states )
    prev_receivd_rewrd = ts([0]);    

    # initialize the hidden states / recursive inout to zeros
    cx = torch.zeros(1, self.num_rnn_units).view(1,1,-1)
    hx = torch.zeros(1, self.num_rnn_units).view(1,1,-1)
    
    #one batch ~ one episode of 200 trials, will be saved internally
    self.epb_values = torch.zeros((200))
    self.epb_entrop = torch.zeros((200))
    self.epb_logprb = torch.zeros((200))
    self.epb_policy = torch.zeros((200, self.num_actions))

    self.epbuffer = []
  
    taskenv.reset()

    for i in range(number_of_feedforward_steps):
      # i is also actually also our timestep variable

      cinput  = torch.cat((oh_prev_action, oh_prev_reached_state, prev_receivd_rewrd, ts([i])), 0).float().view(1,1,self.input_size);
      
      # run the lstm on a single trial
      out, (hx, cx) = self.lstm(cinput, (hx, cx))

      # LSTM output is used as input for two different layers
      # estimated value of the current state
      # ~ cummulative estimate for the future, kind of
      value = self.value_outp_layer(out)
      policy_out = self.action_outp_layer(out)

      # draw action from the last and only action distribution
      policy_smx = self.act_smx(policy_out)
      policy_distrib = policy_smx.contiguous()
      act_distr = torch.distributions.Categorical(policy_distrib.view(-1,self.num_actions)[-1])
      act = act_distr.sample()
     
      # mean entopy of our action distribution (not needed)
      # acc_entropy = acc_entropy+act_distr.entropy().mean()

      # execute action in the task_environment
      [reached_state, reward] = taskenv.conduct_action(act.item())
      # out: reached state is either 0 or 1; reward also either 0 or 1


      # do not keep any gradient releveant info, aka dont save any tensors
      # save as: action done -> stated reached upon that action -> reward received for that state, and actually predicted value of that action, all @ timepoint i
      self.epbuffer.append([act.detach(), reached_state, reward, i, value.item()]);

      # .copy_() copies in place and keeps the gradient information
      # all these keep reference to the original variables and their gradients
      # and thus 'taggs' them for update
      self.epb_values[i].copy_(value.squeeze())
      self.epb_entrop[i].copy_(act_distr.entropy().mean())
      self.epb_logprb[i].copy_(act_distr.log_prob(act))
      self.epb_policy[i].copy_(policy_smx.squeeze())


      # prepare vars for the next trial
      oh_prev_reached_state = F.one_hot(ts(reached_state), self.num_states)
      oh_prev_action = F.one_hot(act, self.num_actions)
      prev_receivd_rewrd = ts([reward])

    # end of for number of feedforward steps

    cinput  = torch.cat((oh_prev_action, oh_prev_reached_state, prev_receivd_rewrd, ts([number_of_feedforward_steps])), 0).float().view(1,1,self.input_size);
    out, (hx, cx) = self.lstm(cinput, (hx, cx))
    self.final_value = self.value_outp_layer(out).squeeze().detach()
    
    return self.epbuffer;




  def calc_loss_and_update_weights(self, epbuffer=None, t = 0):

    # run the entire set of 200 trials again, not one by one, but instead as batch
    # use exactly the same data & actions as before
    # just so we have a better way for backpropagating the single 
    # as it works by having the entire input as matrix


    # standard procedure is to take the internal buffer, it can also be given explicitly to make it nicer
    # to make the code more readily readable
    if epbuffer != None: epbuffer=self.epbuffer;

    epbuffer = np.array(epbuffer)

    ## prepare the input

    actions         = epbuffer[:,0]                      # based on the policy head output of the A2C
    reached_states  = epbuffer[:,1]
    rewards         = epbuffer[:,2].astype(np.long)      # may be nessesary, as nparray may happen to be of type object np array, if we 
    timesteps       = epbuffer[:,3].astype(np.long)      # i.e. use it to be tensors, otherwise no problem (so could also leave it away)
    pred_values     = epbuffer[:,4]                      # based on the value head output of the A2C


    # get the buffered network outputs with the gradient info preserved
    # in contrast to epbuffer, which only keeps track of things without grad
    values = self.epb_values;
    policy_out = self.epb_policy;


    ## do the loss calculation 

    # calculate the policy loss (has biggest influence)
    ohactions = F.one_hot(ts(actions.tolist()).long(), self.num_actions)
    resp_outps = torch.sum(policy_out.squeeze() * ohactions, dim=1)

    value_plus = np.asarray(pred_values.tolist() + [0.0])
    #value_plus = np.asarray(values.squeeze().tolist() + [0.0])
    und_adv = rewards + self.gamma * value_plus[1:] - value_plus[:-1];
    advantages = discount(und_adv, self.gamma)

    policy_loss = -torch.sum(torch.log(resp_outps +1e-7) * ts(advantages.copy()))


    # calculate the value loss 
    # compute the targets for the value head
    rewards_plus = np.asarray(rewards.tolist() + [0.0])
    # equals target_v, these are our targets, because they are the only real value we have
    disc_cumm_future_rewards = discount(rewards_plus,self.gamma)[:-1]
    
    # have to create a copy of the numpy array, to have the conscutive items of the array also in 
    # conscutive positions on the memory, which is required for the transformation into a tensor
    diff = ts(disc_cumm_future_rewards.copy())  - values.squeeze();
    value_loss = 0.5* torch.sum( diff * diff)


    # calculate the entropy loss 
    # how certain is the network of its own decision
    entropy_loss = -torch.sum( policy_out * torch.log(policy_out + 1e-7))

    # conbine it all into one loss
    loss =  0.05 * value_loss + policy_loss - 0.05 * entropy_loss


    # reset the gradient
    self.optimizer.zero_grad()

    # calculate the gradient
    #loss.backward(retain_graph=True);
    loss.backward();

    # make sure the gradient is not too big    
    torch.nn.utils.clip_grad_norm_(self.parameters(), 999.0)


    ### Here do all the bookkeeping
    # gradient will be applied afterwards
    self.wr.add_scalars('losses', {'loss': loss, 'val_loss': value_loss, 'pol_loss': policy_loss, 'ent_loss' : entropy_loss}, t)

    self.wr.add_scalar('sum_rewards', rewards.sum(), t)

    # plot the parameters before the gradients have been applied
    self.wr.add_scalars('ValueLayerParams', get_tb_dir_for_tensor_param_stats(self.value_outp_layer), t)
    self.wr.add_scalars('PolcyLayerParams', get_tb_dir_for_tensor_param_stats(self.action_outp_layer), t)
    self.wr.add_scalars('LSTMNLayerParams', get_tb_dir_for_tensor_param_stats(self.lstm), t)

    self.wr.add_scalars('ValueLayerCGrads', get_tb_dir_for_tensor_param_stats(self.value_outp_layer, grad=True), t)
    self.wr.add_scalars('PolcyLayerCGrads', get_tb_dir_for_tensor_param_stats(self.action_outp_layer, grad=True), t)
    self.wr.add_scalars('LSTMNLayerCGrads', get_tb_dir_for_tensor_param_stats(self.lstm, grad=True), t)


    # apply the gradient
    self.optimizer.step()

    return {'loss': loss.item(), 'acc_ep_reward':  rewards.sum().item() }

  def save_model(self, episode_count):
    save_pth = log_dir + '/model.tar';
    if os.path.isfile(save_pth): os.remove(save_pth);
    print("Saving model after training: ", save_pth)
    torch.save({
    'model_type_version': os.path.realpath(__file__) + ":OwnRnn",
    'epochs_trained': episode_count,
    'model_state_dict': self.state_dict(),
    'optimizer_state_dict' : self.optimizer.state_dict(),
    'optimizer': type(optimizer)
    }, save_pth)


  ## from here on, we only have plotting functions
  def plot(self, episode_count, transition_count):

        fig, ax = plt.subplots()
        x = np.arange(2)
        ax.set_ylim([0.0, 1.0])
        ax.set_ylabel('Stay Probability')

        row_sums = transition_count.sum(axis=-1)
        stay_probs = transition_count / row_sums[:,:,np.newaxis]

        # own rsc
        uncommon = [stay_probs[1,0,1],stay_probs[0,0,1]]
        common = [stay_probs[1,1,1],stay_probs[0,1,1]]

                
        ax.set_xticks([1.3,3.3])
        ax.set_xticklabels(['Last trial rewarded', 'Last trial not rewarded'])
        
        c = plt.bar([1,3],  common, color='b', width=0.5)
        uc = plt.bar([1.8,3.8], uncommon, color='r', width=0.5)
        ax.legend( (c[0], uc[0]), ('common', 'uncommon') )
        
        path = self.output_pref + str(episode_count) + ".png"
        plt.savefig(path)
        print("Saved plot as: ", path)


def print_tensor_param_stats(params, grad = False):
    retdir = {}
    for p in params.named_parameters():
      name = params.__str__()[:15] +"\t " + ("g~" if grad else "") + p[0]
      citem = p[1] if grad==False else p[1].grad;
      
      if citem!=None:
        maxp = citem.max().item();
        minp = citem.min().item();
        medp = citem.median().item();
        avgp = citem.mean().item();

        avgp = citem.mean().item();
        absavgp = citem.abs().mean().item();

        stdp = citem.std(unbiased=True).item();
        sump = citem.sum().item()

        retdir.update({name+"_avgp":avgp, name+"_abs_avgp":absavgp, name+"_stdp":stdp, name+"_minp":minp, name+"_maxp":maxp,name+"_sump":sump})
        phrase = (name +" "*34)[:34] + ("\t%+8.4f\t%+8.4f\t%+8.4f\t%+8.4f\t%+8.4f\t%+8.4f" % (avgp, medp, stdp, minp, maxp,sump)) ;
      else:
        phrase = "{:15s}".format(name) + " \tNone";
      print(phrase)
    return retdir;

def get_tb_dir_for_tensor_param_stats(params, grad = False):
    retdir = {}
    for p in params.named_parameters():
      name = ('gr/' if grad else "") + params.__str__() +"_" + p[0]
      citem = p[1] if grad==False else p[1].grad;
      
      if citem!=None:
        maxp = citem.max().item();
        minp = citem.min().item();
        medp = citem.median().item();
        avgp = citem.mean().item();

        absavgp = citem.abs().mean().item();

        stdp = citem.std(unbiased=True);
        sump = citem.sum().item()
        abssump = citem.abs().sum().item()
        sump = citem.abs().sum().item()

        retdir.update({name+"/avgp":avgp, name+"_abs_avgp":absavgp})
        if not torch.isnan(stdp):
          retdir.update({name+"/stdp":stdp.item(), name+"/minp":minp, name+"/maxp":maxp,name+"/abssump":abssump, name+"/sump":sump})

    #print(retdir)
    return retdir; 

def add_tb_param_histograms(wr, params, t, grad = False):
    retdir = {}
    for p in params.named_parameters():
      name = ('grad_' if grad else "") + params.__str__() +"_" + p[0]
      citem = p[1] if grad==False else p[1].grad;

      citem = citem.reshape([-1])

      if citem != None:
        wr.add_histogram('histogram of ' + name, citem, t, bins = 'tensorflow', max_bins=200);
      else: 
        print("Warning: ", name, " is none.")
    



if __name__ == '__main__':

  timest = strftime("%m%d-%H%M%S", gmtime())
  log_dir = 'out1/' + timest
  optimizer = 'RMS'

  env = rsc_two_step();

  # create and train the model
  rnn = OwnRnn(log_dir = log_dir, optimizer_str = optimizer)
  rnn.do_training(env, num_episodes = 20000, single_episode_length = 200, stats_every_x_episodes = 500)


  # save the model, and load again for prediciton

  save_pth = log_dir + '/model.tar';
  os.remove(save_pth)
  print("###########################################")
  print("Saving model after training: ", save_pth)

  torch.save({
  'model_type_version': os.path.realpath(__file__) + ":OwnRnn",
  'model_train_timestamp': timest,
  'epochs_trained': 20000,
  'model_state_dict': rnn.state_dict(),
  'optimizer_state_dict' : rnn.optimizer.state_dict(),
  'optimizer': type(rnn.optimizer)
  }, save_pth)

  print("Loading again from ", save_pth)
  checkpoint = torch.load(save_pth)
  loadedrnn = OwnRnn(log_dir = log_dir, optimizer_str = optimizer)
  loadedrnn.load_state_dict(checkpoint['model_state_dict'])
  loadedrnn.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  print("Maybe successfully loaded model if type/version: ", checkpoint['model_type_version'] )

  # make ready to only use for predictions; alternatively call model.train() to put it into training mode 
  # and continue with training
  loadedrnn.eval()

  print("Start running some predictions with that model: ")

  env.stats = np.zeros((2,2,2));
  rewards = []
  # run 100 episodes:
  for x in range(1,100):
    episode_result = loadedrnn.run_x_times(200, env)
    sum_rewards = np.array(episode_result)[:,2].sum();
    rewards = rewards + [sum_rewards]

  # ownrnn-xpred100.png will be the filename
  loadedrnn.plot('xpred100', env.stats)
  print("Prediction rewards: ", rewards)
  print("Mean rewards of 100 episodes predictions: ", np.array(rewards).mean())


