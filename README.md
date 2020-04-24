# metalearn_rnn

This is a pytorch implementation of the agent described in [Wang et al., 2018](https://www.biorxiv.org/content/biorxiv/early/2018/04/13/295964.full.pdf). 

This largely mirros and is based on on a tensorflow implementation of the same agent by [Michaël Trazzi and Yasmine Hamdani, under the supervision of Olivier Sigaud](https://github.com/mtrazzi/two-step-task), who themselves made use of [awjuliani's Meta-RL implementation](https://github.com/awjuliani/Meta-RL)(a single-threaded A2C LSTM implementation). For conceptual understanding you can also read [Michaels article on Medium](https://blog.floydhub.com/meta-rl/) and [Arthurs post](https://medium.com/hackernoon/learning-policies-for-learning-policies-meta-reinforcement-learning-rl%C2%B2-in-tensorflow-b15b592a2ddf).

Additionally to previous implementations, I added more logging and put a focus on code readibility/understanding of the implementation.

Specifically:
* `basenet.py` contains the basic implementation
* `basenet_onepass.py` just uses one pass for each trial through the network instead of two (episode rollout and gradient tags are gathered from one forward pass of the network); the computed gradients are identical in both approaches
* `basenet_altloss.py` uses a different kind of loss (as described on [here](https://github.com/rpatrik96/pytorch-a2c/blob/master/src/runner.py)); havent yet found a working hyperparameter config (or maybe my implementation is wrong)
* `basenet_altloss2.py` uses a different kind of loss (using the code of [Ilya Kostrikov](https://github.com/ikostrikov/pytorch-a3c/blob/master/train.py)); 
* extnet.py introduces a novel task (described in XXX). It also only uses one forward pass through the network to calculate everything nessesary.

This is my first implementation of an ANN and hence errors / suboptimal code may be contained.
An online version can also be found on google colab: XXX

## Requirements

The python scripts were written using python 3.7 and pytorch 1.4.0+cpu
```sh
# Pytorch Version: 1.4.0+cpu
pip install torch numpy scipy

# for visuzalization / plotting the images 
pip install matplotlib
# for timing?
pip install pytictoc
```

## Execution
```sh
python -u sscandn.py
```

## Findings

to view the progress in greater detail use tensorboard, i.e.

```
tensorboard --logdir metalearn_rnn\out1\0421-193012\tb
```

### Results

* convergence with PDSP otimizer @ roughly 12k
* learns how to play the game
* not relfected in loss but much more the cummulative reward per episode (random reward baseline would be 100)

### Early weight changes anticipating market improvement (timeline from one execution)


| Step/Epoch    | Are           |
| :------------ |:-------------:|
| 8k            | a few LSTM weights are increased  |
| 12k           | centered      |
| 15k           | are neat      |

[Images here]

### Tensorboard

dada

### LSTM implementation in torch

even though on a high level we have a lot of different gates, the can be trained by the same forumlas, hence they were pooled into two variables in their pytorch implementation. for more background see here and here. 

### Loss.backward() in PyTorch for A2C/multiple types of losses

tl;dr: gradients are computed for each loss part separately and summed up at the level of weights.

I still find it quite confusing how the loss is backpropagated within pytorch. Some good explanations with very simple examples can be found here, here and here. Yet I wasnt sure how for example the gradient is computed when a single variable (i.e. the ih_weights of the LSTM) feature in two kinds of losses. Hence I tried to dissect it using the following code that plots stats about the calculated gradients at the different variables after distinct backwarded losses:

```python
# we always need to make sure retrain=true, otherwise the saved activation per variable (i.e. weights)
# is lost after the first .backward() call
loss.backward(retain_graph=True);

print("#######################\n#-- Normal Loss backward: 0.05 * value_loss + policy_loss - 0.05 * entropy_loss")
print_tensor_param_stats(self.value_outp_layer, grad=True)
print_tensor_param_stats(self.action_outp_layer, grad=True)
print_tensor_param_stats(self.lstm, grad=True)
...
# zero the gradients again, and do a different loss backward propagation
self.optimizer.zero_grad()
alt_loss= 0.05 * entropy_loss;
alt_loss.backward(retain_graph=True)
print("#######################\n#-- entropy loss backward: 0.05 * entropy_loss")
print_tensor_param_stats(self.value_outp_layer, grad=True)
print_tensor_param_stats(self.action_outp_layer, grad=True)
print_tensor_param_stats(self.lstm, grad=True)
...
```

Done for all kinds of different losses This yielded the following (sample output for episode nr 20; using raw, unclipped gradient info):

```
Name                                    avgp    medp    stdp      minp      maxp      sump
-----------------------------------------------------------------------------------------------
#######################
#-- Normal Loss backward: 0.05 * value_loss + policy_loss - 0.05 * entropy_loss
Linear(in_featu	 g~weight         	 -2.3694	 -8.8223	+10.4206	-11.7314	+11.7338	-113.7291
Linear(in_featu	 g~bias           	-12.1059	-12.1059	    +nan	-12.1059	-12.1059	-12.1059
Linear(in_featu	 g~weight         	 -0.0000	-11.1242	+29.6898	-32.6978	+32.6978	 -0.0000
Linear(in_featu	 g~bias           	 -0.0000	-31.4037	+44.4115	-31.4037	+31.4037	 -0.0000
LSTM(6, 48)	 g~weight_ih_l0       	 -0.2958	 +0.0005	 +2.7064	-43.3604	 +2.1435	-340.8133
LSTM(6, 48)	 g~weight_hh_l0       	 -0.0020	 +0.0001	 +0.0479	 -0.3188	 +0.3197	-18.8611
LSTM(6, 48)	 g~bias_ih_l0         	 -0.0105	 +0.0016	 +0.0562	 -0.3248	 +0.1617	 -2.0241
LSTM(6, 48)	 g~bias_hh_l0         	 -0.0105	 +0.0016	 +0.0562	 -0.3248	 +0.1617	 -2.0241
#######################
#-- After second time Loss backward
Linear(in_featu	 g~weight         	 -4.7387	-17.6445	+20.8411	-23.4627	+23.4676	-227.4583
Linear(in_featu	 g~bias           	-24.2118	-24.2118	    +nan	-24.2118	-24.2118	-24.2118
Linear(in_featu	 g~weight         	 -0.0000	-22.2485	+59.3796	-65.3956	+65.3956	 -0.0001
Linear(in_featu	 g~bias           	 -0.0000	-62.8074	+88.8231	-62.8074	+62.8074	 -0.0000
LSTM(6, 48)	 g~weight_ih_l0       	 -0.5917	 +0.0011	 +5.4129	-86.7208	 +4.2870	-681.6266
LSTM(6, 48)	 g~weight_hh_l0       	 -0.0041	 +0.0001	 +0.0959	 -0.6377	 +0.6393	-37.7222
LSTM(6, 48)	 g~bias_ih_l0         	 -0.0211	 +0.0032	 +0.1124	 -0.6495	 +0.3234	 -4.0482
LSTM(6, 48)	 g~bias_hh_l0         	 -0.0211	 +0.0032	 +0.1124	 -0.6495	 +0.3234	 -4.0482
#######################
#-- After zero grad
Linear(in_featu	 g~weight         	 +0.0000	 +0.0000	 +0.0000	 +0.0000	 +0.0000	 +0.0000
Linear(in_featu	 g~bias           	 +0.0000	 +0.0000	    +nan	 +0.0000	 +0.0000	 +0.0000
Linear(in_featu	 g~weight         	 +0.0000	 +0.0000	 +0.0000	 +0.0000	 +0.0000	 +0.0000
Linear(in_featu	 g~bias           	 +0.0000	 +0.0000	 +0.0000	 +0.0000	 +0.0000	 +0.0000
LSTM(6, 48)	 g~weight_ih_l0       	 +0.0000	 +0.0000	 +0.0000	 +0.0000	 +0.0000	 +0.0000
LSTM(6, 48)	 g~weight_hh_l0       	 +0.0000	 +0.0000	 +0.0000	 +0.0000	 +0.0000	 +0.0000
LSTM(6, 48)	 g~bias_ih_l0         	 +0.0000	 +0.0000	 +0.0000	 +0.0000	 +0.0000	 +0.0000
LSTM(6, 48)	 g~bias_hh_l0         	 +0.0000	 +0.0000	 +0.0000	 +0.0000	 +0.0000	 +0.0000
#######################
#-- Again after backward
Linear(in_featu	 g~weight         	 -2.3694	 -8.8223	+10.4206	-11.7314	+11.7338	-113.7291
Linear(in_featu	 g~bias           	-12.1059	-12.1059	    +nan	-12.1059	-12.1059	-12.1059
Linear(in_featu	 g~weight         	 -0.0000	-11.1242	+29.6898	-32.6978	+32.6978	 -0.0000
Linear(in_featu	 g~bias           	 -0.0000	-31.4037	+44.4115	-31.4037	+31.4037	 -0.0000
LSTM(6, 48)	 g~weight_ih_l0       	 -0.2958	 +0.0005	 +2.7064	-43.3604	 +2.1435	-340.8133
LSTM(6, 48)	 g~weight_hh_l0       	 -0.0020	 +0.0001	 +0.0479	 -0.3188	 +0.3197	-18.8611
LSTM(6, 48)	 g~bias_ih_l0         	 -0.0105	 +0.0016	 +0.0562	 -0.3248	 +0.1617	 -2.0241
LSTM(6, 48)	 g~bias_hh_l0         	 -0.0105	 +0.0016	 +0.0562	 -0.3248	 +0.1617	 -2.0241
#######################
#-- Policy loss backward
Linear(in_featu	 g~weight         	 +0.0000	 +0.0000	 +0.0000	 +0.0000	 +0.0000	 +0.0000
Linear(in_featu	 g~bias           	 +0.0000	 +0.0000	    +nan	 +0.0000	 +0.0000	 +0.0000
Linear(in_featu	 g~weight         	 -0.0000	-10.5594	+28.2874	-31.1688	+31.1688	 -0.0000
Linear(in_featu	 g~bias           	 +0.0000	-29.8683	+42.2401	-29.8683	+29.8683	 +0.0000
LSTM(6, 48)	 g~weight_ih_l0       	 -0.0401	 +0.0029	 +0.8189	-13.2338	 +9.7934	-46.2375
LSTM(6, 48)	 g~weight_hh_l0       	 +0.0003	 +0.0002	 +0.0157	 -0.0966	 +0.0970	 +2.5330
LSTM(6, 48)	 g~bias_ih_l0         	 +0.0063	 +0.0074	 +0.0290	 -0.1087	 +0.1483	 +1.2022
LSTM(6, 48)	 g~bias_hh_l0         	 +0.0063	 +0.0074	 +0.0290	 -0.1087	 +0.1483	 +1.2022
#######################
#-- Value loss backward
Linear(in_featu	 g~weight         	-47.3872	-176.4451	+208.4111	-234.6275	+234.6759	-2274.5837
Linear(in_featu	 g~bias           	-242.1179	-242.1179	    +nan	-242.1179	-242.1179	-242.1179
Linear(in_featu	 g~weight         	 +0.0000	 +0.0000	 +0.0000	 +0.0000	 +0.0000	 +0.0000
Linear(in_featu	 g~bias           	 +0.0000	 +0.0000	 +0.0000	 +0.0000	 +0.0000	 +0.0000
LSTM(6, 48)	 g~weight_ih_l0       	 -5.0818	 -0.0107	+40.3586	-591.6526	 +4.3569	-5854.2749
LSTM(6, 48)	 g~weight_hh_l0       	 -0.0465	 -0.0009	 +0.7156	 -4.3632	 +4.3728	-428.3057
LSTM(6, 48)	 g~bias_ih_l0         	 -0.3368	 -0.0750	 +0.8561	 -4.5424	 +1.0780	-64.6722
LSTM(6, 48)	 g~bias_hh_l0         	 -0.3368	 -0.0750	 +0.8561	 -4.5424	 +1.0780	-64.6722
#######################
#-- entropy and value loss backward: 0.05 * value_loss - 0.05 * entropy_loss
Linear(in_featu	 g~weight         	 -2.3694	 -8.8223	+10.4206	-11.7314	+11.7338	-113.7291
Linear(in_featu	 g~bias           	-12.1059	-12.1059	    +nan	-12.1059	-12.1059	-12.1059
Linear(in_featu	 g~weight         	 -0.0000	 -0.5648	 +1.4027	 -1.5300	 +1.5300	 -0.0000
Linear(in_featu	 g~bias           	 +0.0000	 -1.5354	 +2.1714	 -1.5354	 +1.5354	 +0.0000
LSTM(6, 48)	 g~weight_ih_l0       	 -0.2557	 -0.0005	 +2.0433	-30.1266	 +0.2279	-294.5758
LSTM(6, 48)	 g~weight_hh_l0       	 -0.0023	 -0.0000	 +0.0362	 -0.2222	 +0.2227	-21.3941
LSTM(6, 48)	 g~bias_ih_l0         	 -0.0168	 -0.0037	 +0.0432	 -0.2312	 +0.0535	 -3.2263
LSTM(6, 48)	 g~bias_hh_l0         	 -0.0168	 -0.0037	 +0.0432	 -0.2312	 +0.0535	 -3.2263
...
```
* after calling `self.optimizer.zero_grad()`, all the gradients are reset to zero
* The gradients double when calling a second time i.e. avg  for first linear layer weight goes from -2.3694 to -4.7387
* gradients simply add up, i.e. adding up the `entropy and value loss` (-294.5758) and  `Policy loss` (-46.2375) weight sums for the LSTM ih_weights yields the `Normal Loss/original loss` (-340.8133).
* `value loss` only assigns/adds to the gradient of the first linear layer (the value output head) but not the second, and the opposite is true for `policy loss`
* there is no std for the `value bias` gradients (first linear layer), because we only have one bias unit and hence cannot calculate a standard deviation
* gradients in the beginning can become rather big, so maybe its a good idea to clip it using `torch.nn.utils.clip_grad_norm_(self.parameters(), 999.0)`


## OpenQuestions
There is certain things i do not understand fully yet
* ~~how exactly are the different classes of loss backpropagated, i.e. do their gradients add up or similiar~~ (answered above)
* why do we have to run each episode/epoch twice, one for accumulation of the episode buffer and one for the calculation of gradients, can that be merged?
* make this faster?
* at what level does gradient clipping work?
