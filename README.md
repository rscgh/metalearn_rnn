# metalearn_rnn

This is the general high level description.  XXX is an implementation of the agent described in XXX. The implementation is based on a tensorflow implementation of the same agent by XXX . For conceptual understanding you can also reed his article on XXX.

* .py is the one to one transferred
* .py was modified to increase understanding (both task and ...) and includes loaded/saving options + fixing weights ...
* .py introduces certain modifications (adam?; different learning rates, different episode sizes)
* py uses a different training algorithm (as described on XXX)
* .py introduces a novel task (described in XXX)

This is my first implementation of a neural 

## Requirements

The python scripts were written using python 3.7 and pytorch X.X
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

to view the progress in greater detail use tensorboard

```
tensorboard --logdir metalearn_rnn\out1\0421-193012_firstworks\tb
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


## OpenQuestions
There is certain things i do not understand fully yet
* how exactly are the different classes of loss backpropagated, i.e. do their gradients add up or similiar
* why do we have to run each episode/epoch twice, one for accumulation of the episode buffer and one for the calculation of gradients, can that be merged?
* make this faster?
* why 
