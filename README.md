# metalearn_rnn

This is the general high level description.  XXX is an implementation of the agent described in XXX. The implementation is based on a tensorflow implementation of the same agent by XXX . For conceptual understanding you can also reed his article on XXX.

* .py is the one to one transferred
* .py was modified to increase understanding (both task and ...) and includes loaded/saving options + fixing weights ...
* .py introduces certain modifications 
* .py introduces a novel task (described in XXX)

## Requirements

The python scripts were written using python 3.7 and pytorch X.X
```sh
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

### Early weight changes anticipating market improvement (timeline from one execution)

8k  -   dadasd
10k -   dudud

### Tensorboard


## OpenQuestions
There is certain things i do not understand fully yet
* how exactly are the different classes of loss backpropagated, i.e. do their gradients add up or similiar
* why do we have to run each episode/epoch twice, one for accumulation of the episode buffer and one for the calculation of gradients, can that be merged?
* make this faster?
* why 
