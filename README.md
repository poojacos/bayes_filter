# bayes_filter
In this project a Bayes Filter for 2d discretized world is implemented. Every cell in the gridworld is characterized by a color (0 or 1). The robot is equipped with a noisy odometer and a noisy color sensor. 

The project contains the following files - 
- **histogram_filter.py** : This takes in a prior belief distribution, a colormap, action, and observation, and returns the posterior belief distribution according to the Bayes Filter.
- **example_test.py** : This is a started code to test the Bayes Filter implementation for the starter.npz data
- **starter.npz** : This file file contains a binary color-map, a sequence of actions, a sequence of observations, and a sequence of the correct belief states.

A description of the problem is as follows - 
### Sensor Model
z = true observation with probability 0.9
z = false observation with probability 0.1

### Action Model
x<sub>t+1</sub> = x<sub>t</sub>  with probability 0.9
x<sub>t+1</sub> = x<sub>t</sub> with probability 0.1

When the robot is at the edge of the gridworld and is tasked with executing an action that would take it outside the boundaries of the gridworld, the robot remains in the same state with p = 1. We start with a uniform prior on all states. 

This has been implemented as part of course ESE 650. 
