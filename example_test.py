import numpy as np
import matplotlib.pyplot as plt
from histogram_filter import HistogramFilter
import random


if __name__ == "__main__":

    # Load the data
    data = np.load(open('data/starter.npz', 'rb'))
    cmap = data['arr_0']
    actions = data['arr_1']
    observations = data['arr_2']
    belief_states = data['arr_3']


    #### Test your code here
    n,m = cmap.shape[0], cmap.shape[1]
    belief = np.ones((n,m))*(1/(n*m))
    hp = HistogramFilter()
    
    for i in range(observations.shape[0]):
        print('action =',actions[i])
        belief, coord = hp.histogram_filter(cmap, belief, actions[i], observations[i])
        if(abs(belief[coord[0], coord[1]] - belief[belief_states[i][0], belief_states[i][1]])!=0):
            print("{} diverges".format(i+1))
            print("value of calculated state = %f\n value of expected state is = %f"%(belief[coord[0], coord[1]], belief[belief_states[0][0], belief_states[0][1]]))
        print('my=',coord)
        print('given=',belief_states[i])
        print('\n\n')