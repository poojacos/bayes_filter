import numpy as np
from matplotlib import pyplot as plt

class HistogramFilter(object):
    """
    Class HistogramFilter implements the Bayes Filter on a discretized grid space.
    """

    def histogram_filter(self, cmap, belief, action, observation):
        '''
        Takes in a prior belief distribution, a colormap, action, and observation, and returns the posterior
        belief distribution according to the Bayes Filter.
        :param cmap: The binary NxM colormap known to the robot.
        :param belief: An NxM numpy ndarray representing the prior belief.
        :param action: The action as a numpy ndarray. [(1, 0), (-1, 0), (0, 1), (0, -1)]
        :param observation: The observation from the color sensor. [0 or 1].
        :return: The posterior distribution.
        Use starter.npz data given in /data to debug and test your code. starter.npz has sequence of actions and 
        corresponding observations. True belief is also given to compare your filters results with actual results. 
        cmap = arr_0, actions = arr_1, observations = arr_2, true state = arr_3
    
        ### Your Algorithm goes Below.
        '''
        n,m = cmap.shape[0], cmap.shape[1]
        m_action_right, m_action_left = np.zeros((n,m)), np.zeros((n,m))
        
        m_obs_0 = cmap*0.1
        m_obs_0[m_obs_0 == 0] = 0.9
        
        m_obs_1 = cmap*0.9
        m_obs_1[m_obs_1 == 0] = 0.1

        plt.imsave('m_obs_0.png',m_obs_0)
        # prior
        eta = 0
        final_bel = belief
        ans = []
        
        #creating action matrix
        for i in range(1, n):
            m_action_left[i][i] = 0.1
            m_action_left[i][i-1] = 0.9
        m_action_left[0][0] = 1
        

        m_action_right = np.flip(np.flip(m_action_left, axis = 1), axis=0)
    
        act = action
        # left
        if (act == [-1,0]).all() == True:
            final_bel = final_bel @ m_action_left
        #right
        elif (act == [1,0]).all() == True:
            final_bel = final_bel @ m_action_right
        #up
        elif (act == [0,1]).all() == True:
            final_bel = np.rot90((np.rot90(final_bel, k=1) @ m_action_right), k=-1)
        #down
        elif (act == [0,-1]).all() == True:
            final_bel = np.rot90((np.rot90(final_bel, k=-1) @ m_action_left), k=1)
        #unknown action
        else:
            print('Unknown action ', action)
        
        #perceptual data update
        obs = observation
        #obs = 0
        if obs == 0:
            final_bel = np.multiply(final_bel, m_obs_0) #element-wise
        #obs = 1
        elif obs == 1:
            final_bel = np.multiply(final_bel, m_obs_1) #element-wise
        #invaild observation
        else:
            print('invalid observation')
            
        # normalize
        eta += np.sum(final_bel)
        final_bel/=eta
        
        coords = np.unravel_index(final_bel.argmax(), final_bel.shape)
        print('these are the coordinates', coords[0], coords[1])
        return  final_bel, [coords[1], n-1-coords[0]]
        