import numpy as np

def Data(data_path):
    '''Data manipulation
    Read in spin configurations
    Output list of unique configuration and their probabilities
    '''

    s_counter = {}

    f = open(data_path, "r")
    for x in f:
        my_str = x[0:4]
        
        if my_str in s_counter:
            s_counter[my_str] += 1
        else:
            s_counter[my_str] = 1

    s = np.zeros((len(s_counter),4))
    p_data = np.zeros(len(s_counter))
    for index,key in enumerate(s_counter):
        p_data[index] = (s_counter[key]/1000)
        s[index,:] = [1 if i=='+' else -1 for i in list(key)]
    
    return s, p_data