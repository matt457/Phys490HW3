import numpy as np
import matplotlib.pyplot as plt
from data import Data
import seaborn as sns
import sys

if __name__ == '__main__':

    # Command line arguments
    arg = sys.argv
    data_path = arg[1]

    # Parse data
    (s, p_data) = Data(data_path)
    s_shift = np.roll(s, -1, axis=1)

    # Hyperparameters
    learning_rate = 1
    num_epochs = 50
    J = np.random.rand(4)*2 - 1 # initialize weights randomly from -1 to 1
    KL_loss_list = []

    xi_xj = s*s_shift

    for epoch in range(num_epochs):
        temp = np.multiply(J, xi_xj)
        E = -np.sum(temp, axis=1)
        exp = np.exp(-E)
        Z = np.sum(exp)
        p_model = exp/Z

        neg_phase = np.sum( (p_model * xi_xj.T).T , axis=0 )
        pos_phase = np.sum( (p_data * xi_xj.T).T , axis=0 )
        delta_J = learning_rate * (pos_phase - neg_phase)
        J += delta_J

        KL_loss = np.sum(p_data * np.log(p_data/p_model))
        KL_loss_list.append(KL_loss)

    J = np.round(J) # round should make them +/-1
    J_dict = {
        '(0, 1)': J[0],
        '(1, 2)': J[1],
        '(2, 3)': J[2],
        '(3, 0)': J[3]}
    print(J_dict)

    plt.plot(np.arange(num_epochs),KL_loss_list)
    plt.title('KL loss vs. epoch')
    plt.grid()
    plt.savefig('plots/KL_loss.png')




