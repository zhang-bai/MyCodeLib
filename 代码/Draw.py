"""
1.double_curve
    同一个坐标系下多条曲线


"""
import numpy as np


def double_curve(inputs: np.ndarry):
    data = inputs
    row, col = data.shape
    x = b[:,0]

    fig,ax = plt.subplots()

    plt.plot(x, data[:, 1],'-',label='test acc',color='darkorange')
    plt.plot(x, data[:,2], '-', label='most acc')

    ax.set_xlabel('seed')
    ax.set_ylabel('acc', rotation='vertical', horizontalalignment='right')

    ax.tick_params(direction='in')
    ax.grid()


    plt.legend()

    plt.show()