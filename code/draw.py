import matplotlib
import matplotlib.pyplot as plt
import time
import matplotlib.cm as cm
import numpy as np

iter = 0
leader = 0

while(True):

    try:
        f = open("./output/framePositions_" + str(iter) + ".txt",'r')
    except OSError:
        break

    with f:

        X, Y = [], []

        for i,row in enumerate(f):
            
            if(i != 0):
                values = [int(s) for s in row.split()]
                X.append(values[0])
                Y.append(values[1])

        plt.title("framePositions_" + str(iter) + ".txt", fontsize = 20)
        colors = ["blue" for i in range(len(Y))]
        colors[leader] = "red"
        plt.scatter(X, Y, color = colors)
        plt.xlim([-500, 500])
        plt.ylim([-500, 500])
        
        plt.draw()
        plt.pause(.05)
        plt.clf()

        iter += 1