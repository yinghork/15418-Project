import matplotlib
import matplotlib.pyplot as plt
import time
import matplotlib.cm as cm
import numpy as np
from matplotlib.animation import FuncAnimation

iter = 0
leader = 0
X, Y = [], []
xdim, ydim = 0,0


while(True):

    try:
        f = open("./output/framePositions_" + str(iter) + ".txt",'r')
    except OSError:
        break

    with f:
        
        x = []
        y = []

        for i,row in enumerate(f):
            
            if(i != 0):
                values = [float(s) for s in row.split()]
                x.append(values[0])
                y.append(values[1])
            elif(i == 0 and xdim == 0):
                values = [int(s) for s in row.split()]
                xdim = values[0]
                ydim = values[1]
        
        X.append(x)
        Y.append(y)

        iter += 1


colors = ["blue" for i in range(len(X[0]))]
colors[leader] = "red"

fig, ax = plt.subplots()
ax.axis([-xdim/2, xdim/2,-ydim/2, ydim/2])
ax.set_aspect("equal")

line = ax.scatter([0 for i in range(len(X[0]))], [0 for i in range(len(X[0]))], c = colors)

# Updating function, to be repeatedly called by the animation
def animate(timestep):
    # obtain point coordinates 
    x,y = X[timestep], Y[timestep]

    line.set_offsets(np.column_stack([x, y]))
    line.set_facecolor(colors)

    # ax.clear()

    # # replot 
    # line = ax.scatter(x, y, c = colors)
    return line,

ani = FuncAnimation(fig, animate, frames=len(X), interval=50, blit=True, repeat=False)

plt.show()