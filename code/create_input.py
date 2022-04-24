import random

try:
    f = open("./input/hard.txt",'wb')
except OSError:
    exit(0)

with f:

    num_boids = 1000
    x = 2000
    y = 2000

    f.write((str(x) + " " + str(y)).encode())
    f.write("\n".encode())
    f.write((str(num_boids)).encode())
    f.write("\n".encode())

    for i in range(num_boids):

        x = random.randrange(-500, 500)
        y = random.randrange(-500, 500)

        f.write((str(x) + " " + str(y)).encode())

        if(i != num_boids - 1):
            f.write("\n".encode())
    
    f.close()