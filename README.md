# 15418-Project


### TITLE
Optimizing Parallelization of Boid Simulation

Claire Ko (yinghork), Haley Carter (hcarter)


### SUMMARY
We are going to implement an optimized Boid Movement Calculation on the NVIDIA GPUs in the lab. We will use various work distribution strategies and communication mechanisms to create a more efficient simulation of many objects. 


### BACKGROUND

The combined motion of a group, called "boids", is common in nature. Thus, simulating the behaviors of a large number of moving objects is important in fields such as virtual reality and computer animation, where each group member interacts with its neighbors. When simulating the group, each individual boid will compute its position in the next time step, without colliding with its neighbors and attempting to move towards a global goal or attractor. The general group behavior is defined from the individual boid movement and the (possibly changing) location of the global goals. The image below describes the three rules for boid movement, in relation to the neighbors of each individual boid: 

![image](https://user-images.githubusercontent.com/56246022/159631377-de96b45e-c626-4197-9c08-5b721b90ea6c.png)

Current solutions to the boid simulation problem are not very scalable, in terms of the number of boids, because computation cost as well as the communication costs increase. Our goal is to improve on the current solutions, to build a parallel program to simulate a smooth, generally correct movement for each boid, but not to build a good parallel visual interface for our computations. 

Aspects of the problem that might benefit from parallelism include computing the next best location to move for each boid, and efficient communication between the boids. As there are many boids, it will be essential to parallelize the computing process for every boid, where parallelization is across boids. Since different boids have data dependencies, such as the fact that one boid should avoid the other boids around it, while moving towards/following a global goal location, and the optimization is carried over all boids, our parallel implementation will require communication, either shared state or message passing, and synchronisation over different processes.


### THE CHALLENGE

The problem is challenging because we will need to devise a way to achieve good load balancing across processors. In other words, we will explore optimal ways to distribute work between threads, such that there will be generally equal computation between them. This is challenging because the boids are changing with each time step, and some may have more neighbors than others, and thus more computation than others. Mapping the workload to the current group state is not constant, and will shift as the group moves. 

Another challenge is minimizing communcation and synchronization costs; each boid depends on its neighbors, and so the neighbors' locations will have to be accessed by the current boid to allow for the detection of neighborhood relationships as shown in the image above. There are data dependencies across boids when searching for the next location, because boids share the same physical space. There is data locality as each boid depends only on the general neighbors around it; however, the range of neighbor distance that is covered will need to be tuned. 

We plan to compare the shared address space model with the message passing model using MPI, OpenMP, and Cilk. We also plan to determine and tune the tradeoff between better global movement, that follow the movement rules, versus better speedup and less communication overhead. We will determine how to evaluate the quality of a boid's move, how to evaluate speedup, as well as clearly define the evaluation metrics of the "goodness" of the solution. 


### RESOURCES

Resources that we will use include the following paper, which describes the idea of boids:

1. [Flocks, Herds, and Schools: A Distributed Behavioral Model](https://dl.acm.org/doi/pdf/10.1145/37402.37406)


2. [GPU enhanced parallel computing for large scale data clustering](https://www.sciencedirect.com/science/article/pii/S0167739X12001707)

3. [Parallel Cloud Movement Forecasting based on a Modified Boids Flocking Algorithm](https://ieeexplore.ieee.org/abstract/document/9521639)
4. [Autonomous Boids](https://onlinelibrary.wiley.com/doi/10.1002/cav.123)
5. [Parallel Bird Flocking Simulation](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.48.8430&rep=rep1&type=pdf)
6. [A Neighborhood Grid Data Structure for Massive 3D Crowd Simulation on GPU](https://ieeexplore.ieee.org/abstract/document/5479102)



We will also learn from various sources (not yet determined) about building a boid simulator, the boid movement constraints and how to represent them in code, and how to construct a evaluation metric for determining the "goodness" of the solution.


### GOALS AND DELIVERABLES
GOALS AND DELIVERABLES: Describe the deliverables or goals of your project. This is by far the most important section of the proposal!
Separate your goals into what you PLAN TO ACHIEVE ("100%" -- what you believe you must get done to have a successful project and get the grade you expect) and an extra goal or two ("125%") that you HOPE TO ACHIEVE if the project goes really well and you get ahead of schedule, as well as goals in case the work goes more slowly ("75%"). It may not be possible to state precise performance goals at this time, but we encourage you be as precise as possible. If you do state a goal, give some justification of why you think you can achieve it. (e.g., I hope to speed up my starter code 10x, because if I did it would run in real-time.)

If applicable, describe the demo you plan to show at the poster session (Will it be an interactive demo? Will you show an output of the program that is really neat? Will you show speedup graphs?). Specifically, what will you show us that will demonstrate you did a good job?
If your project is an analysis project, what are you hoping to learn about the workload or system being studied? What question(s) do you plan to answer in your analysis?
Systems project proposals should describe what the system will be capable of and what performance is hoped to be achieved.


We plan to achieve to have a sequential, and then multiple (2-3) parallel versions of boid movement computation, as well as a visual representation of the movement (that may not be parallelized, as it is simply to show the result of the computations). 

We hope to achieve a parallelized, real time version of the visuals, and not just the computations, as this will show visually the difference between the sequential and parallel computation models. 

If things move slowly, we may only have 1 or 2 parallel versions of the sequential code for computing boid movement, as we explored less than we wanted to in terms of various parallel frameworks and communication models.

Our demo will contain graphs of speedup over various inputs, as well as graphs comparing the speedup over different parallel versions. Additionally, we will run a visual demonstration of the boid movement, which will most likely be created with Unity. We are hoping to answer the question of what is the best load balance in this specific use case, as well as what communication model benefits this computation the most. In general, we want to be able to compare the models we have learned in class, for both computation and communication, and explore how they can work together.  


### PLATFORM CHOICE

We will be using the GHC machines for development, testing, and experiments, because it is more accessible and contains CUDA, OpenMP, and MPI support needed for development. We hope to use Bridges-2 in order to leverage the machines on there, to get more accurate measurements. 


### SCHEDULE

Week 1: 3/28 - 4/1	  Research about boids and boid movement algorithms

Week 3: 4/3 - 4/8	    Implement sequential/baseline version of boid movement

4/11	                Milestone Report

Week 3: 4/11 - 4/15	  Parallelize the sequential program using different parallel frameworks 

Week 4: 4/18 - 4/22	  Optimize the parallel implementation, conduct experiments and analysis on different parallel frameworks and communication models

Week 5: 4/25 - 4/29	  Write project report and prepare for poster session

4/29                  Final Report

5/5                   Poster Session

