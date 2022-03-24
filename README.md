# 15418-Project


### TITLE
Optimizing Parallelization of Boid Simulation

Claire Ko (yinghork), Haley Carter (hcarter)


### SUMMARY
We are going to implement an optimized Boid Movement Calculation on the NVIDIA GPUs in the lab. We will use various work distribution strategies and communication mechanisms to create a more efficient simulation of many objects. 


### BACKGROUND

===
The "boids" algorithm seeks to simulate the natural flocking motion of animals, primarily birds. Simulating the behaviors of a large number of moving agents interacting with one another is important in fields such as virtual reality and computer animation. The boids algorithm also has much in common with other n-body and particle system simulations, and has therefore seen use in domains related to those problems, such as weather simulation and data visualization. Rather than control the motion of the flock globally, the algorithm creates emergent flocking behavior through a few simple behaviors performed by each boid, based on the limited information within its range of vision. When simulating the flock at each time step, each individual boid updates its position and velocity towards some goal based on three forces. The image below describes the three rules for boid movement, in relation to the neighbors within the limited vision of each individual boid:

===

The combined motion of a group, called "boids", is common in nature. Thus, simulating the behaviors of a large number of moving objects is important in fields such as virtual reality and computer animation, where each group member interacts with its neighbors. When simulating the group, each individual boid will compute its position in the next time step, without colliding with its neighbors and attempting to move towards a global goal or attractor. The general group behavior is defined from the individual boid movement and the (possibly changing) location of the global goals. The image below describes the three rules for boid movement, in relation to the neighbors of each individual boid: 

![image](https://user-images.githubusercontent.com/56246022/159631377-de96b45e-c626-4197-9c08-5b721b90ea6c.png)

===
The general structure of the algorithm is as follows: for each time step, for each boid, find the boid's neighbors. For each neighbor, calculate its effect on separation, alignment, and cohesion forces. Calculate the goal-seeking force. Finally, combine those forces to calculate the new velocity and position.

While each boid is dependent on its neighbors, its limited range of vision means that its number of neighbors remains relatively constant as the number of boids in the flock increases. As well, the force calculations based on neighbors, then updating position and velocity, are the same for each boid. This ensures that the work performed by each boid in each time step is relatively similar, meaning that the update of each boid will benefit from a parallel implementation. Additionally, with proper workload division, a parallel application could benefit highly from locality, as boids near each other will also have similar sets of neighbors.

As we will discuss further below, a major aspect of the boids algorithm comes from determining the neighbors of each boid. The naive approach has O(n<sup>2</sup>) complexity in the number of boids in the flock, which means that the sequential implementation is not very scalable in the number of boids. While this is a challenge for implementation, any potential speedup from a parallel implementation is highly useful. Reducing the computation time of large scale simulation to real time would allow for significantly more use of the algorithm, such as real-time animation in video games.

===


Current solutions to the boid simulation problem are not very scalable, in terms of the number of boids, because computation cost as well as the communication costs increase. Our goal is to improve on the current solutions, to build a parallel program to simulate a smooth, generally correct movement for each boid, but not to build a good parallel visual interface for our computations. 

Aspects of the problem that might benefit from parallelism include computing the next best location to move for each boid, and efficient communication between the boids. As there are many boids, it will be essential to parallelize the computing process for every boid, where parallelization is across boids. Since different boids have data dependencies, such as the fact that one boid should avoid the other boids around it, while moving towards/following a global goal location, and the optimization is carried over all boids, our parallel implementation will require communication, either shared state or message passing, and synchronisation over different processes.


### THE CHALLENGE

As described above, the main challenge of the algorithm comes in neighbor determination. Each boid is depend on its neighbrs in the flock, but due to the fact that all boids are constantly updating their positions, the neighbors of a boid will change over time. This necessitates updating the set of neighbors for each boid, which naively has O(n<sup>2</sup>) complexity. Therefore, we will need to explore more efficient approaches to finding the neighborhood of each boid. 

Additionally, the changing nature of the system combined with the dependence on nearby neighbors provides another challenge for workload distribution. There is high locality between boids near one another in each particular time step, which we could take advantage of with our workload distribution. However, we will also need to be careful to limit the amount of communication required between processors. As well, we will need to consider which partition schemes are most effective for dynamic workload balance, as boids potentially cluster in certain areas of the environment or shift position between spartial divisions rapidly. Updating the partition between processors is costly, so we will want to minimize the amount of dynamic load re-balancing we need to perform while still ensuring good load balance.


===

The problem is challenging because we will need to devise a way to achieve good load balancing across processors. In other words, we will explore optimal ways to distribute work between threads, such that there will be generally equal computation between them. This is challenging because the boids are changing with each time step, and some may have more neighbors than others, and thus more computation than others. Mapping the workload to the current group state is not constant, and will shift as the group moves. 

Another challenge is minimizing communcation and synchronization costs; each boid depends on its neighbors, and so the neighbors' locations will have to be accessed by the current boid to allow for the detection of neighborhood relationships as shown in the image above. There are data dependencies across boids when searching for the next location, because boids share the same physical space. There is data locality as each boid depends only on the general neighbors around it; however, the range of neighbor distance that is covered will need to be tuned. 

We plan to compare the shared address space model with the message passing model using MPI, OpenMP, and Cilk. We also plan to determine and tune the tradeoff between better global movement, that follow the movement rules, versus better speedup and less communication overhead. We will determine how to evaluate the quality of a boid's move, how to evaluate speedup, as well as clearly define the evaluation metrics of the "goodness" of the solution. 


### RESOURCES

Resources that we will use include the following papers, which describes the idea of boids and existing attempts at parallelization:

1. [Flocks, Herds, and Schools: A Distributed Behavioral Model](https://dl.acm.org/doi/pdf/10.1145/37402.37406)
2. [Autonomous Boids](https://onlinelibrary.wiley.com/doi/10.1002/cav.123)
3. [Parallel Bird Flocking Simulation](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.48.8430&rep=rep1&type=pdf)
4. [Parallel simulation of group behaviors](https://ieeexplore.ieee.org/abstract/document/1371337)

We plan to write our own boid simulator code, and we will test on the GHC machines, which have NVIDIA GeForce RTX 2080 B GPUs. 

The main resource which we currently lack is a visualizer for our boid algorithm implementations. While we have a team member with existing experience with the Unity engine, we are uncertain what the best way for visualizing the output of our algorithm is yet.

===

We will also learn from various sources (not yet determined) about building a boid simulator, the boid movement constraints and how to represent them in code, and how to construct a evaluation metric for determining the "goodness" of the solution.


### GOALS AND DELIVERABLES

First, we plan to create a sequential implementation of the boids algorithm. Following that, we plan to create a CPU-based parallel implementation using OpenMP, to create a baseline parallel implementation. Finally, we plan to create a GPU-based parallel implementation using CUDA, where we will attempt to maximize our speedup of the algorithm. In creating our CUDA implementation, we plan to explore various strategies for work partitioning and data structures for improving the efficiency of finding neighbors. We are hoping to answer the question of which load balancing strategies and data structures benefit this specfici use case the most.

If things go well, we hope to create an animated visual representation of the boid simulation to show the result of the parallel computation, and to achieve the speedup required for real-time simulation of the visuals.

If things move slowly, we may only have one or two parallel version of the boid algorithm, and will instead focus on analyzing what challenges and bottlenecks prevented us from being able to achieve better speedup, without animated visualization.

Our demo will contain graphs of speedup over various inputs, as well as graphs comparing the speedup over different parallel versions. As mentioned, we will hopefully run a visual demonstration of the boid movement, which will most likely be created with Unity. 

====

We plan to achieve to have a sequential, and then multiple (2-3) parallel versions of boid movement computation, as well as a visual representation of the movement (that may not be parallelized, as it is simply to show the result of the computations). 

We hope to achieve a parallelized, real time version of the visuals, and not just the computations, as this will show visually the difference between the sequential and parallel computation models. 

If things move slowly, we may only have 1 or 2 parallel versions of the sequential code for computing boid movement, as we explored less than we wanted to in terms of various parallel frameworks and communication models.

Our demo will contain graphs of speedup over various inputs, as well as graphs comparing the speedup over different parallel versions. Additionally, we will run a visual demonstration of the boid movement, which will most likely be created with Unity. We are hoping to answer the question of what is the best load balance in this specific use case, as well as what communication model benefits this computation the most. In general, we want to be able to compare the models we have learned in class, for both computation and communication, and explore how they can work together.  


### PLATFORM CHOICE

We will be using the GHC machines for development, testing, and experiments, because it is more accessible and contains CUDA and OpenMP support needed for development.


### SCHEDULE

Week 1: 3/28 - 4/1	  Research about boids and boid movement algorithms

Week 3: 4/3 - 4/8	    Implement sequential/baseline version of boid movement, add OpenMP framework (CPU) as initial parallel implementation. 

4/11	                Milestone Report

Week 3: 4/11 - 4/15	  Parallelize the sequential program using different parallel framework (CUDA/GPU). 

Week 4: 4/18 - 4/22	  Optimize the parallel implementation, conduct experiments and analysis on different parallel frameworks and communication models

Week 5: 4/25 - 4/29	  Write project report and prepare for poster session

4/29                  Final Report

5/5                   Poster Session

