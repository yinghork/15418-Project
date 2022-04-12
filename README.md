# 15418-Project


### TITLE
Optimizing Parallelization of Boid Simulation

Claire Ko (yinghork), Haley Carter (hcarter)


### SUMMARY
We are going to implement an optimized Boid Movement Calculation on the NVIDIA GPUs in the lab. We will use various work distribution strategies and communication mechanisms to create a more efficient simulation of many objects. 


### BACKGROUND


The "boids" algorithm seeks to simulate the natural flocking motion of animals, primarily birds. Simulating the behaviors of a large number of moving agents interacting with one another is important in fields such as virtual reality and computer animation. The boids algorithm also has much in common with other n-body and particle system simulations, and has therefore seen use in domains related to those problems, such as weather simulation and data visualization. Rather than control the motion of the flock globally, the algorithm creates emergent flocking behavior through a few simple behaviors performed by each boid, based on the limited information within its range of vision. When simulating the flock at each time step, each individual boid updates its position and velocity towards some goal based on three forces. The image below describes the three rules for boid movement, in relation to the neighbors within the limited vision of each individual boid:


![image](https://user-images.githubusercontent.com/56246022/159631377-de96b45e-c626-4197-9c08-5b721b90ea6c.png)


The general structure of the algorithm is as follows: for each time step, for each boid, find the boid's neighbors. For each neighbor, calculate its effect on separation, alignment, and cohesion forces. Calculate the goal-seeking force. Finally, combine those forces to calculate the new velocity and position.

While each boid is dependent on its neighbors, its limited range of vision means that its quantity of neighbors remains relatively constant as the number of boids in the flock increases. Also, the force calculations based on neighbors, and then updating position and velocity, are the same for each boid. This ensures that the work performed by each boid in each time step is relatively similar, meaning that the update of each boid will benefit from a parallel implementation. Additionally, with proper workload division, a parallel application could benefit highly from locality, as boids near each other will also have similar sets of neighbors.

As we will discuss further below, a major aspect of the boids algorithm that might benefit from parallelism comes from determining the neighbors of each boid. The naive approach has O(n<sup>2</sup>) complexity in the number of boids in the flock, which means that the sequential implementation is not very scalable in the number of boids. While this is a challenge for implementation, any potential speedup from a parallel implementation is highly useful. Reducing the computation time of large scale simulation to real time would allow for significantly more use of the algorithm, such as real-time animation in video games.

Our goal is to improve on the current boid algorithms, to build a parallel program with optimized load balancing to simulate a smooth, generally correct movement for each boid. Since different boids also have data dependencies, such as the fact that one boid should avoid the other boids around it, while moving towards/following a global goal location, and the optimization is carried over all boids, our parallel implementation will require communication, either shared state or message passing, and synchronization over different processes.


### THE CHALLENGE

As described above, the main challenge of the algorithm comes in neighbor determination, and managing the corresponding synchronization costs of communicating neighbor locations and movement. Each boid is depend on its neighbors in the flock, but due to the fact that all boids are constantly updating their positions, the neighbors of a boid will change over time. This necessitates updating the set of neighbors for each boid, which naively has O(n<sup>2</sup>) complexity. Therefore, we will need to explore more efficient approaches to finding the neighborhood of each boid. 

Additionally, the changing nature of the system combined with the dependence on nearby neighbors provides another challenge for workload distribution, as there could be possibly divergent execution due to varying number of neighbors. However, there is opportunity for optimization as there is high locality between boids near one another in each particular time step, which we could take advantage of with our workload distribution. We will need to consider which partition schemes are most effective for dynamic workload balance, as boids potentially cluster in certain areas of the environment or shift position between spatial divisions rapidly. Updating the partition between processors is costly, so we will want to minimize the amount of dynamic load re-balancing we need to perform while still ensuring good load balance. We will also need to consider the amount of communication required between processors, given the workload distribution.

We plan to compare the CUDA model using CPU, with the OpenMP model using GPU. We also plan to determine and tune the tradeoff between better global movement, that accurately follow the movement rules, versus better speedup and less communication overhead. We will determine how to evaluate the quality of a boid's move, how to evaluate speedup, as well as clearly define the evaluation metrics of the "goodness" of the solution.


### RESOURCES

Resources that we will use include the following papers, which describes the idea of boids and existing attempts at parallelization:

1. [Flocks, Herds, and Schools: A Distributed Behavioral Model](https://dl.acm.org/doi/pdf/10.1145/37402.37406)
2. [Autonomous Boids](https://onlinelibrary.wiley.com/doi/10.1002/cav.123)
3. [Parallel Bird Flocking Simulation](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.48.8430&rep=rep1&type=pdf)
4. [Parallel simulation of group behaviors](https://ieeexplore.ieee.org/abstract/document/1371337)


We plan to write our own boid simulator code, and we will test on the GHC machines, which have NVIDIA GeForce RTX 2080 B GPUs. We will also determine how to construct a evaluation metric for determining the "goodness" of the algorithm that we create.

The main resource which we currently lack is a visualizer for our boid algorithm implementations. While we have a team member with existing experience with the Unity engine, we are uncertain what the best way for visualizing the output of our algorithm is yet. 


### GOALS AND DELIVERABLES

First, we plan to create a sequential implementation of the boids algorithm. Following that, we plan to create a CPU-based parallel implementation using OpenMP, to create a baseline parallel implementation. Finally, we plan to create a GPU-based parallel implementation using CUDA, where we will attempt to maximize our speedup of the algorithm. In creating our CUDA implementation, we plan to explore various strategies for work partitioning and data structures for improving the efficiency of finding neighbors. We are hoping to answer the question of which load balancing strategies and data structures benefit this specific use case the most.

If things go well, we hope to create an animated visual representation of the boid simulation to show the result of the parallel computation, and to achieve the speedup required for real-time simulation of the visuals.

If things move slowly, we may only have one parallel version of the boid algorithm, and will instead focus on analyzing what challenges and bottlenecks prevented us from being able to achieve better speedup, without animated visualization.

Our demo will contain graphs of speedup over various inputs, as well as graphs comparing the speedup over different parallel versions. As mentioned, we will hopefully run a visual demonstration of the boid movement, which will most likely be created with Unity. 


### PLATFORM CHOICE

We will be using the GHC machines for development, testing, and experiments, because it is more accessible and contains CUDA and OpenMP support needed for development.


### SCHEDULE

Week 1: 3/28 - 4/1	  Research about boids and boid movement algorithms.


Week 2: 4/3 - 4/8     Implement sequential/baseline version of boid movement, add OpenMP framework (CPU) as initial parallel implementation. 


4/11	                Milestone Report


Week 3: 4/11 - 4/15	  

Haley - Parallelize the current sequential program using different parallel framework (CUDA). 

Claire - Spend half the week to update the sequential program with more fluid boid movement, adding and refining possible rules and parameters. Then around 4/14, work on the parallelization with Haley. 


Week 4: 4/18 - 4/22	  Optimize the parallel implementation, conduct experiments and analysis on communication models.

Haley - Experiment with different frameworks for optimizing neighbor determination/communication. 
                      
Claire - Experiment with different work distributions and the parameters associated with boid movement.


Week 5: 4/25 - 4/29	  Create visualization, write project report and prepare for poster session.

Haley - First half: Create a visualization using OpenGL/GLUT. 

Haley - Second half: Write project report and work on poster.

Claire - First half: Run final tests and collect information for report/poster.

Claire - Second half: Write project report and work on poster.


4/29                  Final Report


5/5                   Poster Session



### MILESTONE

We wrote a sequential version of the boid algorithm, complete with flock centering, neighbor collision avoidance, velocity matching, leader following, and box bound following rules. We apply these rules to each boid, to compute the next location of each boid. We store these locations at each time step, or frame, in output files of the form "framePositions_i.txt" where i is the time step from 0 to x. We also wrote a python script to parse these output files and render each time step sequentially, to observe the general movement of the boid group over frames. This is a basic implementation of the animation of boid movement, which we are using to judge the quality of the boid algorithm output. We have created a project framework for the CUDA implementation, so that we can complete an initial CUDA implementation by the end of the week.


After meeting with our assigned TA and receiving advice, we re-assessed our original goals; rather than creating both an OpenMP and CUDA implementation, we will only be focusing on our CUDA implementation. Beyond that, our planned deliverables for the CUDA implementation are still the same. If things go well, we hope to create a better visualization/animation representation of the boid simulation, otherwise we will use our current base rendering. If things move slowly, we may have a CUDA version with minimal speedup, and will instead focus on analyzing what challenges and bottlenecks prevented us from being able to achieve better speedup, without animated visualization.


For the demo and final report, we plan to show speedup graphs to demonstrate the improved performance of our CUDA implementation, focusing on reducing computation time for a fixed problem size. We will also have a demo animation which represents the output of our algorithm. If time permits, we will expand on this animation, but otherwise, we will stay with our existing python script, which renders each frame from the output files generated by our algorithm.


As of now, we do not have any major concerns about the project. It is a matter of executing our plan, as we have the resources needed to do so. 


