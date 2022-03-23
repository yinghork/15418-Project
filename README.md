# 15418-Project


### TITLE
Optimizing Parallelization of Boid Simulation
Claire Ko (yinghork), Haley Carter (hcarter)

### SUMMARY
We are going to implement an optimized Boid Movement Calculation on the NVIDIA GPUs in the lab. We will use various work distribution strategies and communication mechanisms to create a more efficient simulation of many objects. 

### BACKGROUND

The combined motion of a group, called "boids", is common in nature. Thus, simulating the behaviors of a large number of moving objects is important in fields such as virtual reality and computer animation, where each group member interacts with its neighbors. When simulating the group, each individual boid will compute its position in the next time step, without colliding with its neighbors and attempting to move towards a global goal or attractor. The general group behavior is defined from the individual boid movement and the (possibly changing) location of the global goals.

![image](https://user-images.githubusercontent.com/56246022/159631377-de96b45e-c626-4197-9c08-5b721b90ea6c.png)

Current solutions to the boid simulation problem are not very scalable, in terms of the number of boids, because computation cost as well as the communication costs increase. Our goal is to improve on the current solutions, to build a parallel program to simulate a smooth, generally correct movement for each boid, but not to build a good parallel visual interface for our computations. 

Aspects of the problem that might benefit from parallelism include computing the next best location to move for each boid, and efficient communication between the boids. As there are many boids, it will be essential to parallelize the computing process for every boid, where parallelization is across boids. Since different boids have data dependencies, such as the fact that one boid should avoid the other boids around it, while moving towards/following a global goal location, and the optimization is carried over all boids, our parallel implementation will require communication, either shared state or message passing, and synchronisation over different processes.


### THE CHALLENGE
THE CHALLENGE: Describe why the problem is challenging. What aspects of the problem might make it difficult to parallelize? In other words, what to you hope to learn by doing the project?
Describe the workload: what are the dependencies, what are its memory access characteristics? (is there locality? is there a high communication to computation ratio?), is there divergent execution?
Describe constraints: What are the properties of the system that make mapping the workload to it challenging?

The problem is challenging because 

### RESOURCES
RESOURCES: Describe the resources (type of computers, starter code, etc.) you will use. What code base will you start from? Are you starting from scratch or using an existing piece of code? Is there a book or paper that you are using as a reference (if so, provide a citation)? Are there any other resources you need, but havenâ€™t figured out how to obtain yet? Could you benefit from access to any special machines?

### GOALS AND DELIVERABLES
GOALS AND DELIVERABLES: Describe the deliverables or goals of your project. This is by far the most important section of the proposal!
Separate your goals into what you PLAN TO ACHIEVE ("100%" -- what you believe you must get done to have a successful project and get the grade you expect) and an extra goal or two ("125%") that you HOPE TO ACHIEVE if the project goes really well and you get ahead of schedule, as well as goals in case the work goes more slowly ("75%"). It may not be possible to state precise performance goals at this time, but we encourage you be as precise as possible. If you do state a goal, give some justification of why you think you can achieve it. (e.g., I hope to speed up my starter code 10x, because if I did it would run in real-time.)

If applicable, describe the demo you plan to show at the poster session (Will it be an interactive demo? Will you show an output of the program that is really neat? Will you show speedup graphs?). Specifically, what will you show us that will demonstrate you did a good job?
If your project is an analysis project, what are you hoping to learn about the workload or system being studied? What question(s) do you plan to answer in your analysis?
Systems project proposals should describe what the system will be capable of and what performance is hoped to be achieved.


### PLATFORM CHOICE
PLATFORM CHOICE: Describe why the platform (computer and/or language) you have chosen is a good one for your needs. Why does it make sense to use this parallel system for the workload you have chosen?


### SCHEDULE
SCHEDULE: Produce a schedule for your project. Your schedule should have at least one item to do per week. List what you plan to get done each week from now until the parallelism competition in order to meet your project goals. Keep in mind that due to other classes, youâ€™ll have more time to work some weeks than others (work that into the schedule). You will need to re-evaluate your progress at the end of each week and update this schedule accordingly. Note the intermediate checkpoint deadline is April 11th. In your schedule we encourage you to be precise as precise as possible. Itâ€™s often helpful to work backward in time from your deliverables and goals, writing down all the little things youâ€™ll need to do (establish the dependencies).
