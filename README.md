# Kidnapped Vehicle Project (Particle Filter)
*Self-Driving Car Engineer Nanodegree Program*
---
[//]: # (Image References)

[image1]: ./screenshot.png "Screenshot of running simulator"

In this project I implemented a particle filter to determine the poise (2D position and yaw) of a simulated moving vehicle, based on noisy LIDAR measurements of known landmarks.
  
The particle filter is a non-parametric Bayes filter. It represents the belief of the state posterior distribution with a random state sample (particles) taken from this posterior. For a complete description of the algorithm, including its mathematical derivation, a good reference is "Probabilistic Robotics", written by Sebastian Thrun, Wolfram Burgard and Dieter Fox, The MIT Press. 

A map is known with a list of landmarks, each with exact x and y coordinates. The simulated vehicle measures the visible landmark positions with a LIDAR. Each measurement is a pair of (noisy) x and y coordinates, that the filter implementation needs to associate with a likely corresponding landmark from the map.
  
My program implements the particle filter and sends the simulator the estimated vehicle position and yaw (heading), along with the list of landmarks currently observed from the vehicle.
      
The simulator displays the received vehicle poise between the landmarks, and the landmark observations as segments originating in the vehicle. The simulator also computes errors in the estimated poise againts a ground truth. See screenshot below.

![Simulator screenshot][image1]

## Dependencies
The program requires Udacity's simulator, which can be downloaded [here](https://github.com/udacity/self-driving-car-sim/releases). 

The following are also required:
* cmake >= v3.5
* make >= v4.1
* gcc/g++ >= v5.4

The program was tested under Ubuntu 16.04.

## Build and Run the Program
Once cloned (or downloaded) the project repository, in the project directory type:

1. `mkdir build`
2. `cd build`
3. `cmake ..`
4. `make`
5. `./UnscentedKF` TODO check!

Run the simulator, make sure **Project 3: Kidnapped Vehicle** is selected in the simulator and click **SELECT**. Click **Start** to have the simulator begin to send data to the running Particle Filter implementation and display its results. 

To stop the Particle Filter program execution hit `Ctrl+C`.

## Inputs to the Particle Filter
You can find a file with the ground truth in the `data` directory. The simulator uses the same data to measure the error of the Particle Filter estimates.

#### The Map
`map_data.txt` includes the position of landmarks (in meters) on an arbitrary Cartesian coordinate system. Each row has three columns
1. x position
2. y position
3. landmark id

*Map data provided by 3D Mapping Solutions GmbH.*
