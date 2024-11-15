# Simulating a Roller Coaster
#### Pranav Rathod

## Overview
In this assignment, I used Catmull-Rom splines along with OpenGL core profile shader-based lighting and texture mapping to create a roller coaster simulation. I implemented two shaders: one for texture mapping (to render the ground), and one for Phong shading (to render the roller coaster rail). The simulation will run in a first-person view, allowing the user to "ride" the coaster in an immersive environment. I also created an interesting animation after finishing the program itself.


## Setup

### Operating System
This assignment was completed on Apple Silicon.

### Additional Setup Instructions
For detailed setup instructions, please refer to the [assignment page](https://viterbi-web.usc.edu/~jbarbic/cs420-s24/assign1/index.html).


The initial code from the hw2-starter.cpp was plugged into a copy of HW1 - hw1.cpp and which was modified. 
To run the program please open the sub-directory 'hw1' and run ./h1 splines/<filename>.sp



## Features and Controls
The program does not use any keyboard or mouse input for Rotate, Scale, Translate

- 'x' : Take a screenshot
- esc : Terminate Program / Close Display Window 

## Extra Credit.
- No Extra credit implemented as of yet.

## For the grader
- Please Find the Screenshots for the animation requirement in the AnimationFrames folder.

Due to the way I have computed tangents the Initial V0 using the sloan method, the following have a proper camera orientation and movement : 
- goodRide.sp
- rollerCoaster.sp
- xaxis.sp
- yaxis.sp

for the files circle.sp, lefturn.sp 'arbitraryVector' in the computeUpVector() function - needs to be changed to [0,0,-1] to see proper camera movement and orientation.
# roller-coaster-splines
