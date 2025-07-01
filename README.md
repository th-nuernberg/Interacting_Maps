
# Interacting Maps 

This is a C++ Implementation of the Interacting Maps algorithm.

The algorithm jointly estimates visual quantities like image intensity, optic flow from event data of a DVS

As there is no code provided with the original paper, this implementation was done based on the descriptions in the paper.
The implementation is done on an event basis in contrast to the original implementation which uses accumulated event frames.
The executable creates a folder of output frames for visual inspection, a time series of estimated rotational velocities

The events currently have to be provided with via a text file, called events.txt. Each event consists of a time stamp, a x and y coordinate and a polarity [0,1]



## Acknowledgements

 - M. Cook, L. Gugelmann, F. Jug, C. Krautz and A. Steger, "Interacting maps for fast visual interpretation," The 2011 International Joint Conference on Neural Networks, San Jose, CA, USA, 2011, pp. 770-776, doi: 10.1109/IJCNN.2011.6033299.


## Authors

- [@Daniel Pommer](https://www.github.com/DanielBanana)

