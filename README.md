# Make a Neural Net Live Demo

## Overview

This is an implementation of a two-layer neural network during the live demo by @Sirajology on [Youtube](https://youtu.be/vcZub77WvFA). The training method is stochastic (online) [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) with momentum. It computes XOR for the given input. It uses two activation functions, one for each layer. One is a tanh function and the other is the sigmoid function. It uses [cross-entropy](http://neuralnetworksanddeeplearning.com/chap3.html) as it's loss function. This is all done in less than 100 lines of code. We're building this thing from scratch!

## Dependencies

* Numpy

## Usage

Just run the following in terminal to see it run. 

``
python demo.py
``

## Credits

The credits for the majority of this code go to [lightcaster](https://github.com/lightcaster). I've merely created a wrapper to get people started.
