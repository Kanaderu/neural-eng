# Week 4
## Temporal Representation

TODO: Summary

### Notes

Temporal filtering

Receptors:
* NMDA: Tau = 50-100 ms
* AMPA: Tau = 5 ms (?)

Why use spikes if it introduces noise? (fluctuations in encoding)
* Sending a continuous signal over a longer distance is metabolically expensive due to decay
* Sending spikes and boosting them if they're detected at the end is cheaper
* Small animals (i.e. insects) don't use spikes
* Larger animals use spikes

This lesson and previous lessons are vital as fundamentals in understanding the next lessons.

## Transformation

TODO: Summary

### Notes

Time to start connecting neurons together.

If you want to find decoders for transformation, solve least-squares where x is f(x) (when calculating upsilon). This means we can change our code in one place and it will produce a transformation.

Addition is "free" in neuron populations because current + current gives you double current. Current adds, so neurons don't have to do anything special. Neurons default at zero current and don't have a starting place, so when they start, they have to build up current to get to the position where the current is driving.

Neurons have distortion error, but also there is a noise error associated with spiking.

How do we do multiplication? There would be a nonlinearity in the dendrite. But there is a way around it using a "hidden layer" of an extra neuron population. Combining the two inputs into thie layer allows a subsequent nonlinear transform to be computed onto the final population. Multiplication is useful in gating of signals, where unimportant stimuli are attenuated and important stimuli are magnified.

Instead of randomly distributing encoders around unit space, you can optimize for nonlinear transformations. For example, if x and y change with each other, the encoders would cluster more at 45 degrees in the space.

### Questions

1. What is the cell soma?
2. What is the optimal versus temporal filtering?
3. What is filtering in the neural sense? (i.e. in this context)
4. What is population and temporal coding?
5. What is transformation?
    * Input stimuli -> neuron population -> neuron population -> transformed result
    * Connection between neuron populations represents a transformation
    * Ex. y = 2x ~ 2xhat
6. What is the interpretation of the weight matrices w_ij?
7. How does the multiplication work?
8. What is binding?
    * Binding of stimuli together
    * Ex. colour and object shape

### Miscellaneous


