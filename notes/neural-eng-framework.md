# Neural Engineering: Computation, Representation, and Dynamics in Neurobiological Systems
#### **Authors:** *Chris Eliasmith and Charles H. Anderson*
## Personal Notes

I would like to get as much as I can out of this book; however given my current set of commitments, a thorough read is not possible. I will be skimming through and taking notes of what interests me. In particular, I have a project for the course Professor Eliasmith teaches in which I will be modelling the locomotion and control of a lamprey in a 2-Dimensional plane. From this book I would like to understand the model discussed and understand the requisite knowledge for the implementation and extension of this model.

## Contents

1. Of Neurons and Engineers
2. Representation of Populations of Neurons
3. Extending Population Representation
4. [Chapter 4: Temporal Representation in Spiking Neurons][]
  i. [Section 4.3 Decoding Neural Spikes][]
5. Population-Temporal Representation
6. Feed-Forward Transformations
7. Analyzing Representation and Transformation
8. [Chater 8: Dynamic Transformations][]
  i. [Section 8.1: Control Theory and Neural Models][]
  ii. [Section 8.5: Lamprey Locomotion][]
9. Statistical Inference and Learning

## Chapter 4: Temporal Representation in Spiking Neurons

This chapter covers the development of a spiking version of the leaky integrate-and-fire (LIF) neuron model. Also covered is how a signal may be decoded given its spiking nature. Mean square error is used again to find optimal linear decoders, so it applies to both population and temporal decoding.

### Section 4.3: Decoding Neural Spikes

#### Notes

Consider neurons in pairs like the push-pull amplifiers of the 1920s. It is beneficial to use two signals to overlap in order to compensate for the shortcomings of both. In the brain, 'on' and 'off' neurons are observed; for example, retinal ganglion cells exhibit lateral inhibition. Another example is to increase the range of stimuli detection by using two neurons, one which fires faster with increasing stimulus intensity, and one which fires faster with decreasing stimuli intensity. This way the average firing rate is constant, and the range is doubled. Signal-to-noise ratio is increased because a small change in stimulus intensity results in a large change in firing rate.

It is assumed that the decoding is linear, such that the spike train can be decoded by taking the convolution of the spike train (sum delta(t - t_n)) and the decoding function (h(t)). This is simply the sum of the decoding function weighted by the spike train. Since convolution with a delta function results in the original signal, convolution with a spike train results in a sum of decoding functions centered at the spike locations t_n (x_hat(t) = sum(n) h(t - t_n)).

#### Questions

1. First question?
2. Second question?

#### Miscellaneous

* Push-pull amplifiers use complementary pairs of tubes that were more efficient in all ways than a single element circuit (1920s)

## Chapter 8: Dynamic Transformations

Up to this point, linear and nonlinear transformations of representations have been covered, but we want to look into how these systems perform over time in a dynamical sense. This means looking at temporal neuron responses. Classical control theory used a behaviourism approach of looking at inputs and ouputs of a system which characterized its goal-oriented behaviour in the context of closed-loop feedback systems. This in conjunction with neurobiology spawned a branch of study called cybernetics. Modern control theory arose due to the fact that one needs to look inside the system in order to understand and design. So we can use modern control theory as a tool for making dynamics work in simulated neurobiological systems.

### Section 8.1: Control Theory and Neural Models

I need to understand the standard control theory from this section, as it is applied in the lamprey swimming model in section 8.5.

#### Notes

Control theory is summarized with these state equations describing the internal description of a linear system (I have not figured out how to do equations in markdown yet!):

dx(t)/dt = Ax(t) + Bu(t)
y(t) = Cx(t) + Du(t)

x - state vector (summarizes all past input and describes state of system)
y - output vector
A - dynamics matrix
B - input matrix
C - output (sensor) matrix
D - feedthrough matrix

In the (neurobiological) framework, we deal only with lumped systems (systems whose state variables are finite). An infinite number of state variables consistutes a distributed system.

The transfer function (output over input) must be a function of time, so must use a neurobiological equivalent which acts over time -- synaptic filter h(t) or soma voltage V(t).

#### Questions

1. What is the soma?
  * The soma is the neuron cell body.
2. What is soma voltage V(t)?
  * Voltage in the cell body.
3. What is the synaptic filter h(t)?
  * It is discussed thoroughly in chapter 4.
  * See chapter 4 notes (and hopefully come back to edit this)

#### Miscellaneous

* First random tidbit
* Another random tidbit

### Section 8.5: Lamprey Locomotion

#### Notes

Lamprey model:
* Isolated spinal cord activates when bathed in excitatory amino acids
  * Spine portions are modeled by *biphasic oscillator* (BPO)
  * Whole spine is chain of BPOs
* Spinal cord is series of *central pattern generators* (CPG)
  * Bottom-up approach

Lamprey characteristics:
1. continuous spinal cord, but made of about 100 segments
2. connectivity is local, but spans several segments
3. connectivity is asymmetric
4. individual neurons code muscle tension over small regions
5. neural firing is observed as counter-phase on either side (i.e. biphasic oscillation)
6. length of lamprey is about 1 period of swimming wave
7. 6 is true regardless of swimming speed
8. lamprey can swim forwards at 1/4 to 10 Hz, and can swim backwards

#### Questions

1. What is a *biphasic oscillator* (BPO)?
  * Two identical sine waves that are 180 degrees out of phase
2. What is a *central pattern generator* (CPG)?
  * Group of neurons that can produce rhythmic patterns without sensory input
  * Lamprey spinal cord is series of CPGs
3. What does a low Reynold's number represent (creeping or Stokes' flow)?
  * Reynold's number:
    * It's defined as the ratio of inertial forces to viscous forces
    * Quantifies relative importance of these two forces for given flow conditions
    * Helps predict similar flow patterns in different flow situations (ex. scaling models)
  * Low Reynold's number (Stokes' flow)
    * Laminar flow
    * Viscous forces are dominant (ex. honey)
    * Smooth, constant fluid motion
    * No turbulence
4. What are orthonormal spatial harmonic functions?
  * Harmonic functions:
    * Twice differentiable function which satisfies Laplace's equation
    * Laplace's equation states that div-grad (Laplacian) of F equals zero
    * Basically, net flux is zero in the region, there are no sources
  * Orthonormal means orthogonal basis vectors are unit length
5. What is the standard control theory from section 8.1?
6. What is LIP?
  * Lateral intraparietal cortex
7. What is the function of the angled brackets in the mathematical modeling?
  * <.>_x indicates integration over the range of x

#### Miscellaneous

* First random tidbit
* Another random tidbit

## Chapter X: Description

#### Notes

Begin the notes

An example equation (does not render in GitHub at the very least):
\begin{equation}
  a^2+b^2=c^2
\end{equation}

#### Questions

1. First question?
2. Second question?

#### Miscellaneous

* First random tidbit
* Another random tidbit
