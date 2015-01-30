# Week 3
## Temporal Representation

* What if stimulus x changes over time?
* Model we have thus far is not accurate
    * It is an average firing rate over a long period of time
* tau_ref is refactory period of neuron
    * Input current has no effect on voltage during this period
* tau_RC is the RC circuit constant of the neuron
* How are these voltages measured in real brains?
    * How is ground represented?
    * Voltage is measured in relation to what?
    * Inside vs outside of cell
    * To drive current, stick an electrode inside cell membrane

* Time constant tells the neuron how much "memory" it has in terms of past current efffect on current voltage (50 ms)
    * To get voltage activity from current activity, convolve current with decaying exponential

* There is variability in neuron spike magnitudes and in width of spike, etc
    * There is no proof that this has an effect on information transmission
    * Therefore we ignore it and use identical spikes

* Thus far our model is a point neuron
    * Missing some of the complexities of the neuron
* There may be non-linearities in the dendrites of neurons
    * Each neuron has on average 10000 inputs
    * Non-linearities may boost signals coming from further cells

### Debate of time versus rate coding

Do neurons use a code based on timing of spikes, or based on the rate of fire?
* Ex. |.....|.|..||...
* Vs. ..|..||...|....|
* Same rate but different timing
* Monkey vision system shows rate causes same behaviour
* Flies react within 10 ms, so they use a timing code
* Use one decoder with a parameter for decoding "slow" (rate) or "fast" (timing)

## Miscellaneous

* In Python, "lambda" is a way to define a function inline

