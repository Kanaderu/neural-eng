# Week 5
## Dynamics

TODO: Summary

### Notes

We must not forget about time. Some networks are connected to themselves, so they can change over time. This can be described as a differential equation.

With feedback:
* more neurons = smoother
* smaller synapse constant (?) = faster slope
* larger radius = higher saturation point

Using a differential equation with these neurons is not the same as 'just' solving a differential equation, as it has different behaviour.

### Questions

1. Why does it take a while for the representation to be driven to input value?
  * Because the receptors were modelled as decaying exponentials (rather than being prescient)

### Miscellaneous

Interesting result from Google machine learning networks. They can classify almost all images from the net at 5% error. However, add noise along some gradient to the image until two different images are classified as the same thing. To the human eye, the images don't look different. So clearly they're doing feature detection differently than humans do.

* Prof. Eliasmith will be gone next week Thursday
