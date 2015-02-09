# Week 6
## Decoder Analysis

What kind of functions can we represent with neurons, what are the differences in error with relation to neuron tuning configuration, and how can we choose the best neuron tuning for a particular function to represent?

### Notes

Increasing exponents x^n with random neurons:
* As we increase the exponent, the error increases
* High exponents have sharper edges, which are hard to represent

Rotating bases:
* Represent same point with rotated bases by using the projection of the point onto the new basis as the scaling factor for each orthonormal coordinate

The tuning curves are like a set of overcomplete basis functions for the space they represent.
    * The space they represent is the space you can compute well from that population

With a population of neurons firing together, the resulting combinations of firing rates does not span the entire neuron space and thus has lower dimensionality. Singular Value Decomposition (SVD) computes the pseudo-inverse of gamma matrix (Gram matrix) (A^T*A), which also tells us the orthonormal basis that most compactly represents the firing rates A.

### Questions

1. What do orthonormal bases have to do with decoding?
    * Can be used for representation with less noise
    * Apparently Fourier transform is an orthonormal basis and is compact
2. In what way specifically is it useful to use overcomplete bases for calculating decoders?
    * Apparently we did this unknowingly in the first assignment
    * Apparently Georgopolous did this as well (which inspired the assignment question)
3. What is a biorthogonal space?
4. What is a basis function?
    * Fourier basis uses sines and cosines to form an orthonormal basis
5. What is a Legendre Polynomial?

### Miscellaneous

* Gaussian relationship between time and frequency domains:
    * As a Gaussian sharpens to a delta function in the time domain, the frequency domain representation becomes flat
    * The time domain is "asking" for higher frequencies as it gets sharper
* If there is some noise or error in basis "measurements", then having an overcomplete basis (dimension < size(basis)) can be useful
    * For example image processing (But what specifically?)
