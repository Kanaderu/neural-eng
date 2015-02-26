# Miscellaneous Lectures
## Symbols and Symbol-like Representation in Neurons

This lecture looks into how we can simulate neurobiological systems that can represent, understand and manipulate symbols.

### Notes

Humans are very good at processing structure of abstract relations. Studies done on apes that have been thoroughly trained to communicate with symbols and sign language, have shown that when attempting to understand analogies (ex. scissors and paper, knife and orange) they tend to relate perceptual qualities like colour.

Human language has structure (ex. Subj-Verb-Obj) and can become recursive. Is it just too complicated currently to simulate?

#### Traditional Cognitive Science

Newell and Simon made a 'General Problem Solver' that manipulate representations like this:
  * after(eight, nine)
  * chase(dog, cat)
  * knows(Anne, thinks(Bill, likes(Charlie, Dave)))

Their approach and further approaches seem to match well to behaviour, but how is this done in neuron-land?

Jackendoff (2002) presented four linguistic challenges for cognitive neuroscience:
1. The binding problem
  * Ventral stream versus dorsal stream hold the 'what' and the 'where', respectively
  * Quick flash of blue and red squares and circles; which shape was the red object?
    * People get it wrong
    * The stimuli seems to be getting 'bound' to particular symbols but cannot bind in time if stimuli flashed too quickly
2. The problem of 2
  * Sort of a repeat of binding
  * "The little star's beside the big star"
  * How do we keep these instances of 'star' apart from each other?
3. The problem of variables
  * Words are grouped into nouns, verbs, etc., so how do you make use of it?
4. Working memory versus long-term memory
  * How do we transfer information between these two memory structures?
  * It seems like moving brain activity and storing them as weights

#### Possible Solutions

1. Oscillations
  * (?)
2. Neural Blackboard Architecture (General-purpose symbol-binding)
  * But this does not mirror how the brain does this
3. Vector Operators
  * Tensor product (outer product)
  * Problem with scaling (makes bigger space N^2)

#### Semantic Pointers

But we need a way to keep the space the same size. Tensor products were on the right track, but we need to compress that space into the same vector size for a single binding. This can be done with circular convolution, which does not increase the size of the space. The pointers are called "semantic" because circular convolution operator is good for language and symbol processing, and the content of the pointers is derived (by compressing it) from semantically related representations. Image processing has different operators that handle vision information more effectively.

Every time you bind, you get rid of some information, but as long as you can unbind, you can use this as an effective system.

### Questions

1. How do humans represent symbols?
2. What is the 'oscillation' solution?

### Miscellaneous

* Symbolists brushed off 'implementation problem' of making models accurate to how the brain functions, which Connectionists were actually concerned with
* Jackendoff (linguist) thought that audio parsing went through phoneme deconstruction, then re-binding to collect into a word, then bind into sentence structure, then grammatical structure, etc.
  * 'Massiveness' of the binding problem
