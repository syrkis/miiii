#import "@preview/unequivocal-ams:0.1.1": ams-article, theorem, proof



#show: ams-article.with(
  title: [Mechanistic Interpretability and Implementability of Irreducible Integer Identifiers],
  authors: (
    (
      name: "Noah Syrkis",
      department: [Department of Mathematics],
      organization: [University of South Carolina],
      location: [Columbia, SC 29208],
      email: "howard@math.sc.edu",
      url: "www.math.sc.edu/~howard",
    ),
  ),
  abstract: lorem(100),
  bibliography: bibliography("zotero.bib"),
)


// cover //////////////////////////////////////////////////////////////////////


= Abstract

Deep learning models are increasingly ubiquitous, while their interpretability is inherently opaque.
This paper explores the mechanistic interpretability of deep learning models trained to solve problems
related to prime numbers.
#set heading(numbering: "1.")

// body ///////////////////////////////////////////////////////////////////////

= Introduction

Current state of the art deep learning (DL) models are inherently hard to interpret.
Indeed, DL is subsymbolic in nature, meaning atomic parts, that weights, of the models
do not in and ofthemselves convey any meaning. @to_be_trained_or_not_to_be_trained shows
a Hinton diagram of a trained and un trained embedding layer.

#figure(
  image("figs/latest/train_loss_hinton.svg"),
  caption: "Initial (left) and trained (right) Hinton plot positional embeddings",
)<to_be_trained_or_not_to_be_trained>

#figure(
  image("figs/latest/valid_loss_hinton.svg"),
  caption: "Initial (left) and trained (right) Hinton plot positional embeddings",
)


- Deep learning is sub-symbolic.
- Interpretability is difficult to define @lipton_mythos_2018.
- What is mechanistic interpretability.
- Why are prime numbers a good challenge?

= Related work

*Generalization* — #cite(<power2022>, form: "prose") shows generalization can happen "[...] well past the point of overfitting". This is now well established: @nanda2023, @humayun2024, @wang2024a. By regarding the series of gradients of the gradients as a stochastic signal, #cite(<lee2024b>, form: "prose") decomposes the signal into two components: a fast-varying overfitting component and a slow-varying generalization component. They then show that amplification of the slow varying component significantly accelerates grokking. This is similar to momentum, but different, as the authors describe #cite(<lee2024b>, supplement: "p. 8").

*Mechanistic interpretability* — #cite(<nanda2023>, form:"prose") trains a transformer model to generalize the modular addition task.
The learned algorithm is then reverse engineered using a qualitative approach (probing, plotting, and guessing).
It is discovered that the generalized circuit uses a discrete Fourier transform (rotation in $CC$) to solve the problem.
#cite(<conmy2023>, form: "prose") further attempts to automate aspects of the mechanistic interpretability work flow.

*Mechanistic implementability* — #cite(<weiss2021>, form: "prose") presents the coding language RASP, which incorporates the architectural constraints of the transformer model into the language itself.
This forces the programmer to be "thinking like a transformer" (which is the title of their paper).
The multi layer perception (MLP) can be thought of as performing a map, performing a function on every element of a set. The attention mechanism can be thought of as a reduce (or map-reduce) operation, where the
attention mechanism is a function that takes a set of elements and

*Primality detection* — Multiple papers describe the use of deep learning to detect prime numbers @egri_compositional_2006, @lee2024, @wu_classification_2023.

*Transformers* — Various modifications/simplifications have been made to the transformer block @he2023, @hosseini2024.

= Preliminaries

The set of all primes is referred to as $PP$.
A number referred to as $p$ is in $PP$.
A number referred to as $n$ is in $NN$.
A given number $n$ is in $PP$ if and only if it is not a multiple of any member of ${p | p <= sqrt(n)}$.
The matrix $[X|Y]$ denotes the dataset, with $X$ being representations of natural numbers, and $Y$ indicating primality, along with various other tasks.


= Methods

A deep learning model is trained to predict if a given natural number $n$ is $PP$, what primes it can be factorized by (if any), how close $n$ is to be factorzable by a given prime.

A dataset, $[X|Y]$, of the first 12 769 natural numbers is constructed, with $X$ representing nautral numbers, and $Y$ indicating primarly.

#figure(
  image("figs/polar_nats_and_sixes.svg"),
  caption: "The first 2048 primes minus the first 1024 primes.",
)<prime_numbers_1024_to_2048>

Similarly, prime_numbers_1024_to_2048 shows the first 2048 prime numbers minus the first 1024 prime numbers.
The random nature of the prime numbers is evident in the figure, with no clear pattern emerging.

#figure(
  image("figs/exploration/polar_primes.svg"),
  caption: "The first 2048 primes minus the first 1024 primes.",
)<prime_numbers_1024_to_2048>



The primality of a given number $n$ can be determined by testing if it is divisible by any of the prime numbers
less than $sqrt(n)$. This is the basis of the Sieve of Eratosthenes, which is an ancient algorithm for finding
all prime numbers up to a given limit. The algorithm works by iteratively marking the multiples of each prime number
starting from 2, and then finding the next number that is not marked as a multiple of a prime number, which is the next
prime number. The algorithm is efficient, with a time complexity of $O(n log log n)$. Relating it to our polar
plots, the Sieve of Eratosthenes can be seen as first plotting all natural numbers up to a limit $n$, and then
removing the multiples of the prime numbers less than $sqrt(n)$.


The Zeolite of Eratosthenes is a variant of the Sieve of Eratosthenes,
in which the multiples of the prime numbers are not filtered deterministically,
but rather probabilistically—in inappropriately—by using a deep learning model.

Then I create my own RASP based prime detecting algorithm, bending over backwards to introduce rotational symmetry.


The paper uses a JAX implementation of a two layer transformer model, with a hidden size of 128 and 8 heads,
as per @nanda2023. As prime classification is considerably more complex than modular addition, target vector $y$ rather than being a single one-hot number indicating the primality of a given sample,
is a vector of length $sqrt(n) + 1$, where the $i$ element is 1 if the sample is divisible by the $i$th prime number, and 0 otherwise, with the $sqrt(n)$-th element being 1 if the sample is prime, in which case all other elements are 0.

== Tasks

#figure(
  table(
    columns: (auto, auto, auto),
    table.header[$x$ (base 10)][$x$ (base 2)][\[$y_2 space y_3 space y_p$\]],
    ["04"], ["0100"], [\[1 0 0\]],
    ["05"], ["0101"], [\[0 0 1\]],
    ["06"], ["0110"], [\[1 1 0\]],
    ["07"], ["0111"], [\[0 0 1\]],
    ["08"], ["1000"], [\[1 0 0\]],
    ["09"], ["1001"], [\[0 1 0\]],
  ),
  caption: [Base 10 and 2 dataset for $n = 6$,
    note $sqrt(9) = 3$, so $y$ tests for multiples of 2 and 3, along with primality.],
)<probe-a>

In the following, $X$ denotes the input data, and $Y$ the target data, with $x$ and $y$ denoting individual samples.
$X$ for a dataset of size $n$ was constructed by creating the vector $[2..n+1]$.
This vector was than converted to the desired number system.
$Y$ was constructed by first querying all prime numbers less than or equal to $n+1$, creating a one hot vector for each sample, in $X$ indicating primality. $Y$ was further augmented by $sqrt(n)$ vectors, each indicating divisibility by the $i$th prime number up to $sqrt(n)$.
Thus, the sum of all $y$ of primes is 1, and the sum of all $y$ of non-primes is >= 1 (one if if is divisible by a single other number).

Note that the row sum of $Y$ can be thought of as a sort of measure of how "close" to being prime a given number is. For example, 20 is very much not a prime since it is a multiple of 2, 4, 5 and 10, while 51 (in base 10) looks like a prime but can, in fact, be factorized into 3 and 17.

$Y$ thus includes information about why a given number is not prime.
The inclusion of these extra tasks also allows for interpretability to be on simpler tasks, by training the model on the simpler tasks first, and then training on the more complex tasks.

This allows for comparison of how the model solves the different tasks, when learning them in isolation versus in conjunction.

Primality might be tested on three granularity levels:

1. *primality* — Is $n$ in a prime number?
2. *Factorization* — If $n$ is not a prime number, what primes are its factors.
3. *Modulation* — For every potential prime factorization, why does it fail?


For each of the $sqrt(n) + 1$ tasks, the focal loss, f1, accuracy, and precision are calculated every epoch.
The frequency of a positive samples with in task $i$ is used as the weight for the focal loss during training.
Furthermore, a one-hot vector is used to mask tasks, so as to shield the model from a particular signal during training.

== Model

== Evaluation

= Results

= Analysis

= Discussion

= Conclusion

// Bibliography
