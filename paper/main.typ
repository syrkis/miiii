#import "@preview/unequivocal-ams:0.1.2": ams-article, theorem, proof
#import "@preview/equate:0.2.1": equate // <- for numbering equations

#let run = "2024-09-19_20-12-55"
#show: equate.with(breakable: true, sub-numbering: true)
#set math.equation(numbering: "(1.1)")

#show: ams-article.with(
  title: [Mechanistic Interpretability and Implementability of Irreducible Integer Identifiers],
  authors: (
    (
      name: "Noah Syrkis",
      // department: [Department of Computer Science],
      organization: [University of Copenhagen],
      location: [Copenhagen, Denmark],
      url: "syrkis.com",
    ),
    (
      name: "Anders Søgaard",
      // department: [Department of Computer Science],
      organization: [University of Copenhagen],
      location: [Copenhagen, Denmark],
      url: "anderssoegaard.github.io/",
    ),
  ),
  abstract: [
    An attention-based deep learning model $cal(M)$ is trained to solve tasks related to prime numbers.
    Specifically, $cal(M)$ is trained to predict if a given natural number $n$ is prime and what, if any,
    prime numbers it can be factorized by. The model is then reverse engineered to understand the
    learned algorithms for the tasks for which it generalizes well. Similar to #cite(<nanda2023>, form: "prose"),
    who trained a transformer model to perform modular addition ($a + b$ mod $113$ for all $a, b < 113$),
    focuses on the task "is $a * n^1 + b * n^0$ prime?" for all $a, b < n$. Setting $n$ to 113 yields
    a dataset of size 12,769.
  ],
  bibliography: bibliography("zotero.bib"),
)
// body ///////////////////////////////////////////////////////////////////////

= Introduction

It is well established that deep learning models can function as both archives (overfitting to training data) and algorithms (learning a generalizing rule), with more recent work focusing on the transition between these two modes @power2022, @nanda2023, @conmy2023. This work aims to further understand this dynamic, by training a deep learning model to solve an array of tasks of increasing difficulty. Specifucally, the tasks asks if a given natural number $n_0$ less than $n$ is a multiple of every prime number less than $sqrt(n)$. The reader should note that a natural number $n$ is prime if and only if it is not a multiple of any prime number less than $sqrt(n)$ (Sieve of Eratosthenes).


= Related work

*Generalization and grokking* — #cite(<power2022>, form: "prose") shows generalization can happen #quote(attribution: cite(<power2022>), "[...] well past the point of overfitting"), dubbing the phenomenon "grokking". The phenomenon is now well established @nanda2023, @humayun2024, @wang2024, @conmy2023, @lee2024a. By regarding the series of gradients as a stochastic signal, #cite(<lee2024a>, form: "prose") propose decomposing the signal into two components: a fast-varying overfitting component and a slow-varying generalization component. They then show that amplification of the slow-varying component significantly accelerates grokking substantially (more than fifty-fold in some cases). This is similar to momentum and AdamW, but the authors explain why it is not the same and can be used in alongside AdamW #cite(<lee2024a>, supplement: "p. 8") (I am trying to understand this better).

*Mechanistic interpretability (MI)* — #cite(<nanda2023>, form:"prose") reverse engineers a transformer model trained to generalize to compute solutions to @nanda

$
  y = (x_1 + x_2) mod p, quad forall x_1, x_2 in {
    0, 1, ..., p-1
  }, quad p "is prime"
$<nanda>

The learned algorithm is then reverse engineered using a qualitative approach (probing, plotting, and guessing).
It is discovered that the generalized circuit uses a discrete Fourier transform (rotation in the complex plane) to solve the problem.

#cite(<conmy2023>, form: "prose") further attempts to automate aspects of the mechanistic interpretability work flow.
MI is a relatively new field, and the methods are still being developed.
#cite(<lipton2018>, form: "prose") discusses various definitions of interpretability, including mechanistic interpretability (though they don't call it that).

*Mechanistic _implementability_* — #cite(<weiss2021>, form: "prose") presents the coding language RASP, which incorporates the architectural constraints of the transformer model into the language itself.
This forces the programmer to be "thinking like a transformer" (which is the title of their paper).
The multilayer perception (MLP) can be thought of as performing a map, (applying a function to every element of a set.

*Primality detection* — Multiple papers describe the use of deep learning to detect prime numbers @egri2006, @lee2024, @wu2023a.
None are particularly promising as prime detection algorithms, as they do not provide speedups, use more memory, or are less accurate than traditional methods.
However, in exploring the foundations of deep learning, the task of prime detection is interesting, as it is a simple task that is difficult to learn, and is synthetic, meaning that the arbitrary amounts of data are generated by a simple algorithm.

*Transformers* — Various modifications/simplifications have been made to the transformer block @he2023, @hosseini2024.
Transformers combine self-attention (a communication mechanism) with feed-forward layers (a computation mechanism).
Importantly, transformers tend to rely on residual streams (I will elaborate).
I am currently using the original transformer block, but I want to switch to @he2023's block, as it is simpler and more interpretable—but there is not much research on it yet.


= Preliminaries

$cal(M)$ is an attention-based deep learning model.
The set of all primes is referred to as $PP$.
A number referred to as $p$ is in $PP$.
A number referred to as $n$ is in $NN$.
The dataset, $cal(D)$, is a set of pairs $[X|Y]$.
$x in X$ and $y in Y$ are elements of the dataset.
$X$ is a matrix of representations of $n in NN < |cal(D)|$.
$Y$ is a one-hot tensor indicating $x in PP$, and which primes $n$ can be factorized by.
There are about $n/ln(n)$ primes less than $n$.
To test if a given number $n$ is prime, it is sufficient to test if it is divisible by any of the prime numbers less than $sqrt(n)$ (Sieve of Eratosthenes) of which there are about $sqrt(n)/ln(sqrt(n))$.
The task is referred to as $cal(T)$, with a subscript indicating the task number.


= Methods

Like #cite(<nanda2023>, form: "prose"), $cal(D)$ is constructed from the first 12 769 natural numbers ($113^2$).
113 was chosen as it allows for a small dataset that can fit in memory, while still being large enough to be interesting.
Indeed, their model was able to generalize modular addition to the held-out test set.


$
  y_i = (
    x_0 dot p^0 + x_1 dot p^1
  ) mod t_i, quad forall t_i < p "where" t_i "is prime",
$<miiii>

While #cite(<nanda2023>, form: "prose") follows the template $a + b mod 113 = x$ for all $a, b < 113$,
this paper, when using base `113` numbers, follows the template $a times 113 + b times 1 in PP$ for all $a, b < 113$.
The choice of $113$ as a base (when n is $113^2$) ensures that the structure of our dataset is similar to that of #cite(<nanda2023>, form: "prose").
Using the cube-root one could extend the task to include a $c * 113^2$-term. However, as the cartestian product of 113 and 113
lends itself much better to visualization (than the 113, 113, 113 cube), we choose to stick with the square, usoing base-113 numbers,
in the initial phase of our experiment.

#figure(
  image("figs/attention_one.svg"),
  caption: [Attenion from digit $b$ to itself in the first head of the first layer for all ($a$, $b$)-pairs.],
)<atten_weight>

In @atten_weight we see a vertical peroiodic pattern, which is expected, as the model is trained to predict the prime factorization of the number $a * 113 + b$.
#lorem(140)


#figure(
  image("figs/attention_layer_0.svg"),
  caption: "Attenion from digit a to digit b",
)

#figure(
  image("figs/attention_layer_1.svg"),
  caption: "Attenion from digit a to digit b",
)

The model, $cal(M)$, is trained to predict if a given natural number $n$ is prime ($cal(T)_1$) and what primes it can be factorized by if it is not prime ($cal(T)_2)$. $cal(T)_1$ is strictly harder than $cal(T)_2$, as $cal(T)_1$ is a binary classification indicating failure to factorize by all primes tested for in Task 2. A Task 3, predicting the remainder of the division of $n$ by the prime it is attempted to factorize by, is also defined, but not used in this paper.

Architectural decisions are made to align with #cite(<lee2024a>, form: "prose") and #cite(<nanda2023>, form: "prose").
The model is trained using the AdamW optimizer with a learning rate of $10^(-3)$,
and weight decay of $1.0$. Dropout is used with a rate of $0.5$.
A hidden size of 128, a batch size of $|cal(D)|$, and a maximum of 6 000 epochs.
GELU activation is used. Each attention layer has 4 heads (32 dimensions per head).
The MLP layers map the input from 128 to 512 to 128.
Layer normalization is also used.
The gradients were modified in accordance with the method described by #cite(<lee2024a>, form: "prose"),
to speed up generalization.
3 transformer layers are used. The model is trained on a single Apple M3 Metal GPU with JAX @jax2018github.
Optax @deepmind2020jax was used for the optimizer.

$cal(T)$ is harder to solve than the modular addition task by #cite(<nanda2023>, form: "prose"),
as it consists of multiple modular multiplication sub-tasks shown in @task_1.

$
  forall p < sqrt((|X|)/ln(|X|)) and n!=p [n mod p != 0]
$ <task_1>

$T_1$ is being strictly harder than $T_2$,
might merit and increase in the number of layers, heads, and hidden size,
which I am currently investigating (update for Anders).

#figure(
  image("figs/polar_nats_and_sixes.svg"),
  caption: [
    $NN < 2^(10)$, $2 NN < 2^(10)$, and $(3NN union 6NN) < 2^(10)$ in polar coordinates.
  ],
)<nats>

The rotational inclinations of the transformer model shown by #cite(<nanda2023>, form: "prose") motivate the use of a polar coordinate system to visualize the distribution of primes. Each prime number $p$ is mapped to the point $(p, p mod tau)$ in polar coordinates, as seen in @primes and @nats.

#figure(
  image("figs/exploration/polar_primes.svg"),
  caption: "The first 2048 primes minus the first 1024 primes.",
)<primes>

One could imagine tightening and loosening the spiral by multiplying $tau$ by a constant, to align multiples of a given number in a straight line (imagining this is encouraged).

The reader is asked to recall that the Sieve of Eratosthenes is an algorithm for finding all prime numbers up to a given limit $n$, by 1) noting that 2 is a prime number, 2) crossing out all multiples of 2, 3) finding the next number that is not crossed out, which is the next prime number, and 4) crossing out all multiples of that number, and so on. Step 2) corresponds to subtracting the center plot in @nats from the left-side plot.

The Cheese Cloth of Eratosthenes is a variant of the Sieve of Eratosthenes,
in which the multiples of the prime numbers are not filtered deterministically,
but rather probabilistically—and inappropriately—by using a deep learning model.

// BELOW HERE IS A MESS

This vector was then converted to the desired number system.
// $Y$ was constructed by first querying all prime numbers less than or equal to $n+1$, creating a one hot vector for each sample, in $X$ indicating primality. $Y$ was further augmented by $sqrt(n)$ vectors, each indicating divisibility by the $i$th prime number up to $sqrt(n)$.
Thus, the sum of all $y$ of primes is 1#footnote[Except for primes less than $sqrt(n)$ as those are both primes and a multiple of themselves (i.e., 2 is a prime number, but it is also a multiple of 2) (note to self: account for this in @task_1).], and the sum of all $y$ of non-primes is $>= 1$.

Note that the row sum of $Y$ can be thought of as a sort of measure of how "close" to being prime a given number is. For example, 20 is very much not a prime since it is a multiple of 2, 4, 5 and 10, while 51 (in base 10) looks like a prime but can, in fact, be factorized into 3 and 17.


$Y$ thus includes information about _why_ a given number is not prime.
The inclusion of these extra tasks also allows for interpretability to be on simpler tasks, by training the model on the simpler tasks first, and then training on the more complex tasks.
This allows for comparison of how the model solves the different tasks, when learning them in isolation versus in conjunction.
For each of the $sqrt(n) + 1$ tasks, the focal loss @lin2018 (@focal_loss) and f1 score are calculated every epoch. Focal loss is cross-entropy weighted by the inverse of the frequency of the positive class ($alpha$) in the task. The loss function can be tweaked to emphasize difficult samples with the $gamma$ parameter.

$
  L_("focal") = -alpha times (1 - p_t)^gamma times log(p_t)
$
<focal_loss>

The frequency of a positive sample in task $i$ is used as the weight for the focal loss during training.
Furthermore, a one-hot vector is used to mask tasks to shield the model from a particular signal during training.

= Results

$cal(M)$ was trained on $cal(D)$ for 6 000 epochs, with a batch size of 12 769.
It generalized to some of the $sqrt(n)$ tasks, but not all.
$cal(M)$ thus exists in a particularly interesting state,
of partial generalization to the tasks in $cal(T)_2$,
As $cal(T)_1$ is strictly harder than $cal(T)_2$, the model has not generalized to $cal(T)_1$.
@train_loss_hinton shows the focal loss of the tasks in $cal(T)$ on the train data during training.
@valid_loss_hinton shows the focal loss of the tasks in $cal(T)$ on the validation data during training.

#figure(
  image("figs/runs/" + run + "/train_loss_hinton.svg"),
  caption: [The focal loss of $cal(T)$ on the train data during training.],
)<train_loss_hinton>

#figure(
  image("figs/runs/" + run + "/valid_loss_hinton.svg"),
  caption: [The focal loss of $cal(T)$ on the validation data during training.],
)<valid_loss_hinton>

@train_loss_hinton shows the focal loss decreasing significantly across tasks in $cal(T)$. Classification of primes $PP$ (i.e. $cal(T)_1$)
starts with a slightly higher loss than other tasks but quickly converges to the same loss as the other tasks, blah blah.
We see in @valid_loss_hinton that we are indeed overfitting in spite of the heavy regularization, as the validation loss is increasing, blah blah.

It is also clear that the sub-tasks in $cal(T)_2$ increase in difficulty
with the $p$ is being tested for. This makes intuitive sense, as it is easier to see if a number is a multiple of 2 than if it is a multiple of 17. There are also more multiples of 2 than 17, though the use of $alpha$ in the focal loss should account for this (should I square and sqrt for alpha?).

#figure(
  image("figs/runs/" + run + "/train_f1_hinton.svg"),
  caption: [The f1 score of $cal(T)$ on the train data during training.],
)<train_f1_hinton>

#figure(
  image("figs/runs/" + run + "/valid_f1_hinton.svg"),
  caption: [The f1 score of $cal(T)$ on the validation data during training.],
)<valid_f1_hinton>

The f1 score of the tasks in $cal(T)$ on the train data during training is shown in @train_f1_hinton, and the f1 score of the tasks in $cal(T)$ on the validation data during training is shown in @valid_f1_hinton.




= Analysis

= Discussion

= Conclusion

// Bibliography
