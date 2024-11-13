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
    a dataset of size 12,769. The code for this project is available at github.com/syrkis/miiii.
  ],
  bibliography: bibliography("zotero.bib"),
)
// body ///////////////////////////////////////////////////////////////////////

= Introduction

Recent years have seen deep learning models demonstrate remarkable proficiency in solving complex computational tasks. These models exhibit parallels to information-theoretic concepts, particularly in lossy data compression. For instance, the weights of GPT-2 are about a tenth of the size of its training data, akin to the compression ratios achieved by techniques like Huffman coding (mention lossy technique instead). Importantly, deep learning architectures can function both as archives—overfitting to training data—and as generalized algorithms @power2022.

A system capable of transitioning from archive to algorithm presents intriguing questions: Why doesn't it skip the archiving step and directly learn algorithms? What types of algorithms does it learn, and how reliably? Can the learning process be expedited? How does the presence of multiple tasks affect the learning process? Critically, what specific algorithm has been learned by a given system? Addressing these questions is essential for advancing the theoretical understanding of deep learning and enhancing their practical applications.

In deep learning, however, theory often lags behind practice, limiting our ability to mechanistically explain basic models that have generalized on even relatively simple, synthetically generated tasks. Exploring the mechanics of deep learning models is perhaps more akin to studying biology or botany than traditional computer science. This paper, for example, reverse-engineers a simple transformer model trained to solve modular arithmetic tasks. The simplicity of this training is akin to discovering an intriguing plant in a botanical garden (easy), while understanding its mechanics is akin to dissecting the plant to uncover the principles governing its growth and function (hard).

My investigation probes the fundamental algorithmic structures internalized by a transformer model trained on basic modular arithmetic tasks, with slight variations in complexity. This approach provides insights into how and why specific algorithmic patterns emerge from seemingly straightforward learning processes. @generalization_levels the levels of generalization achieved across these tasks.

#figure(
  image("figs/miiii_f1_tasks_113_20000_last.svg", width: 110%),
  caption: [Final per task F1 scores (note generalization on primes 2, 3, 5 and 7).],
)<generalization_levels>


#figure(
  image("figs/miiii_f1_tasks_113_20000.svg", width: 100%),
  caption: [Evolution of per task F1 scores.],
)<generalization_levels_>


= Related work

This transition between archival and algorithmic modes is smooth under at least some circumstances @nanda2023. Recent research has revealed that amplifying slow-varying gradients can significantly accelerate generalization @lee2024a.

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

My dataset, like that of #cite(<nanda2023>, form: "prose"), has as $X$ the cartestian product of $p$. @miiii_x_11 represents $X$ for $p=11$, varying $x_0$ and $x_1$ through the rows and columns respectively.


#figure(
  image("figs/ds_miiii_11_x.svg"),
  caption: [Representation of $X$, showing all pairs of $(x_0, x_1)$ for $p=11$. Top left shows (base 11) representation of 0, and bottom right represention of 120],
)<miiii_x_11>


$Y$ is constructed by, for each potential prime factor $f_j < p$ we compute the $x_i mod f_j$ . In the simplified case of $p=11$, the potential prime factors are 2, 3, 5, and 7. In that case, $Y$ thus becomes a $11^2 times 4$ (11^2 samples, and 4 tasks for each) matrix, which is shown rearranged to $4 times 11 times 11$ in @miiii_y_11. The most second to top left most value of the left-most plot in the figure, shows $0 dot 11^1 + 0 dot 11^0 mod 2$ to be 0.

$
  y_i = (
    x_0 dot p^0 + x_1 dot p^1
  ) mod f_j, quad forall f_j < p "where" t_j "is prime",
$<miiii>


#figure(
  image("figs/ds_11_y.svg", width: 120%),
  caption: [Representation of our $Y$ for $p = 11$. 2, 3, 5 and 7 and the four primes (and therefore tasks) less than 11.],
)<miiii_y_11>


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
  image("figs/polar.svg"),
  caption: [
    Multiples of 7 or 23 (left), 11 (middle), and primes (right) less than $113^2$ in polar coordinates ($n$, $n$).
  ],
)<nats>

The rotational inclinations of the transformer model shown by #cite(<nanda2023>, form: "prose") motivate the use of a polar coordinate system to visualize the distribution of primes. Each prime number $p$ is mapped to the point $(p, p mod tau)$ in polar coordinates, as seen in and @nats.


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

- sin/cos lookup tables in embedding layer.
- does pos not matter for this task? No, cos it is not comotative. (a + b) mod p = (b + a) mod p -> Nanda. But (a p^1 + b p^0) mod p != (b p^1 + a p^0) mod p.

#figure(
  image("figs/weis_miiii_113_slice_23.svg", width: 110%),
  caption: [Top left $23 times 23$ slice of attention from $y$ to $x_0$ for $p=113$],
)


#figure(
  image("figs/ffwd_miiii_113_4_neurons_slice_23.svg", width: 110%),
  caption: [Top left $23 times 23$ slice of attention from $y$ to $x_0$ for $p=113$],
)


#figure(
  stack(
    dir: ttb,
    spacing: 1em,
    image("figs/miiii_113_U_top_10.svg", width: 110%),
    image("figs/miiii_113_S_top_37.svg", width: 100%),
  ),
  caption: [We see that the 10 most significant vectors of $U$ are periodic!],
)


// #figure(
//   image("figs/miiii_113_W_E.svg"),
//   caption: [The focal loss of $cal(T)$ on the validation data during training.],
// )

#figure(
  image("figs/miiii_113_F_W_E.svg"),
  caption: [The focal loss of $cal(T)$ on the validation data during training.],
)


#figure(
  image("figs/miiii_113_F_neuron_0.svg"),
  caption: [The focal loss of $cal(T)$ on the validation data during training.],
)



shows the focal loss decreasing significantly across tasks in $cal(T)$. Classification of primes $PP$ (i.e. $cal(T)_1$)
starts with a slightly higher loss than other tasks but quickly converges to the same loss as the other tasks, blah blah.
We see in that we are indeed overfitting in spite of the heavy regularization, as the validation loss is increasing, blah blah.

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
