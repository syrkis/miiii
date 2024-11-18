#import "@preview/unequivocal-ams:0.1.2": ams-article, theorem, proof
#import "@preview/equate:0.2.1": equate // <- for numbering equations

#let f_hash = "7ddd799ee00349b9b94acd5d"
#let p_hash = "7ddd799ee00349b9b94acd5d"
#show: equate.with(breakable: true, sub-numbering: true)
#set math.equation(numbering: "(1.1)", supplement: "Eq.")

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
    This paper investigates the emergence and mechanistic nature of algorithmic learning in transformer models through the lens of modular arithmetic and prime factorization. Building upon recent work in mechanistic interpretability, a transformer model is trained to predict remainders when dividing base-p numbers by prime factors. For numbers represented as $x_0 p^0 + x_1 p^1$ (where $x_0,x_1 < p$), the model must learn distinct strategies for each prime factor, with task difficulty scaling naturally with the size of the factor. Setting $p=113$ yields 29 parallel tasks and 12,769 samples, allowing for the study of how the model develops different computational strategies for tasks of varying complexity. Analysis of learned representations and attention patterns reveals distinct periodicities in the model's internal representations, suggesting the emergence of trigonometric basis functions similar to those found in simpler modular arithmetic tasks. This work contributes to our understanding of how neural networks discover and implement mathematical algorithms, particularly in settings with multiple related tasks of varying complexity.
    #footnote[https://github.com/syrkis/miiii].
  ],
  bibliography: bibliography("zotero.bib"),
)

// body ///////////////////////////////////////////////////////////////////////

= Introduction

Recent years have seen deep learning models demonstrate remarkable proficiency in solving complex computational tasks. These models exhibit parallels to information-theoretic concepts, particularly in lossy data compression. For instance, the weights of GPT-2 are about a tenth of the size of its training data, akin to compression ratios. Importantly, deep learning architectures can function both as archives—overfitting to training data—and as generalized algorithms @power2022.

A system capable of transitioning from archive to algorithm presents intriguing questions: Why doesn't it skip the archiving step and directly learn algorithms? What types of algorithms does it learn, and how reliably? Can the learning process be expedited? How does the presence of multiple tasks affect the learning process? What specific algorithm has been learned by a given system? Can it exist as an archive and an algorithm simultaneously? Addressing these questions is essential for advancing the theoretical understanding of deep learning and enhancing its practical applications.

In deep learning, however, theory often lags behind practice, limiting our ability to mechanistically explain basic models that have generalized on even relatively simple, synthetically generated tasks. Exploring the mechanics of deep learning models is perhaps more akin to studying biology or botany than traditional computer science. This paper, for example, reverse-engineers a simple transformer model trained to solve modular arithmetic tasks. The simplicity of this training can be likened to discovering an intriguing plant in a botanical garden (easy), while understanding its mechanics is akin to dissecting the plant to uncover the principles governing its growth and function (hard).

Prime numbers, in particular, are an interesting domain for deep learning. A frequent feature of number theoretical problems is the ease with which they can be stated. This is true for trivial problems (such as proving there are infinitely many primes) and deceptive problems (such as "all even numbers can be expressed as the sum of two primes"). The latter, known as Goldbach's conjecture, remains unsolved. There are about $n/ln(n)$ primes less than $n$. To test if a given number $n$ is prime, it is sufficient to test if it is divisible by any prime less than $sqrt(n)$ (Sieve of Eratosthenes), of which there are about $sqrt(n)/ln(sqrt(n))$.

However, how exactly a given model implements an algorithm is a non-trivial question—as we shell see, even modular addition is implemented in an obscure way @nanda2023.
This investigation probes the fundamental algorithmic structures internalized by a transformer model trained on a set of basic prime number-related modular arithmetic tasks, with slight variations in complexity. This approach provides insights into how and why specific algorithmic patterns emerge from seemingly straightforward learning processes.

= Related work

*Generalization and grokking* — #cite(<power2022>, form: "prose") shows generalization can happen #quote(attribution: cite(<power2022>), "[...] well past the point of overfitting"), dubbing the phenomenon "grokking". The phenomenon is now well established @nanda2023, @humayun2024, @wang2024, @conmy2023. #cite(<nanda2023>, form: "prose") shows that, unlike some have argued (CITE), the generalized circuits slowly emerge from that start, rather than being a relatively sudden and stochastic encounter. By regarding the series of gradients as a stochastic signal, #cite(<lee2024a>, form: "prose") propose decomposing the signal into two components: a fast-varying overfitting component and a slow-varying generalization component. They show that amplification of the slow-varying component accelerates grokking substantially (more than fifty-fold in some cases). This echoes the idea that generalized circuits go through a sort of embryology, through the entire duration of training, rather than suddenly occur in. To the extent that this phenomenon is wide spread, it bows well for generalizable deep learning, in that the generalizing signal that one would want to amplify exists long before the model is fully trained.


*Mechanistic interpretability (MI)* —
MI is a relatively new field, and the methods are still being developed.
#cite(<lipton2018>, form: "prose") discusses various definitions of interpretability, including mechanistic interpretability (though they don't call it that) in which the mechanisms of the model are reverse engineered. This is on the opposite scale of forms of interpretability such as feature importance, which is a measure of how much a feature contributes to the model's prediction (i.e. a person with a beard is identified as male my an image model because of the beard).
#cite(<nanda2023>, form:"prose") reverse engineers a transformer model trained to generalize to compute solutions to @nanda. This task, referred to here as $cal(T)_a$, is used extensively throughout the paper.

$
  y = (x_0 + x_1) mod p, quad forall x_0, x_1 in {
    0, 1, ..., p-1
  }, quad p = 113
$<nanda>

The learned algorithm is then reverse engineered using a qualitative approach (probing, plotting, and guessing).
It is discovered that the generalized circuit uses a discrete Fourier transform (rotation in the complex plane) to solve the problem. Specifically, the embedding layer learns a lookup tale for the cosine and sine values of the input, while the feed word layer of the transformer block learns to combine these values though multiplication, addition and trigonometric identities. Note that $cal(T)_a$ is commutative:

$
  (x_0 + x_1) mod p = (x_1 + x_0) mod p
$<commutative>

#cite(<conmy2023>, form: "prose") automates aspects of the mechanistic interpretability work flow for specific tasks, namely the circuit discovery. In MI a circuit refers to a path through the model weights that detects a specific pattern. Ablation studies show that the elements of the model not involved in the circuit can be removed without negatively affecting the generalization performance of the model.

*Mechanistic _Implementability_* — #cite(<weiss2021>, form: "prose") presents the coding language RASP, which incorporates the architectural constraints of the transformer model into the language itself.
This forces the programmer to be "thinking like a transformer" (which is the title of their paper).
The multilayer perception (MLP) can be thought of as performing a map, (applying a function to every element of a set, while the attention mechanism is a way to combine (reduce) the information from different layers. A language grammatically constrained by the invariances of the transformer, is perhaps the best way for a human to demystify the architecture. However, the generalized circuits are as implementable in any other language.

*Deep Number Theory* — Multiple papers describe the use of deep learning to detect prime numbers @egri2006, @lee2024, @wu2023a.
None are particularly promising as prime detection algorithms, as they do not provide speedups, use more memory, or are less accurate than traditional methods.
However, in exploring the foundations of deep learning, the task of prime detection is interesting, as it is a simple task that is difficult to learn, and is synthetic, meaning that the arbitrary amounts of data are generated by a simple algorithm.

*Transformers* — Various modifications/simplifications have been made to the transformer block @he2023, @hosseini2024.
Transformers combine self-attention (a communication mechanism) with feed-forward layers (a computation mechanism).
Importantly, transformers tend to rely on residual streams (I will elaborate).
I am currently using the original transformer block, but I want to switch to @he2023's block, as it is simpler and more interpretable—but there is not much research on it yet.


= Methods

The methodology extends #cite(<nanda2023>, form: "prose")'s modular addition task $cal(T)_a$ to the base-$p$ prime factorization focused task $cal(T)_b$. Throughout this paper, $p$ denotes a prime number, $f$ denotes a prime less than $p$, and $x$ denotes a natural number less than $p^2$. For $p=113$, there are 29 primes to be tested. For a given prime $p$, a dataset $[X|Y]$ is constructed where $X$ represents the Cartesian product of digits less than $p$, encoding all numbers of the form $x_0 dot p^0 + x_1 dot p^1$ where $x_0, x_1 < p$. $Y$ indicates which primes $f < p$ divide the number represented by $X$ and what, if any, the remainder is. When $x mod f = 0$, $f$ is a prime factor of $x$. Unlike modular addition, which is commutative, this polynomial representation introduces an asymmetry: $(a x + b) mod p ≠ (b x + a) mod p$. Formally, the task can be expressed as:

$
  y = ( x_0 dot p^0 + x_1 dot p^1 ) mod f, quad forall f < p
$<miiii>

The operation $x mod p$ falls under a cyclic group, meaning that any number $n$ can be written as $n = k p + r$ where $0 <= r < p$ is the remainder and $k$ is some integer. This means that the elements of the group cycle through the possible remainders $0, 1, ..., p-1$ as $n$ increases. Convenient consequences hereof includes that the distribution of the remainders is uniform.

== Tasks

Stated plainly: the task predicts the remainder when dividing a two-digit base-$p$ number by each prime factor less than $p$.
For $p=113$, this yields 29 parallel tasks, one for each prime less than $p$. Each task predicts a remainder in the range $[0, f-1]$. This means smaller primes like 2 and 3 require binary and ternary classification respectively, while the largest prime less than $p$, 109, requires predictions across 109 classes. The tasks thus naturally vary in difficulty: predicting $mod 2$ requires distinguishing odd from even numbers (which in binary amounts to looking at the last bit), while predicting $mod 109$ involves making a selection between many relatively similar classes. From an information-theoretical perspective, the expected cross entropy for an $n$-classed problem is $ln(n)$, which has implications for the construction of the loss function, further discussed in @training.

== Data

*Input Space ($X$)*
Each input $x in X$ represents a number in base $p$ using two digits, $(x_0,x_1)$, where the represented number is $x_0 p^0 + x_1 p^1$. For example, with $p=11$, the input space consists of all pairs $(x_0,x_1)$ where $x_0,x_1 < 11$, representing numbers up to $11^2-1 = 120$. This yields a dataset of 121 samples. @miiii_x_11 visualizes this input space, with each cell representing the value $x_0 p^0 + x_1 p^1$.

#figure(
  image("figs/x_11_plot.svg", width: 110%),
  caption: [Visualization of input space $X$ for $p=11$. Each cell $(x_0,x_1)$ represents the number $x_0 p^0 + x_1 p^1$. The top left shows 0 $(0,0)$, and bottom right shows 120 $(10,10)_{11}$],
)<miiii_x_11>

*Output Space ($Y$)*
For each input $x$, a vector $y in Y$ contains the remainder when dividing by each prime less than $p$. For $p=11$, this means predicting the remainder when dividing by 2, 3, 5, and 7. Each element $y_i$ ranges from $0$ to $f_i-1$ where $f_i$ is the $i$-th prime. @miiii_y_11 visualizes these remainders, with each subplot showing the remainder pattern for a specific prime divisor. For comparison, the rightmost plot shows the output space of @nanda2023's modular addition task.

#figure(
  image("figs/y_11_plot.svg", width: 120%),
  caption: [Output space $Y$ for $p=11$. The first four plots show remainders when dividing by 2, 3, 5, and 7 respectively. The rightmost plot shows the output space of the modular addition task for comparison.],
)<miiii_y_11>

To provide insight into the periodic structure of these remainders (and motivate thinking in rotational terms), @nats visualizes various modular patterns in polar coordinates. The periodic nature of remainders becomes apparent when plotting points $(n, n)$ in polar coordinates, where clustering indicates common remainders.

#figure(
  image("figs/polar.svg"),
  caption: [
    Periodic patterns in polar coordinates $(n, n)$ for numbers less than $12 769$. Left: numbers with remainder 0 mod 7 or 23 (see the two spirals). Middle: numbers with remainder 0 mod 11. Right: prime numbers.
  ],
)<nats>



== Model

The model follows the original transformer architecture @vaswani2017 with several key design choices aligned with recent work on mechanistic interpretability @nanda2023, @lee2024a: biases are disabled, and layer normalization is not used. The model consists of three main components: an embedding layer, transformer blocks, and an output layer. All weights are initialized following #cite(<he2015>, form: "prose").

Input tokens are embedded into a $d$-dimensional space using learned token and positional embeddings:
$
  z = "TokenEmbed"(x) + "PosEmbed"("pos")
$<embed>

Each transformer block comprises multi-head attention:
$
  "Attention"(Q, K, V) = "softmax"(Q K^T / sqrt(d_k))V
$<attn>
where $Q$, $K$, and $V$ are linear projections of the input. Attention heads are combined through addition rather than concatenation. This is followed by a feed-forward network with ReLU activation:
$
  "FFN"(x) = "ReLU"(x W_("in"))W_("out")
$<ffn>
mapping from $d$ → $4d$ → $d$ dimensions. Each component includes residual connections and dropout.

The final representation is projected to predict remainders for each prime factor:
$
  hat(y) = z_(-1) W_("out")
$<output>
where $W_("out")$ projects to $sum_(i=1)^k f_i$ dimensions for $k$ prime factors, with $f_i$ being the $i"th"$ prime less than $p$.














== Training<training>

Hyper-parameter optimization was conducted using Optuna @akiba2019, searching over @hyper_param_search_space.


#figure(
  table(
    columns: (1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: center,
    table.header(
      "dropout",
      $lambda$,
      "wd",
      $d$,
      "lr",
      "heads",
    ),

    $1 / 2, 1 / 5, 1 / 10$,
    $1 / 2, 2$,
    $1 / 10, 1 / 2, 1$,
    $128, 256$,
    "3e-4, 1e-4",
    "4, 8",
  ),
  caption: "Hyperparameter search space for training.",
)<hyper_param_search_space>

The model is trained using AdamW @loshchilov2019 with $beta_1=0.9$, $beta_2=0.98$ following @nanda2023. To handle the varying number of classes across tasks (from 2 classes for mod 2 to 109 classes for mod 109), a weighted cross entropy loss is employed:


$
  L_("ce") = -sum_(i=1)^k alpha_i times log(p_t)
$

// $
// L_("focal") = -alpha times (1 - p_t)^gamma times log(p_t)
// $<focal_loss>

where $alpha_i = 1/ln(f_i)$ accounts for the varying difficulty across tasks with different prime factors $f_i$.

To accelerate generalization, gradient filtering @lee2024a is implemented:

$
  g_t = nabla_theta L + lambda(alpha e_(t-1) + (1-alpha)g_(t-1))
$<grad>

where $e_t$ is the exponential moving average of gradients with decay rate $alpha=0.98$, and $lambda=2$ controls the influence of the slow-varying component.

Training uses full batch gradient descent with the entire dataset of $p^2$ samples. The model is evaluated on a held-out validation set after each epoch, tracking per-task accuracy and loss.

== Visualization

Much of the data worked with here is inherently high dimensional. For training, for example, we have $n$ steps, two splits (train/valid) about $p/ln(p)$ tasks, and two metrics (accuracy, and loss). This, along with the inherent opaqueness of deep learning models, motivated the developed custom visualization library, esch#footnote[https://github.com/syrkis/esch] to visualize attention weights, intermediate representations, training metrics, and more.

== Mechanistic Interpretability

A combination of linear products is itself a linear product. As a mechanistic interpretability rule of thumb, one should look at the outputs of the non-linear transformations. In our case that will be the attention weights, and the intermediate representations with each transformer block's MLP (which follows a ReLU activation).
Additionally, the embeddings layers will be inspected. blah blah.

Our interpretability approach combines visualization techniques with frequency analysis to understand the learned algorithmic patterns. Following @nanda2023, we analyze both the attention patterns and the learned representations through several lenses:

*Attention Visualization*
Using esch, the custom visualization library, to visualize attention weights and intermediate representations. The library allows for the visualization of attention patterns across different layers, as well as the visualization of intermediate representations at each layer. These visualizations provide insights into the learned patterns and help identify potential areas of improvement.

*Fourier Analysis*
To quantify periodic patterns in both attention weights and intermediate representations, we decompose them into their constituent frequencies using the discrete Fourier transform:

$
  X_k = sum_(n=0)^(N-1) x_n e^(-2pi i k n / N)
$<fourier>

This analysis helps identify the dominant frequencies in the model's computational patterns, particularly those corresponding to potential modular arithmetic operations.

*Activation Patterns*
We track how representations evolve through the network by:
- Visualizing activation matrices at each layer
- Computing correlation matrices between different positions and features
- Analyzing the residual stream contributions

These analyses are performed across different input patterns to understand how the model distinguishes between prime and composite numbers.
Note that that in figures with periodicity only a top left most $37 times 37$ slice is shown, so as to not overwhelm the reader. Visualizations are available in the Appendix.

== Evaluation

In @atten_weight we see a vertical periodic pattern, which is expected, as the model is trained to predict the prime factorization of the number $x_0 * 113 + x_1$.



The model, $cal(M)$, is trained to predict if a given natural number $n$ is prime ($cal(T)_1$) and what primes it can be factorized by if it is not prime ($cal(T)_2)$. $cal(T)_1$ is strictly harder than $cal(T)_2$, as $cal(T)_1$ is a binary classification indicating failure to factorize by all primes tested for in Task 2. A Task 3, predicting the remainder of the division of $n$ by the prime it is attempted to factorize by, is also defined, but not used in this paper.

Architectural decisions are made to align with #cite(<lee2024a>, form: "prose") and #cite(<nanda2023>, form: "prose").
The model is trained using the AdamW optimizer with a learning rate of $10^(-3)$,
and weight decay of $1.0$. Dropout is used with a rate of $0.5$.
A hidden size of 128, a batch size of $|cal(D)|$, and a maximum of 6 000 epochs.
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
  image("figs/attention_one.svg"),
  caption: [Attention from digit $b$ to itself in the first head of the first layer for all ($a$, $b$)-pairs.],
)<atten_weight>

#figure(
  image("figs/attention_layer_0.svg"),
  caption: "Attenion from digit a to digit b",
)

#figure(
  image("figs/attention_layer_1.svg"),
  caption: "Attenion from digit a to digit b",
)

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
For each of the $sqrt(n) + 1$ tasks, the focal loss @lin2018 () and f1 score are calculated every epoch. Focal loss is cross-entropy weighted by the inverse of the frequency of the positive class ($alpha$) in the task. The loss function can be tweaked to emphasize difficult samples with the $gamma$ parameter.



The frequency of a positive sample in task $i$ is used as the weight for the focal loss during training.
Furthermore, a one-hot vector is used to mask tasks to shield the model from a particular signal during training.




= Results

The best performing model was trained with the hyper-parameters in @hyper_param_search. As seen in figures @trainig_acc, the model grokked on all 29 tasks, achieving perfect accuracy. Note that tasks 2, 3, 5 and 7 occur in succession (with 5 and 7 swapped), while rest happen, more or less simultaneously, after the initial 4. This could indicate that a more geneal solution has been found, differing in a parameterized way from one another.

#figure(
  table(
    columns: (1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: center,
    table.header(
      "dropout",
      $lambda$,
      "wd",
      $d$,
      "lr",
      "heads",
    ),

    "0.2", "2", "1.0", "256", $3 times 10^(-4)$, "4",
  ),
  caption: [Hyper parameter search for the model.],
)<hyper_param_search>


#figure(
  stack(
    dir: ttb,
    image("figs/" + f_hash + "/acc_train_training.svg"),
    image("figs/" + f_hash + "/acc_valid_training.svg"),
  ),
  caption: [Representation of training and validation accuracy ($x$-axis is in log scale).],
)<trainig_acc>

== Positional embeddings


@pos_emb shows the positional embeddings of the $a$-task to be virtually identical (cosine similarity of 0.95), which is to be expected due to the tasks commutativity. Interestingly, the cosine similarity of the $b$-task is -0.64. Neither, completely opposite, nor orthogonal, as expected. However, it is clear that the two positional embeddings translate the tokens in different directions.

#figure(
  image("figs/pos_emb.svg"),
  caption: [Positional embeddings for the a network trained on @nanda (top) and @miiii (bottom)],
)<pos_emb>


== Token embeddings


@s shows us that the singular values of the the token embeddings learned for task $b$ to be much more diffuse than those for task $a$. As stated, the embedding layer i task $a$ represents a look table for the sine and cosine values of the tokens—hance the periodicity of the most significant singular vectors.

#figure(
  image("figs/S.svg"),
  caption: [Singular values of $U$ for $cal(T)_a$ (top) and $cal(T)_b$ (bottom)],
)<s>

#figure(
  image("figs/f_U.svg"),
  caption: [Most significant singular vectors of $U$ for $cal(T)_b$],
)<f_U>

#figure(
  image("figs/p_U.svg"),
  caption: [Most significant singular vectors of $U$ for $cal(T)_a$],
)<p_U>

Projecting the positional embeddings onto a Fourier basis, however, shows that the periodicity is indeed preserved.

#figure(
  stack(
    dir: ttb,
    image("figs/f_f.svg"),
    image("figs/f_f_norm.svg"),
  ),
  caption: [$W_(E_(cal(T)_b))$ in Fourier space (norm below)],
)<f_f>

The fact of periodicity in @f_f despite the presence of multiple tasks with unique rotational steps around the circle, the non commutative nature of the task, is further @nanda2023 indication that trigonometric tables are a reliably used representation of the architecture.

#figure(
  stack(
    dir: ttb,
    // image("figs/p_f.svg"),
    image("figs/p_f_norm.svg"),
  ),
  caption: [Frequencies of $W_(E_(cal(T)_a))$ in Fourier space],
)<p_f>

== Attention patterns

Unlike that @nanda task, our attention heads focus on one digit or the other. This could be due to the non-commutative nature of the task.


== Feed-forward

// #figure(
//   stack(
//     dir: ttb,
//     image("figs/train_acc.svg", width: 100%),
//     image("figs/valid_acc.svg", width: 100%),
//   ),
//   caption: [Training and validation accuracy],
// )<generalization_levels_>


We see that the positional embeddings are orthogonal. The token embeddings of $x_0$ and $x_1$ are offset, allowing for the cosine and sine table to be learned for both.
NOTE: they might not be orthogonal, but rather pointing in opposite directions (we only have two vectors, so orthogonality is not needed.

#figure(
  image("figs/pos_emb.svg", width: 100%),
  caption: [Positional embeddings for the first $37$ for a model trained on @nanda (top) and @miiii (bottom). The low information contained in the positional encoding of @nanda is to be expected as the task is commutative, while @miiii is not—$(x_0 + x_1) mod p = (x_1 + x_0) mod p$ but $((x_0 p^0 + x_1 p^1) mod p) != ((x_1 p^1 + x_0 p^0) mod p)$..],
)<generalization_levels_>


// #figure(
//   image("figs/miiii_f1_tasks_113_20000.svg", width: 100%),
//   caption: [Evolution of per task F1 scores.],
// )<generalization_levels_>



- sin/cos lookup tables in embedding layer.
- does pos not matter for this task? No, cos it is not commutative. (a + b) mod p = (b + a) mod p -> Nanda. But (a p^1 + b p^0) mod p != (b p^1 + a p^0) mod p.

#figure(
  image("figs/weis_miiii_113_slice_23.svg", width: 110%),
  caption: [Attention from $hat(y)$ to $x_0$ for the four attention heads.],
)


#figure(
  image("figs/ffwd_miiii_113_4_neurons_slice_23.svg", width: 110%),
  caption: [Slice of neurons $0$ to $3$ in the feed forward network for $p=113$],
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


#figure(
  image("figs/miiii_113_F_W_E.svg"),
  caption: [Active frequencies in Fourier space of Embedding],
)


#figure(
  image("figs/miiii_113_F_neuron_0.svg"),
  caption: [Neuron 0 for sample 1 in Fourier space.],
)



shows the focal loss decreasing significantly across tasks in $cal(T)$. Classification of primes $PP$ (i.e. $cal(T)_1$)
starts with a slightly higher loss than other tasks but quickly converges to the same loss as the other tasks, blah blah.
We see in that we are indeed overfitting in spite of the heavy regularization, as the validation loss is increasing, blah blah.

It is also clear that the sub-tasks in $cal(T)_2$ increase in difficulty
with the $p$ is being tested for. This makes intuitive sense, as it is easier to see if a number is a multiple of 2 than if it is a multiple of 17. There are also more multiples of 2 than 17, though the use of $alpha$ in the focal loss should account for this.

// #figure(
//   image("figs/runs/" + run + "/train_f1_hinton.svg"),
//   caption: [The f1 score of $cal(T)$ on the train data during training.],
// )<train_f1_hinton>

// #figure(
//   image("figs/runs/" + run + "/valid_f1_hinton.svg"),
//   caption: [The f1 score of $cal(T)$ on the validation data during training.],
// )<valid_f1_hinton>

// The f1 score of the tasks in $cal(T)$ on the train data during training is shown in @train_f1_hinton, and the f1 score of the tasks in $cal(T)$ on the validation data during training is shown in @valid_f1_hinton.

= Conclusion

The sudden learning of 25 tasks, after having generalized independently to a joint solution to the first four, indicates that.
there is indeed an assiting effect to having multiple tasks in the development of these circuits. Masking away those four tasks delays grokking beyond the epochs feasible to train for within the experiment at hand.
