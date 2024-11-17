#import "@preview/unequivocal-ams:0.1.2": ams-article, theorem, proof
#import "@preview/equate:0.2.1": equate // <- for numbering equations

#let hash = "d58e064a6ba74afc89ef5779"
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
    We investigate the emergence and nature of algorithmic learning in transformer models through the lens of prime factorization. Our study trains a transformer model on multiple related tasks: predicting whether a number is prime and identifying its prime factors. Building upon recent work in mechanistic interpretability @nanda2023, we analyze how the model transitions from memorization to algorithmic generalization across these tasks. For numbers represented as $a + b$ (where $a,b < n$), we examine the model's learned representations and attention patterns using Fourier analysis and custom visualization techniques. Setting $n=113$ yields a dataset of 12,769 samples, allowing us to study how the model develops different generalization behaviors for tasks of varying complexity. Our findings reveal distinct patterns in the model's attention mechanisms and intermediate representations, suggesting the emergence of modular arithmetic operations similar to the Sieve of Eratosthenes. This work contributes to our understanding of how neural networks discover and implement mathematical algorithms #footnote[https://github.com/syrkis/miiii].
  ],
  bibliography: bibliography("zotero.bib"),
)

// body ///////////////////////////////////////////////////////////////////////

= Introduction

Recent years have seen deep learning models demonstrate remarkable proficiency in solving complex computational tasks. These models exhibit parallels to information-theoretic concepts, particularly in lossy data compression. For instance, the weights of GPT-2 are about a tenth of the size of its training data, akin to the compression ratios achieved by techniques like Huffman coding (mention lossy technique instead). Importantly, deep learning architectures can function both as archives—overfitting to training data—and as generalized algorithms @power2022. It can be both an archive and an algorithm for different tasks simultaneously.

A system capable of transitioning from archive to algorithm presents intriguing questions: Why doesn't it skip the archiving step and directly learn algorithms? What types of algorithms does it learn, and how reliably? Can the learning process be expedited? How does the presence of multiple tasks affect the learning process? Critically, what specific algorithm has been learned by a given system? Addressing these questions is essential for advancing the theoretical understanding of deep learning and enhancing their practical applications.


In deep learning, however, theory often lags behind practice, limiting our ability to mechanistically explain basic models that have generalized on even relatively simple, synthetically generated tasks. Exploring the mechanics of deep learning models is perhaps more akin to studying biology or botany than traditional computer science. This paper, for example, reverse-engineers a simple transformer model trained to solve modular arithmetic tasks. The simplicity of this training is akin to discovering an intriguing plant in a botanical garden (easy), while understanding its mechanics is akin to dissecting the plant to uncover the principles governing its growth and function (hard).

Primes, in particular, are an intersting domain for deep learning. A frequent feature of number theoretical problems is the ease with which they can be stated. This is true for trivial problems (are there an infinite number of primes?) and trivial sounding problems: "all even numbers can be expressed as the sum of two primes". The latter is known as the Goldbach conjecture, and remains unsolved.

My investigation probes the fundamental algorithmic structures internalized by a transformer model trained on basic modular arithmetic tasks, with slight variations in complexity. This approach provides insights into how and why specific algorithmic patterns emerge from seemingly straightforward learning processes. @generalization_levels the levels of generalization achieved across these tasks.

#figure(
  image("figs/miiii_f1_tasks_113_20000_last.svg", width: 110%),
  caption: [Final per task F1 scores (note generalization on primes 2, 3, 5 and 7).],
)<generalization_levels>

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

*Mechanistic _Implementability_* — #cite(<weiss2021>, form: "prose") presents the coding language RASP, which incorporates the architectural constraints of the transformer model into the language itself.
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

Our methodology closely follows #cite(<nanda2023>, form: "prose"), with key modifications to address prime factorization rather than modular addition. For a given prime $p$, we construct a dataset $[X|Y]$ where $X$ is the Cartesian product of digits less than $p$, representing all numbers of the form $x_0 dot p^0 + x_1 dot p^1$ where $x_0, x_1 < p$ (all two-digit base $p$ numbers). $Y$ is a binary vector indicating which primes $t < p$ are factors of the represented number. A transformer model is trained to predict $Y$ from $X$, with various hyper parameter configurations explored to study the emergence of generalization across different factorization tasks.

== Tasks


In natural language, the task can be described as "is $x$ a multiple of $t$?". For example, is $x=12$ a multiple of $t=3$? Is $x=53$ a multiple of $t=7$?
A number $x$ being a multiple of another $t$ means that $x mod t$ is zero. We are thus still in the domain of modular arithmetic.

The tasks vary in difficulty quite tangably. The expected cross entropy for $n$ classed problem is $ln(n)$. For the first four factors we have

// $
//   expected loss = ln(2), ln(3), ln(5), ln(7)
// $

which allins perfectly with what we see in the training curves.

#table(
  columns: (1fr, 1fr, 1fr),
  inset: 10pt,
  align: horizon,
  table.header(
    [],
    [${0, 1}$],
    [${2, 3, ..., f-1}$],
  ),

  [$f = p$], [Divisible ${0, 1}$], [Prime $f$ = $p$],
  [$f = {2, 3, ..., p_i}$], [Divisible ${0, 1}$], [Prime $f$ = $p$],
)

== Visualization

Much of the data worked with here is inherently high dimensional. For trianing, for example, we have $n$ steps, two splits (train/valid) about $p/ln(p)$ tasks, and three metrics (F1 score, accuracy, and loss). This, along with the inherenint opaqueness of deep learning models, motivated the developed custom visualization library#footnote[https://github.com/syrkis/esch] to visualize attention weights, intermediate representations, training metrics, and more.

== Data


*Input Space ($X$)*
Each input $x in X$ represents a number in base $p$ using two digits, $(a,b)$, where the represented number is $a + b$. For example, with $p=11$, the input space consists of all pairs $(a,b)$ where $a,b < 11$, representing numbers up to $11^2-1 = 120$. @miiii_x_11 visualizes this input space, with each cell showing the pair $(a,b)$.

#figure(
  image("figs/x_11_plot.svg", width: 110%),
  caption: [Representation of $X$, showing all pairs of $(x_0, x_1)$ for $p=11$. Top left shows (base 11) representation of 0, and bottom right representation of 120],
)<miiii_x_11>


*Output Space ($Y$)*
For each input $x$, we construct a binary vector $y in Y$ indicating which prime factors less than $p$ divide the number represented by $x$. For $p=11$, we test for divisibility by the primes 2, 3, 5, and 7, resulting in a binary vector of length 4. @miiii_y_11 visualizes these binary classifications, with each subplot showing divisibility by a specific prime, as well as $Y$ from the modular addition task.

#figure(
  image("figs/y_11_plot.svg", width: 120%),
  caption: [Representation of our $Y$ for $p = 11$. 2, 3, 5 and 7 and the four primes (and therefore tasks) less than 11, and $Y$ for the modular addition task.],
)<miiii_y_11>


The relationship between inputs and outputs can be formally expressed as:

$
  y_i = (
    x_0 dot p^0 + x_1 dot p^1
  ) mod f_j, quad forall f_j < p "where" t_j "is prime",
$<miiii>

Note that $y_i$ indicates divisibility, rather than remainder (as it does for the modular arithmetic task).


To provide additional insight into the structure of these numbers (and encourage the reader to think in rotational terms), @nats visualizes the distribution of various numerical properties in polar coordinates.

#figure(
  image("figs/polar.svg"),
  caption: [
    Multiples of 7 or 23 (left), 11 (middle), and primes (right) less than $113^2$ in polar coordinates ($n$, $n$).
  ],
)<nats>




== Model

The trained model follows the transformer architecture with several key design choices aligned with recent work on mechanistic interpretability @nanda2023, @lee2024a. The model consists of three main components: an embedding layer, transformer blocks, and an output layer. All weights are initialized as per #cite(<he2015>, form: "prose").

Input tokens are embedded into a $d$-dimensional space ($d=128$) using learned token and positional embeddings:
$
  z = "TokenEmbed"(x) + "PosEmbed"("pos")
$<embed>

The model contains three transformer blocks, each comprising multi-head attention with 4 heads (32 dimensions per head):
$
  "Attention"(Q, K, V) = "softmax"(Q K^T / sqrt(d_k))V
$<attn>
where $Q$, $K$, and $V$ are linear projections of the input (attention heads are combined through addition, not concatenation), followed by a feed-forward network with ReLU activation:
$
  "FFN"(x) = "ReLU"(x W_1)W_2
$<ffn>
mapping from 128 → 512 → 128 dimensions. Each component is followed by dropout (rate = 0.5) and includes residual connections.

The final representation is projected to binary predictions for each potential prime factor:
$
  hat(y) = z_(-1) W_("out")
$<output>
where $W_("out")$ projects to the number of prime factors being tested, and read the logits from the last position.

== Training


The model is trained using AdamW @loshchilov2019 with learning rate $10^(-4)$, weight decay 1.0, and $beta_1=0.9$, $beta_2=0.98$ following @nanda2023. We employ focal loss @lin2018 to handle class imbalance:

$
  L_("focal") = -alpha times (1 - p_t)^gamma times log(p_t)
$<focal_loss>

where $alpha$ is inversely proportional to class frequency and $gamma=2$ controls the contribution of well-classified examples. To accelerate generalization, we implement gradient filtering @lee2024a:

$
  g_t = nabla_theta L + lambda(alpha e_(t-1) + (1-alpha)g_(t-1))
$<grad>

where $e_t$ is the exponential moving average of gradients with decay rate $alpha=0.98$, and $lambda=2$ controls the influence of the slow-varying component.

// Training is done with all samples at each step (full batch) for 100,000 epochs. The model is evaluated on the validation set after each epoch, and the F1 score of the tasks in $cal(T)$ on the validation data during training is shown in @valid_f1_hinton.

In accordance with the mechanistic interpretability literature, extreme regularization is used with a dropout rate of 0.5, and weight decay of 1.0 (that's right!).



== Mechanistic Interpretability

A combination of linear products is itself a linear product. As a mechanistic interpretability rule of thumb, one should look at the ouputs of the non-linear transformations. In our case that will be the attention weights, and the intermediate representations with each transformer block's MLP (which follows a ReLU activation).
Aditionally, the embeddings layers will be inspected. blah blah.

Our interpretability approach combines visualization techniques with frequency analysis to understand the learned algorithmic patterns. Following @nanda2023, we analyze both the attention patterns and the learned representations through several lenses:

*Attention Visualization*
We developed a custom visualization library#footnote[https://github.com/syrkis/esch] to visualize attention weights and intermediate representations. The library allows for the visualization of attention patterns across different layers, as well as the visualization of intermediate representations at each layer. These visualizations provide insights into the learned patterns and help identify potential areas of improvement.

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

In @atten_weight we see a vertical periodic pattern, which is expected, as the model is trained to predict the prime factorization of the number $a * 113 + b$.
#lorem(140)



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
For each of the $sqrt(n) + 1$ tasks, the focal loss @lin2018 (@focal_loss) and f1 score are calculated every epoch. Focal loss is cross-entropy weighted by the inverse of the frequency of the positive class ($alpha$) in the task. The loss function can be tweaked to emphasize difficult samples with the $gamma$ parameter.



The frequency of a positive sample in task $i$ is used as the weight for the focal loss during training.
Furthermore, a one-hot vector is used to mask tasks to shield the model from a particular signal during training.

= Results


== Training

== Embeddings

== Attention

== Feed-forward

#figure(
  stack(
    dir: ttb,
    image("figs/train_acc.svg", width: 100%),
    image("figs/valid_acc.svg", width: 100%),
  ),
  caption: [Training and validation accuracy],
)<generalization_levels_>


We see that the positonal embeddings are orthogonal. The token embeddings of $x_0$ and $x_1$ are offset, allowing for the cosine and sine tabel to be learned for both.
NOTE: they might not be orthogonal, but rather pointing in opposite directions (we only have two vectors, so orthogonality is not needed.

#figure(
  image("figs/pos_emb.svg", width: 100%),
  caption: [Positional embeddings for the first $37$ for a model trained on @nanda (top) and @miiii (bottom). The low information contained in the postional encoding of @nanda is to be exxpected as the task is comotative, while @miiii is not—$(x_0 + x_1) mod p = (x_1 + x_0) mod p$ but $((x_0 p^0 + x_1 p^1) mod p) != ((x_1 p^1 + x_0 p^0) mod p)$..],
)<generalization_levels_>


#figure(
  image("figs/miiii_f1_tasks_113_20000.svg", width: 100%),
  caption: [Evolution of per task F1 scores.],
)<generalization_levels_>



- sin/cos lookup tables in embedding layer.
- does pos not matter for this task? No, cos it is not commutative. (a + b) mod p = (b + a) mod p -> Nanda. But (a p^1 + b p^0) mod p != (b p^1 + a p^0) mod p.

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




= Analysis

= Discussion

= Conclusion
