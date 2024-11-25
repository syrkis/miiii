#import "@preview/unequivocal-ams:0.1.2": ams-article, theorem, proof
#import "@preview/equate:0.2.1": equate // <- for numbering equations
#import "@preview/unify:0.6.1": num // <- for making numbers look nice


#let f_hash = "4a98603ba79c4ed2895f9670"
#let p_hash = "0c848c1444264cbfa1a4de6e"
#show: equate.with(breakable: true, sub-numbering: true)
#set math.equation(numbering: "(1.1)", supplement: "Eq.")
#set raw(align: center)

#show: ams-article.with(
  title: [Mechanistic Interpretability of Irreducible Integer Identifiers],
  authors: (
    (
      name: "Noah Syrkis",
      // department: [Department of Computer Science],
      // organization: [University of Copenhagen],
      // location: [Copenhagen, Denmark],
      // url: "syrkis.com",
    ),
    (
      name: "Anders Søgaard",
      // department: [Department of Computer Science],
      // organization: [University of Copenhagen],
      // location: [Copenhagen, Denmark],
      // url: "anderssoegaard.github.io/",
    ),
  ),
  abstract: [
    This paper investigates the emergence and mechanistic nature of algorithmic learning in transformer models through the lens of modular arithmetic and prime factorization. Building upon recent work in mechanistic interpretability, a transformer model is trained to predict remainders when dividing base-p numbers by prime factors. For numbers represented as $x_0 p^0 + x_1 p^1$ (where $x_0,x_1 < p$), the model must learn distinct strategies for each prime factor, with task difficulty scaling naturally with the size of the factor. Setting $p=113$ yields 29 parallel tasks and #num(12769) samples, allowing for the study of how the model develops different computational strategies for tasks of varying complexity. Analysis of learned representations and attention patterns reveals distinct periodicities in the model's internal representations, suggesting the emergence of trigonometric basis functions similar to those found in simpler modular arithmetic tasks. This work contributes to our understanding of how neural networks discover and implement mathematical algorithms, particularly in settings with multiple related tasks of varying complexity. As a second contribution, this paper reproduces #cite(<lee2024a>, form:"prose")'s finding that amplifying slow moving gradients, can significantly speed up generalization.
    #footnote[https://github.com/syrkis/miiii].
  ],
)


#let appendix(body) = {
  set heading(numbering: "A", supplement: [Appendix])
  counter(heading).update(0)
  body
}

// body ///////////////////////////////////////////////////////////////////////


= Introduction

Recent years have seen deep learning (DL) models achieve remarkable proficiency in complex computational tasks, including protein structure prediction @jumper2021, strategic reasoning @dinan2022, and natural language generation—areas previously thought to be the exclusive domain of human intelligence. Traditional (symbolic) programming allows functions like $f(x, y) = cos(a dot x) + sin(b dot y)$ to be implemented with clear typographical isomorphism—meaning the code's structure directly mirrors the mathematical notation. For example, in Haskell: `f x y = cos(a * x) + sin(b * y)`. In contrast, DL models are inherently subsymbolic, meaning that the models' atomic constituents (32-bit floating-point numbers centered around 0) are meaningless when viewed directly. For reference, @subsymbolic shows a DL-based implementation of $f$#footnote[Note that in the rest of the paper $f$ refers to a prime number less than $p$.].

Indeed, the increasing prevalence of DL can be understood as a transition from symbolic to subsymbolic algorithms: the gradual subsuming of computational tasks. Precursors to modern DL methods learned how to weigh human-designed features @shannon1950, with later works learning to create features from data to then weigh @tesauro1993, @silver2017—in combination with tree search strategies, in the case of games @browne2012. Recent DL work has even eliminated tree search, mapping directly from observation space to action space @ruoss2024. Pure DL methods are thus increasingly prevalent, but almost equally inscrutable, with recent works still attempting to define what interpretability even means in the DL context @lipton2018. Given the breadth @cybenko1989 of tasks that DL models can be (and are) trained to solve—along with their subsymbolic nature—it is, however, hardly a surprise that their interpretation remains difficult.

Mathematically, DL refers to a set of methods that combine linear maps (matrix multiplications) with non-linearities (activation functions). Formally, all the potential numerical values of a given model's weights $W$ can be thought of as a hypothesis space $cal(H)$. Often, $cal(H)$ is determined by human decisions (number of layers, kinds of layers, sizes of layers, etc.). $cal(H)$ is then navigated using some optimization heuristic, such as gradient descent, in hope of finding a $W$ that "performs well" (i.e., successfully minimizes some loss $cal(L)$ computed by a differentiable function) on whatever training data we have. This vast hypothesis space, while enabling impressive performance and the solving of relatively exotic#footnote[Try manually writing a Haskell function that classifies dogs and cats.] tasks, makes it challenging to understand how any particular solution actually works.

The ways in which a given model can minimize $cal(L)$ can be placed on a continuum: on one side, we have overfitting (remembering the training data, or functioning as an archive akin to lossy and even lossless compression), and on the other, we have generalizing (learning the rules that govern the relationship between input and output, or functioning as an algorithm).

When describing a mechanistic explanation for a given DL model, generalization is a necessary (though insufficient) condition. Generalization ensures that there _is_ an algorithm present to be uncovered; however, it is possible for that algorithm to be so obscurely implemented that reverse engineering, for all intents and purposes, is impossible. Various tricks, known as "regularization," exist to incentivize the emergence of algorithmic rather than archiving behavior @ba2016, @krizhevsky2017, @krogh1991. As will be covered in @related_works, the mechanistic interpretability (MI) literature has, despite its nascent state, already established some conventions and successes. Circuits solving basic algorithmic tasks have been successfully reverse-engineered @nanda2023, and aspects of this workflow have been automated @conmy2023.

This empirical approach to understanding neural networks makes MI more akin to botany than theoretical computer science: while finding an interesting specimen (training a working model on an original task) is relatively straightforward—like stroling a botanical garden, looking for an unstudied flower—carefully dissecting it to understand its internal mechanisms remains challenging and labor-intensive.

The specimen of this paper was chosen since, as of yet, no MI work has explored the effect of multitask learning, the focus of this paper. Multitask learning also has a regularizing effect @baxter2011—weights $W$ in $cal(H)$ that perform well across tasks are more likely to be general. #cite(<baxter2011>, form:"prose") refer to the set of hyptheses spaces for the different tasks in a given environment of tasks as $cal(H) in HH$. A $W$ performing well across tasks can thus be thought of as the intersection of the hyptheses spaces across $HH$.

In this spirit, the present paper builds on the work of #cite(<nanda2023>, form:"prose"), which trains a transformer @vaswani2017 model to perform modular addition, as seen in @nanda_task:

$
  (x_0 + x_1) mod p, quad forall x_0, x_1 < p, quad p = 113
$<nanda_task>


This is referred to as $cal(T)_("nanda")$. The task of this paper focuses on predicting remainders modulo all primes less than $p$, where $x$ is interpreted as $x_0 p^0 + x_1 p^1$, formally shown in @miiii_task, and is referred to as $cal(T)_("miiii")$:

$
  (
    x_0 p^0 + x_1 p^1
  ) mod f, quad forall x_0, x_1 < p, quad forall f < p, quad p = 113
$<miiii_task>

$cal(T)_("miiii")$ thus differentiates itself from $cal(T)_("nanda")$ in two significant ways: _1)_ it is non-commutative, and _2)_ it is multitask. These differences present unique challenges for mechanistic interpretation, as the model must learn to handle both the order-dependent nature of the inputs and develop shared representations across multiple modular arithmetic tasks. Further, $cal(T)_("miiii")$ is harder than $cal(T)_("nanda")$ the model does not generealize when trained in the same way. Therefore, #cite(<lee2024a>, form:"prose")'s recent work on making generalization happen quicker, by positing the the gradeints through time can be viwed as the sum a slow varying genrealzing component (which is boosted) and a quick varying overfitting component (which is muted), was (successfully) replicated so as to make training tractable.

More genereally, modular arithmetic on primes is a particularly useful task for MI as it ensures uniformity among the output classes, allows for comparison with other MI work, and, from a number-theoretic point of view, primes contain mysteries ranging from the trivially solved—are there an infinite number of primes?—to the deceptively difficult—can all even numbers larger than 4 be described as the sum of two primes? The later, known as Goldbach's Conjecture, remains unsolved after centuries.

Lastly, the reader is asked to accept the inspection of a DL model transitioning from archive to algorithm on multiple simultatnious tasks as inherently interesting, independnt of the current literature's sparcity on the subject.

// Lastly, the fact that MI lags so far behind the cutting edge of DL means that the models in which interesting MI is performed are relatively simple to train. The MI workflow is thus perhaps more similar to botany than theoretical computer science. While the models (the specimens) are easy to cultivate, dissecting them to uncover the principles governing their function remains a challenging endeavor. This paper aims to contribute to this effort by exploring the mechanistic interpretability of models trained on multitask modular arithmetic.

= Related works<related_works>

Mechanistic Interpretability as a field is relatively new, though the objects of its study have been seen wide spread adoption in the last decade. And indeed, many reverse engineering methods from other fields such as neuroscience or even computer forensics, have there uses here. The following sections outlines these fields, and their use for the task at hand.

== Generalization and grokking

#cite(<power2022>, form: "prose") shows generalization can happen #quote(attribution: cite(<power2022>), "[...] well past the point of overfitting"), dubbing the phenomenon "grokking". The phenomenon is now well established @nanda2023, @humayun2024, @wang2024, @conmy2023. #cite(<nanda2023>, form: "prose") shows that, the a generalized circuit #quote(attribution: cite(<nanda2023>), "arises from the gradual amplification of structured mechanisms encoded in the weights"), rather than being a relatively sudden and stochastic encounter of an appropraite region of $cal(H)$. Further, by regarding the series of gradients as a stochastic signal, #cite(<lee2024a>, form: "prose") propose decomposing the signal into two components: a fast-varying overfitting component and a slow-varying generalization component. They show that amplification of the slow-varying component accelerates grokking substantially (more than fifty-fold in some cases). This echoes the idea that generalized circuits go through a sort of embryology, through the entire duration of training @nanda2023, rather than suddenly occuring. To the extent that this phenomenon is wide spread, it bows well for generalizable deep learning, in that the generalizing signal that one would want to amplify exists long before the model is fully trained.

Conceptually, #cite(<lee2024a>, form:"prose") argues that in the case of gradient decent, the ordred sequence of gradient updates can be viewed as consisting of two components: _1)_ a fast varying overfitting component, and _2)_ a slow varying generalizing components. The general algorithm exaplining the realtionship between input and outout is the same for all samples, whereas the weights that allow a given model to function is archive are unique for all samples. Though not proven, this intuition bears out in that generealiazation is sped up fifty fold in some cases.


== Mechanistic interpretability (MI)

MI is a relatively new field, and the methods are still being developed.
#cite(<lipton2018>, form: "prose") discusses various definitions of interpretability, including mechanistic interpretability (though they don't call it that) in which the mechanisms of the model are reverse engineered. This is on the opposite scale of forms of interpretability such as feature importance, which is a measure of how much a feature contributes to the model's prediction (i.e. the presence of red might correlated highly with an image classified as containing a rose).
The model trained by #cite(<nanda2023>, form:"prose") to solve $cal(T)_("nanda")$ is reverse engineered using using a variety of qualitative approaches like visualizing the activations over the entire #num(12769) ($113^2$) dataset, and performing singular value decomposition on the token embeddings matrix.
It is discovered that the generalized circuit uses a discrete Fourier transform (rotation in the complex plane) to solve the problem. Specifically, the embedding layer learns a lookup tale for the cosine and sine values of the input, while the feed word layer of the transformer block learns to combine these values though multiplication, addition and trigonometric identities. Note that $cal(T)_("nanda")$ is commutative meaning that @commutative holds.

$
  (x_0 + x_1) mod p = (x_1 + x_0) mod p
$<commutative>

Algorithmically, through ablation studies it is thus shown that the embeddings layer $W_E$ learns the $cos$ and $sin$ lookup tables, and that the feedword branch of the transformer block performs the multiplication, yielding $cos(w a) dot sin(w b)$ on which the unembedding layer than performs the linear algebra equivalent of an $arg max$ by reading of logits on the $y$-axis.

#cite(<conmy2023>, form: "prose") automates aspects of the mechanistic interpretability work flow for specific tasks, namely the circuit discovery. In MI a circuit refers to a path through the model weights that detects a specific pattern. Ablation studies show that the elements of the model not involved in the circuit can be removed without negatively affecting the generalization performance of the model.

#cite(<weiss2021>, form: "prose") presents the coding language RASP, which incorporates the architectural constraints of the transformer model into the language itself.
This forces the programmer to be "thinking like a transformer" (which is the title of their paper).
The multilayer perception (MLP) can be thought of as performing a map, (applying a function to every element of a set, while the attention mechanism is a way to combine (reduce) the information from different layers. A language grammatically constrained by the invariances of the transformer, is perhaps the best way for a human to demystify the architecture. However, the generalized circuits are as implementable in any other language.

The operation $x mod p$ falls under a cyclic group, meaning that any number $n$ can be written as $n = k p + r$ where $0 <= r < p$ is the remainder and $k$ is some integer. This means that the elements of the group cycle through the possible remainders $0, 1, ..., p-1$ as $n$ increases. Convenient consequences hereof includes that the distribution of the remainders is uniform.


Theory, however, is far behind practice when it comes to DL. Is DL best understood from and information theoretic @yu2021, a geometric @bronstein2021, or a category theretic @gavranovic2024 perspective.
The success of DL has, however, not brought much theoretical understanding.

Therefor the mechanistis interpretability literature tends to focus on simple algorithmic tasks, for which we ourselves can write a clear, concice algorithms, as well using the ReLU acitvation function (which for mathematical reasons favors a privlidged bases, i.e. orthogonality) @nanda2023, @conmy2023.


A system capable of transitioning from archive to algorithm presents intriguing questions:
Why not skip the archiving step and directly learn algorithms? What types of algorithms does it learn, and how reliably?
Can the learning process be expedited? How does the presence of multiple tasks affect the learning process?
What specific algorithm has been learned by a given system?
How can it exist as an archive and an algorithm simultaneously?
Addressing these questions is essential for advancing the theoretical understanding of deep learning and enhancing its practical applications.


== Multi-task Learning in DL

As stated, multi-task learning has been shown to have a regularizing effect @baxter2011, @maurer as the hypthesis *W* that performs well across the of hypthesis spaces $HH$ is more likely to be general. Viewed information theoretically, this concept is reminicent of #cite(<shannon2001>, form:"prose")'s asymptotic equipartition property @cover2006, or even more generally, the law of large numbers, which state that the more samples we have of a distribution, the closer our estimates of its underlying properties will align with the true underlying property.

In the DL context, multi-task learning is done by having the last layer ouput predictions for multiple tasks independently. Thus, whereas $cal(T)_("nanda")$ outputs a single $1 times 113$ vector for each of the potential remainders, $cal(T)_("miiii")$, as we shall see, outputs one vector for each prime $f < p$ (29 when $p=113$), each of which has shape $1 times f$. The embeddings layer and the transformer block is thus shared for all tasks, meaning that representations that perform well across tasks are incentivised.

== Loss Functions and Training Dynamics

Perhaps the most widespread loss functions used in deep learning are mean cross-entropy @mce (for classification) and mean squared error @mse (for regression).

$
  L_("MCE") &= 1 / n sum_(i=1)^n sum_(j=1)^k y_p_(i j) ln(1 / hat(y)_p_(i j))#<mce> \
  L_("MSE") &= 1 / n sum_(i=1)^n (y_i - hat(y)_i)^2 #<mse>
$

These have various computational and mathematical properties that make them convenient to use, while they, however, struggle to generalize @jeon2022. Due to its prevalence, however, MCE is chosen in this paper. However, since have multiple tasks, the MCE is modified as shown in

== Deep Number Theory

Multiple papers describe the use of deep learning to detect prime numbers @egri2006, @lee2024, @wu2023a.
None are particularly promising as prime detection algorithms, as they do not provide speedups, use more memory, or are less accurate than traditional methods.
However, in exploring the foundations of deep learning, the task of prime detection is interesting, as it is a simple task that is difficult to learn, and is synthetic, meaning that the arbitrary amounts of data are generated by a simple algorithm.

Prime numbers, in particular, are an interesting domain for deep learning. A frequent feature of number theoretical problems is the ease with which they can be stated. This is true for trivial problems (such as proving there are infinitely many primes) and deceptive problems (such as "all even numbers can be expressed as the sum of two primes"). The latter, known as Goldbach's conjecture, remains unsolved. There are about $n/ln(n)$ primes less than $n$. To test if a given number $n$ is prime, it is sufficient to test if it is divisible by any prime less than $sqrt(n)$ (Sieve of Eratosthenes), of which there are about $sqrt(n)/ln(sqrt(n))$.

== Transformer Architecture

Various modifications/simplifications have been made to the transformer block @he2023, @hosseini2024.
Transformers combine self-attention (a communication mechanism) with feed-forward layers (a computation mechanism).
Importantly, transformers tend to rely on residual streams (I will elaborate).
I am currently using the original transformer block, but I want to switch to @he2023's block, as it is simpler and more interpretable—but there is not much research on it yet.

Traditional loss functions like cross-entropy and mean squared error,
have been shown to not genrealize well to out of distribution data @yu2021.
Indeed, additional regularization techniques are a hallmark of many modern architectures,
the most extreme example of which is perhaps the original transformer @vaswani2017—layernorm @ba2016,
dropout, weight decay, residual connections, are all integral components of the original architecture,
though recent years have seen simplifications yielding similar performance @he2023.
Importantly, deep learning architectures can function both as archives—overfitting to training data—and as generalized algorithms @power2022.


= Methods

However, how exactly a given model implements an algorithm is a non-trivial question—as we shell see, even modular addition is implemented in an obscure way @nanda2023.
This investigation probes the fundamental algorithmic structures internalized by a transformer model trained on a set of basic prime number-related modular arithmetic tasks, with slight variations in complexity. This approach provides insights into how and why specific algorithmic patterns emerge from seemingly straightforward learning processes.

My setup thus differentiates itself from Nanda's in two crucial ways:

1. Mine is non-commutative.
2. It is multi-task.

A model deep learning model, $cal(M)$, consits of a set of model weights $cal(W)$ and a procedure on how to apply these to a given input $cal(X)$. Viewed in the context of the procedure, the set of potential valuesues of $cal(W)$ can be thought of as a hypothesis space $cal(H)$ on the mapping between $cal(X)$ and $cal(Y)$, with respect to a loss function $cal(L)$. Algorithms like gradient decent, are heiristics for finsing optimal / optimised values of $cal(W)$ within $cal(H)$. $H$ itself is not modified by optimization algorithms of this level (i.e. $a x+b$ yield optimal $a "and" b$ values, but we might need a $x^2$ term to describe the given phenomen.

#cite(<baxter2011>, form:"prose") further extends the notion of generaliation and training to a multi-task paradigm.

== Tasks

Stated plainly: the task predicts the remainder when dividing a two-digit base-$p$ number by each prime factor $f$ less than $p$. The set of prime factors we construct tasks for is thus $F = {f in PP : f < p}$
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
    Periodic patterns in polar coordinates $(n, n)$ for numbers less than #num(12769). Left: numbers with remainder 0 mod 7 or 23 (see the two spirals). Middle: numbers with remainder 0 mod 11. Right: prime numbers.
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


$
  mat(delim:"[", quad x_0 quad x_1 quad \_ quad)
$

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

    $0, 1 / 2, 1 / 5, 1 / 10$,
    $0, 1 / 2, 2$,
    $0, 1 / 10, 1 / 2, 1$,
    $128, 256$,
    "3e-4, 1e-4",
    "4, 8",
  ),
  caption: "Hyperparameter search space for training.",
)<hyper_param_search_space>

The model is trained using AdamW @loshchilov2019 with $beta_1=0.9$, $beta_2=0.98$ following @nanda2023. To handle the varying number of classes across tasks (from 2 classes for mod 2 to 109 classes for mod 109), a modified (weighted) mean cross-entropy (@mce) loss is created, correcting for the difference in expected loss within each task. Note that $EE[L_("MCE")] = ln(1/k)$, where $k$ is the number of classes within the task in question. Correcting for this, the loss function becomes as shown in @mmce.


$
  L_(cal(T)_"miiii") &= - &&sum_(f in F) L_"MCE"_f / (-ln(f)) \
  &= - &&sum_(f in F) (sum_(i=1)^n sum_(j=1)^(f) y_(k_f i j) ln(hat(y)_(k_f i j)) ) / (- n ln(f)) \
  &= &&sum_(f in F)sum_(i=1)^n sum_(j=1)^(f) (y_(k_f i j)ln(hat(y)_(k_f i j)) ) / (n ln(f)) #<mmce>
$

To accelerate generalization, gradient filtering as per #cite(<lee2024a>, form: "prose") is implemented and replicated.

$
  g_t = nabla_theta L + lambda(alpha e_(t-1) + (1-alpha)g_(t-1))
$<grad>

where $e_t$ is the exponential moving average of gradients with decay rate $alpha=0.98$, and $lambda=2$ controls the influence of the slow-varying component.

Training uses full batch gradient descent with the entire dataset of $p^2$ samples (#num(12769) when $p=113$). The model is evaluated on a held-out validation set after each epoch, tracking per-task accuracy and loss. As the setup used in $cal(T)_"nanda"$, training was done on thirty percent of the total dataset, with the remaining used for validation (1000 samples) and testing (remaining). Further as $cal(T)_"miiii"$ involves the learning of 29 (when $p=113$) tasks rather then 1, and due to each tasks non-comotativity, a larger hidden dimension of 256 was added to the hyper parameter search space, as well as the potential for 8 heads ($cal(T)_"nanda"$ was solved with a hidden dimensions of 128, and 4 heads). The number of transformer blocks were kept at 1 as this ensures consistency with $cal(T)_"nanda"$ (and as full generalizaion was possible, as we shall see in the results).


== Visualization

Much of the data worked with here is inherently high dimensional. For training, for example, we have $n$ steps, two splits (train/valid) about $p/ln(p)$ tasks, and two metrics (accuracy, and loss). This, along with the inherent opaqueness of deep learning models, motivated the developed custom visualization library, `esch`#footnote[https://github.com/syrkis/esch] to visualize attention weights, intermediate representations, training metrics, and more. The most important plot type for the reader to keep in mind is seen in @plot_type. As there are only #num(12769) samples when $p=113$, all samples can be fed at once to the model. Inspecting a specific activation thus yields a $1 times$ #num(12796) vector $v$, which can be reshapes at a $113 times 113$ matrix, with the two axis varying $x_0$ and $x_1$ from 0 to 112, respectively. The top left corner than shows the given value for the sample $(0 dot p^0 + 0 dot p^1)$, and so on.

#figure(
  image("figs/plot_intro.svg", width: 110%),
  caption: [Top left $37 times 37$ slice of the attention pattern from $hat(y)$ to $x_0$ in the first attention head of all $(x_0, x_1)$ pairs, for a model trained on $cal(T)_"nanda"$. Note that each squre of the plot represents a unique sample, and are thus entirely independt from one another. The periodicity is thus a function of the model learning an order of the natural numbers in question.],
)<plot_type>


== Mechanistic Interpretability

A combination of linear products is itself a linear product. As a mechanistic interpretability rule of thumb, one should look at the outputs of the non-linear transformations. In our case that will be the attention weights, and the intermediate representations with each transformer block's MLP (which follows a ReLU activation).
Additionally, the embeddings layers will be inspected. blah blah.

Our interpretability approach combines visualization techniques with frequency analysis to understand the learned algorithmic patterns. Following @nanda2023, we analyze both the attention patterns and the learned representations through several lenses:

*Attention Visualization*
Using `esch`, the custom visualization library, to visualize attention weights and intermediate representations. The library allows for the visualization of attention patterns across different layers, as well as the visualization of intermediate representations at each layer. These visualizations provide insights into the learned patterns and help identify potential areas of improvement.

*Fourier Analysis*
As periodicity is established by #cite(<nanda2023>, form: "prose") to be a fundamental feature of the model trained on $cal(T)_"nanda"$, the fast Fourier transform (FFT) algorithm is used to detect which frequencies are in play.
Note that any square image, can be described as a sum of 2d sine and cosine waves varying in frequency from 1 to the size of the image divided by 2 (plus a constant).
This is a fundamental tool used in signal processing. The theory is briefly outlined in @fft for reference.
// To quantify periodic patterns in both attention weights and intermediate representations, we decompose them into their constituent frequencies using the discrete Fourier transform:

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

In @plot_type we see a vertical periodic pattern, which is expected, as the model is trained to predict the prime factorization of the number $x_0 * 113 + x_1$.



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

The best performing model was trained with the hyper-parameters in @hyper_param_search_result. As seen in figures @trainig_acc and @trainig_loss, the model grokked on all 29 tasks, achieving perfect accuracy on all 29 tasks on the validation and test sets. Note that tasks 2, 3, 5 and 7 generealizes in succession of one another, while rest happen, more or less simultaneously, after the initial four. This could indicate that a more geneal solution has been found, allowing for a sort of phase transition for the remaining tasks, by reusing circuitry developed through the first four.

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
)<hyper_param_search_result>


#figure(
  stack(
    dir: ttb,
    image("figs/" + f_hash + "/loss_train_training.svg"),
    image("figs/" + f_hash + "/loss_valid_training.svg"),
  ),
  caption: [Representation of training and validation loss ($x$-axis is in log scale).],
)<trainig_loss>

#figure(
  stack(
    dir: ttb,
    image("figs/" + f_hash + "/acc_train_training.svg"),
    image("figs/" + f_hash + "/acc_valid_training.svg"),
  ),
  caption: [Representation of training and validation acc ($x$-axis is in log scale).],
)<trainig_acc>


== Positional embeddings


@pos_emb shows the positional embeddings of the $cal(T)_"nanda"$ to be virtually identical (cosine similarity of 0.95), which is to be expected due to the tasks commutativity (a given value at $x_0$ or $x_1$ contributes the same to the task). The same measure for a model trained on $cal(T)_"miiii"$ is -0.64, translating the embeddings differently for the two positions. This is to be expected as by the task's non-comutativity $x_0 dot p ^ 0 != x_0 dot p^1$. Inspecting the positional embeddings confirms the obvious: position matters.

#figure(
  image("figs/pos_emb.svg"),
  caption: [Positional embeddings for the a network trained on @nanda_task (top) and @miiii_task (bottom)],
)<pos_emb>


== Token embeddings


@s shows us that the singular values of the the token embeddings learned for task $b$ to be much more diffuse than those for task $a$. As stated, the embedding layer of the $cal(T)_"nanda"$ trained models represents a look table for the sine and cosine values of the tokens—hance the periodicity of the most significant singular vectors @p_U. Visual inspection of the top most vectors of @f_U indeed shows periodicity, but a much large fraction of the vectors is reuired to capture the same amount of variance @s.

#figure(
  image("figs/S.svg"),
  caption: [Singular values of $U$ for $cal(T)_("nanda")$ (top) and $cal(T)_("miiii")$ (bottom)],
)<s>

#figure(
  image("figs/p_U.svg"),
  caption: [Most significant singular vectors of $U$ for $cal(T)_("nanda")$],
)<p_U>

#figure(
  image("figs/f_U.svg"),
  caption: [Most significant singular vectors of $U$ for $cal(T)_("miiii")$],
)<f_U>



To further understand the underlying structure of the token embeddings, the fast Fourier transform (FFT) algorithm is used. @p_f shows the five particularly active frequencies for the $cal(T)_"nanda"$-model. For the $cal(T)_"miiii"$-model we see a much broader spectrum of frequencies are active, though comparind to a randomly initialized baseline, the periodicity remains apperent. This is to be expected if the network too implements the cosine-sine look table @nanda2023, as each task relates to a partifular prime $f$—no point is hit twice when rotating through $CC$ with $f$ steps for very $f in F$.


#figure(
  stack(
    dir: ttb,
    image("figs/fourier_p_m.svg"),
    image("figs/fourier_p_f.svg"),
  ),
  caption: [$W_E_t_(cal(T)_("nanda"))$ in Fourier space with row norm below],
)<p_f>

Projecting the positional embeddings onto a Fourier basis, however, shows that the periodicity is indeed preserved.

#figure(
  stack(
    dir: ttb,
    image("figs/fourier_f_m.svg"),
    image("figs/fourier_f_f.svg"),
    image("figs/fourier_r_m.svg"),
    image("figs/fourier_r_f.svg"),
  ),
  caption: [$W_(E_t_(cal(T)_"miiii"))$ (top) and random @he2015 matrix of same shape (bottom) in Fourier space with row norm below each.],
)<f_f>

As is apparent in @f_f and @p_f a lot more frequencies are in play when training for $cal(T)_("miiii")$ than $cal(T)_("nanda")$.




// The fact of periodicity in @f_f despite the presence of multiple tasks with unique rotational steps around the circle, the non commutative nature of the task, is further @nanda2023 indication that trigonometric tables are a reliably used representation of the architecture.

//#figure(
//stack(
//dir: ttb,
// image("figs/p_f.svg"),
//   image("figs/p_f_norm.svg"),
//),
// caption: [Frequencies of $W_(E_(cal(T)_("nanda")))$ in Fourier space],
//)<p_f>

== Attention patterns

Unlike that @nanda_task task, our attention heads focus on one digit or the other. This could be due to the non-commutative nature of the task.


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

// #figure(
//   image("figs/pos_emb.svg", width: 100%),
//   caption: [Positional embeddings for the first $37$ for a model trained on @nanda_task (top) and @miiii_task (bottom). The low information contained in the positional encoding of @nanda_task is to be expected as the task is commutative, while @miiii_task is not—$(x_0 + x_1) mod p = (x_1 + x_0) mod p$ but $((x_0 p^0 + x_1 p^1) mod p) != ((x_1 p^1 + x_0 p^0) mod p)$..],
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


// #figure(
//   stack(
//     dir: ttb,
//     spacing: 1em,
//     image("figs/miiii_113_U_top_10.svg", width: 110%),
//     image("figs/miiii_113_S_top_37.svg", width: 100%),
//   ),
//   caption: [We see that the 10 most significant vectors of $U$ are periodic!],
// )


// #figure(
//   image("figs/miiii_113_F_W_E.svg"),
//   caption: [Active frequencies in Fourier space of Embedding],
// )

// #figure(
// image("figs/miiii_113_F_neuron_0.svg"),
// caption: [Neuron 0 for sample 1 in Fourier space.],
// )



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
//
= Analysis and Discussion



= Further work

The mysteries of primes and deep learning are both plentiful, and there are many fundamental questions to be answered in mixing the two. How does training a model on $p$ affect it performance on a $q > p$. How does predicting divisibility directly, compare to predicting remainders (both have been explored in this setup). In this spirit, the code associated with this paper is available as a pypi package, and can be installed with `pip install miiii`.

= Conclusion

The sudden learning of 25 tasks, after having generalized independently to a joint solution to the first four, indicates that.
there is indeed an assiting effect to having multiple tasks in the development of these circuits. Masking away those four tasks delays grokking beyond the epochs feasible to train for within the experiment at hand.


#bibliography("zotero.bib")

#pagebreak()

#appendix[
  #heading(level: 1, "Appendix", numbering: none)



  = Fast Fourier Transform<fft>

  The inner product between two vectors $bold(v) "and" bold(u)$ of length $n$ can be written as per @inner_product.

  $
    sum_i^n bold(v)[i] bold(u)[i]
  $<inner_product>

  We can extend the meaning of inner products to functions $f "and" g$ over the interval $[a;b]$ with @inner_product_function.

  $
    integral_a^b f(x)g(x) d x
  $<inner_product_function>

  It is a fact that any continuous, differentiable function $f(x)$, can be written as a sum of cosine and sine terms plus a constant as per:

  $
    f(x) =A_0 / 2 + sum_(k=1)^(infinity) (A_k cos(k x) + B_k sin(k x))
  $

  Where $A_k$ and $B_k$ are the normalized innner products $angle.l f(x), cos(k x) angle.r$ and $angle.l f(x), sin(k x) angle.r$ respectively
  #footnote[Note the pointy brackets denote inner product]. These are explicitly written out in @AB_k.

  $
    A_k = 1 / pi integral_(-pi)^pi f(x) cos(k x) d k, quad
    B_k = 1 / pi integral_(-pi)^pi f(x) sin(k x) d k
  $<AB_k>

  This can be similarly extended for that grid, which is the basis for the two-dimensional FFT.

  #pagebreak()

  = Subsymbolic implementation of $f(x, y)$<subsymbolic>

  Compute $f(x)$ for ${(a,b) in NN^2 : 0 <= a,b < 113}$, by adding the two rows of $W_E_"pos"$ in @embeds to a one-hot encoded $a$ and $b$, and then multiplying by $W_E_"tok"$. Then multiply by $W_k, W_q$ and $W_v$ indepently in perform the operation described in @attn, and then add to the output of the embedding operations. Send that through the a feed-forward network with the weights in @ffwd_fun, and voila. The reader is asked to confirm visually that the weight in the figures indeed compute $f(x, y) = cos (a x) + sin (b x)$ when applied in the order described above.

  #figure(
    stack(
      dir: ttb,
      image("figs/tok_emb_prime.svg", width: 80%),
      image("figs/pos_emb_prime.svg", width: 80%),
    ),
    caption: [$W_E_"tok"$ and $W_E_"pos"$],
  )<embeds>
  #figure(
    stack(
      dir: ltr,
      spacing: 0pt,
      image("figs/attn_v_prime.svg", width: 40%),
      image("figs/attn_k_prime.svg", width: 40%),
      image("figs/attn_q_prime.svg", width: 40%),
    ),
    caption: [$W_k$, $W_q$ and $W_v$],
  )
  #figure(
    stack(
      dir: ltr,
      spacing: 0pt,
      image("figs/ffwd_w_in_prime.svg", height: 90%),
      image("figs/ffwd_w_out_prime.svg", height: 90%),
    ),
    caption: [$W_"in"$ and $W_"out"^T$],
  )<ffwd_fun>
  // #figure(
  //   image("figs/unbeds_prime.svg"),
  // )<unbeds>
]
