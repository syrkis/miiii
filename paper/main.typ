#import "@preview/unequivocal-ams:0.1.2": ams-article, theorem, proof
#import "@preview/equate:0.2.1": equate // <- for numbering equations
#import "@preview/unify:0.6.1": num // <- for making numbers look nice


#let f_hash = "50115caac50c4fbfa6bce4cc"
#let s_hash = "7c2a10494ff64e66a9af2731"
#let p_hash = "0c848c1444264cbfa1a4de6e"
#let masks_hash = "ba88bfb237924d5091006372"
#let nodro_hash = "c7f717cb50ac4762bd866831"
#show: equate.with(breakable: true, sub-numbering: true)
#set math.equation(numbering: "(1.1)", supplement: "Eq.")
#set raw(align: center)

// Set page margins for the entire document
#show: ams-article.with(
  title: [Mechanistic Interpretability on multi-task Irreducible Integer Identifiers],
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
    This paper investigates how neural networks solve multiple related mathematical tasks simultaneously through mechanistic interpretability. A transformer model is trained on 29 parallel tasks, each predicting remainders when dividing two-digit base-113 numbers by all primes less than 113. This setup spans task complexity from binary classification (division by 2) to 109-way classification (largest prime less than 113).
    Further, the model, trained using gradient filtering to accelerate generalization, achieves perfect accuracy across all tasks. Embedding analysis, singular value decomposition, and Fourier analysis of neuron activations reveal complex internal representations. The model initially solves simpler tasks (modulo 2, 3, 5, and 7) before developing a shared strategy for the remaining tasks.
    The increased number of active frequencies in neuron activations suggests a phase where additional circuits form during generalization, some facilitating learning but not present in the final solution. Findings also confirm that amplifying slow-moving gradients significantly accelerates generalization.

    This study shows that multi-task learning shapes the development of internal mechanisms in deep learning models, offering insights into how these models internalize algorithms across tasks. Future research could automate circuit discovery and explore variations in multi-task learning setups. The project repository is available at https://github.com/syrkis/miiii.
  ],
)






#let appendix(body) = {
  set heading(numbering: "A", supplement: [Appendix])
  counter(heading).update(0)
  body
}

// body ///////////////////////////////////////////////////////////////////////
//
// TODO: LABEL PLOTS OR DIE
// - In intro explain it to a 4 year old baby.
// - Concepts.
// - add tables.

= Introduction

Recent years have seen deep learning (DL) models achieve remarkable proficiency in complex computational tasks, including protein structure prediction @jumper2021, strategic reasoning @dinan2022, and natural language generation @radford2018a—areas previously thought to be the exclusive domain of human intelligence. Traditional (symbolic) programming allows functions like $f(x, y) = cos(a dot x) + sin(b dot y)$ to be implemented in code with clear typographical isomorphism—meaning the code's structure directly mirrors the mathematical notation. For example, in the language Haskell: `f x y = cos(a * x) + sin(b * y)`. In contrast, DL models are inherently sub-symbolic, meaning that the models' atomic constituents (often 32-bit floating-point numbers centered around 0) do not map directly to mathematical vocabulary. For reference, @subsymbolic shows a DL-based implementation of the aforementioned function. Indeed, the increasing prevalence of DL can be understood as a transition from symbolic to sub-symbolic algorithms.

Precursors to modern DL methods learned how to weigh human-designed features @shannon1950, with later works learning to create features from data to subsequently weigh @tesauro1993, @silver2017—in combination with tree search strategies, in the case of games @browne2012. Very recent DL work has even eliminated tree search in the case of chess, mapping directly from observation space to action space @ruoss2024. Pure DL methods are thus becoming
ubiquitous but remain largely inscrutable, with recent works still attempting to define what interpretability even means in the DL context @lipton2018. Given the sub-symbolic nature of DL models, it is unsurprising that their interpretation remains difficult.

Mathematically, DL refers to a set of methods that combine linear maps (matrix multiplications) with non-linearities (activation functions). Formally, all the potential numerical values of a given model's weights $W$ can be thought of as a hypothesis space $cal(H)$. Often, $cal(H)$ is determined by human decisions (number of layers, kinds of layers, sizes of layers, etc.). $cal(H)$ is then navigated using some optimization heuristic, such as gradient descent, in the hope of finding a $W$ that "performs well" (i.e., successfully minimizes some loss $cal(L)$ often computed by a differentiable function with respect to $W$) on whatever training data is present. This vast, sub-symbolic hypothesis space, while enabling impressive performance and the solving of relatively exotic#footnote[Try manually writing a function in a language of your choice that classifies dogs and cats from images.] tasks, makes it challenging to understand how any one particular solution actually works (i.e., a black box algorithm).

The ways in which a given model can minimize $cal(L)$ can be placed on a continuum: on one side, we have overfitting, remembering the training data, (i.e. functioning as an archive akin to lossy and even lossless compression); and on the other, we have generalization, learning the rules that govern the relationship between input and output (i.e. functioning as an algorithm).

When attempting to give a mechanistic explanation of a given DL model's behavior, it necessarily entails the _existence_ of a mechanism.
Mechanistic interpretability (MI) assumes this mechanism to be general, thus making generalization a necessary (though insufficient) condition. Generalization ensures that there _is_ a mechanism/algorithm present to be uncovered (necessity); however, it is possible for that algorithm to be so obscurely implemented that reverse engineering, for all intents and purposes, is impossible (insufficiency). Various forms of regularization are used to incentivize the emergence of algorithmic (generalized) and interpretable, rather than archiving (over-fitted) behavior @ba2016, @krizhevsky2017, @krogh1991.

// As will be covered in @related_works, the mechanistic interpretability (MI) literature has, despite its nascent state, already established some conventions and successes. Circuits solving basic algorithmic tasks have been successfully reverse-engineered @nanda2023, and aspects of this workflow have been automated @conmy2023.

// kinde insecure about this
As of yet, no MI work has explored the effect of multi-task learning, the focus of this paper. Multitask learning also has a regularizing effect @baxter2011. Formally, the set of hypotheses spaces for each task of a set of tasks (often called environment) is denoted by $cal(H) in HH$. When minimizing the losses across all tasks in parallel, generalizing $W$'s are thus incentivized, as these help lower loss across tasks (in contrast to memorizing $W$'s that lower loss for one task). A $W$ derived from a multi-task training process can thus be thought of as the intersection of the high-performing areas of all $cal(H) in HH$.

// This empirical approach to understanding neural networks makes MI more akin to botany than theoretical computer science: while finding an interesting specimen (training a working model on an original task) is relatively straightforward—like strolling a botanical garden, looking for an unstudied flower—carefully dissecting it to understand its internal mechanisms remains challenging and labor-intensive.


// weights $W$ in $cal(H)$ that perform well across tasks are more likely to be general. #cite(<baxter2011>, form:"prose") refers to the set of hypotheses spaces for the different tasks in a given environment of tasks as $cal(H) in HH$. A $W$ performing well across tasks can thus be thought of as the intersection of the hypotheses spaces across $HH$..

In this spirit, the present paper builds on the work of #cite(<nanda2023>, form:"prose", style:"american-psychological-association"), which trains a transformer @vaswani2017 model to perform modular addition, as seen in @nanda_task. The task is denoted as $cal(T)_"nanda"$ throughout the paper.

$
  (x_0 + x_1) mod p, quad forall x_0, x_1 < p, quad p = 113
$<nanda_task>


The task of this paper focuses on predicting remainders modulo all primes $q$ less than $p$, where $x$ is interpreted as $x_0 p^0 + x_1 p^1$, formally shown in @miiii_task, and is referred to as $cal(T)_("miiii")$:


$
  (
    x_0 p^0 + x_1 p^1
  ) mod q, quad forall x_0, x_1 < p, quad forall q < p, quad p = 113
$<miiii_task>

$cal(T)_("miiii")$ differentiates itself from $cal(T)_("nanda")$ in two significant ways: _1)_ it is non-commutative, and _2)_ it is, as mentioned, multi-task. These differences present unique challenges for mechanistic interpretation, as the model must learn to handle both the order-dependent nature of the inputs and develop shared representations across multiple modular arithmetic tasks. Further, as $cal(T)_("miiii")$ is harder than $cal(T)_("nanda")$ the model can be expected to generalize slower when trained on the former. Therefore, #cite(<lee2024a>, form:"prose", style:"american-psychological-association")'s recent work on speeding up generalization by positing the model parameters gradients through time can be viewed as a sum of _1)_ a slow varying generalizing component (which is boosted), and _2)_, a quick varying overfitting component (which is suppressed), is (successfully) replicated to make training tractable.

#figure(
  image("figs/polar.svg", width: 120%),
  caption: [
    Visualizing natural numbers less than #num(12769) in polar coordinates $(n, n mod 2 pi)$. Left: union of numbers with remainder 0 mod 17 and 23 (see the two spirals). Middle: numbers with remainder 0 mod 11. Right: prime numbers.
    It is shown here to encourage the reader to think in periodic terms.
  ],
)<nats>

More generally, modular arithmetic on primes is a particularly useful task for MI as it ensures uniformity among the output classes, allows for comparison with other MI work @nanda2023, and, from a number-theoretic point of view, primes contain mysteries ranging from the trivially solved—are there an infinite number of primes?—to the deceptively difficult—can all even numbers larger than 4 be described as the sum of two primes? The latter, known as Goldbach's Conjecture, remains unsolved after centuries. The choice of using every prime less than the square root of the largest number of the dataset also serves the following purpose: to test if a given natural number is prime, it suffices to test that it is not a multiple of any prime less than its square root—the set of tasks trained for here, can thus be viewed in conjunction as a single prime detection task (primes are the only samples whose target vector contains no zeros, since it is not a multiple of any of the factors $q$). There are about $n/ln(n)$ primes less than $n$.
// To test if a given number $n$ is prime, it is sufficient to test if it is divisible by any prime less than $sqrt(n)$ (Sieve of Eratosthenes), of which there are about $sqrt(n)/ln(sqrt(n))$.
// PASED IN MIGHT NEED WORK WITH FLOW

To provide insight into the periodic structure of these remainders for natural numbers less than #num(12769) (and motivate thinking in rotational terms), @nats visualizes various modular patterns in polar coordinates ($n, n mod 2 pi$). One could imagine tightening and loosening the spiral by multiplying $2 pi$ by a constant to align multiples of a given number in a straight line (imagining this is encouraged).

// Lastly, the reader is asked to accept the inspection of a DL model transitioning from archive to algorithm on multiple simultaneous tasks as inherently interesting, independent of the current literature's scarcity on the subject.

// Lastly, the fact that MI lags so far behind the cutting edge of DL means that the models in which interesting MI is performed are relatively simple to train. The MI workflow is thus perhaps more similar to botany than theoretical computer science. While the models (the specimens) are easy to cultivate, dissecting them to uncover the principles governing their function remains a challenging endeavor. This paper aims to contribute to this effort by exploring the mechanistic interpretability of models trained on multitask modular arithmetic.

= Background and related work<related_works>

// Mechanistic Interpretability as a field is relatively new, though the objects of its study have been seen widespread adoption in the last decade. And indeed,

Multiple papers describe the use of deep learning to detect prime numbers @egri2006, @lee2024, @wu2023a.
None are particularly promising as prime detection algorithms, as they do not provide speedups, use more memory, and are less accurate than traditional methods.
However, in exploring the foundations of deep learning, the task of prime detection is interesting, as it is a simple task that is difficult to learn, and is synthetic, meaning that the arbitrary amounts of data are generated by a simple algorithm.


== Mechanistic Interpretability (MI)<mi>

MI is a relatively new field focused on reverse-engineering the internal mechanisms of neural networks. #cite(<lipton2018>, form: "prose", style:"american-psychological-association") explored different definitions of interpretability in this context. MI can be contrasted with other forms of interpretability, such as feature importance analysis; while feature importance measures correlations between inputs and outputs (e.g., red pixels correlating with "rose" classifications), MI aims to understand how the model actually processes information (i.e., the mechanism).


// TODO MAKE SURE YOU USE TERMINOLOGY LATER !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Methods and tools used so far in MI include: Activation visualization across ordered samples; Singular value decomposition of weight matrices; Ablation studies to identify critical circuits. #cite(<conmy2023>, form: "prose", style:"american-psychological-association") even successfully automate circuit#footnote[In the context of MI, "circuit" refers to a subgraph of a neural network that performs a particular function.] discovery.
Many reverse engineering methods from other fields, such as computational neuroscience or signal processing, almost certainly have their uses here as well.

// #cite(<weiss2021>, form: "prose", style:"american-psychological-association")'s RASP language demonstrates how architectural constraints can be made explicit, helping researchers "think like a transformer" by expressing computation in terms of the architecture's native operations (attention as reduction, MLP as mapping). Current research is being done (CITE GROW AI?) into how to grow and merge neural network models, indicating a potential for composition of networks, with "modules" fulfilling certain properties.

In spite of deep learning's practical successes, uncertainty remains about its theoretical underpinnings, echoing the interpretability debate. Recent work attempts to place different DL architectures and concepts in a either geometric @bronstein2021, information theoretic @yu2021, or even category theoretic @gavranovic2024 context. However, no unified theory has emerged. Much interesting deep learning research thus focuses on practical, simple, or algorithmic tasks with known solutions and architectures. For example, grokking @power2022, the (relatively) sudden generalization after overfitting, as elaborated later, is a recent and practical discovery.

=== Case study: modular addition

One such practical discovery is made by #cite(<nanda2023>, form: "prose", style:"american-psychological-association").
A single layer transformer model with ReLU activation function was trained to perform modular addition ($cal(T)_"nanda"$).
#cite(<nanda2023>, form: "prose", style:"american-psychological-association")'s analysis of their trained model exemplifies MI methodology. They discovered that: _1)_ The embedding layer learns trigonometric lookup tables of sine and cosine values as per @nanda_layer_1; _2)_ The feed-forward network combines these through multiplication and trigonometric identities (@nanda_mul), and _3)_ The final layer performs the equivalent of argmax (@nanda_last_layer).

$
  x_0 -> sin(w x_0), cos(w x_0) \
  x_1 -> sin(w x_1), cos(w x_1) \
$<nanda_layer_1>

$
  sin(w(x_0 + x_1)) = sin(w x_0) cos(w x_0) + cos(w x_0) sin(w x_1)\
  cos(w(x_0 + x_1)) = cos(w x_1) cos(w x_1) - sin(w x_0) sin(w x_1)
$<nanda_mul>


$
  "Logit"(c) &prop cos(w(x_0 + x_1 - c)) \
  &= cos(w(x_0 + x_1)) cos(w c) + sin(w(x_0 + x_1)) sin(w c)
$<nanda_last_layer>


// The observation that networks can transition from memorization to algorithmic solutions raises several questions:
// 1. Can we bypass the memorization phase?
// 2. What determines which algorithms emerge?
// 3. How does multi-task learning affect algorithm discovery?
// 4. How can memorization and algorithmic computation coexist?
// 5. What other tricks than slow gradient boosting @lee2024a can speed up grokking?
// 6. If a circuit is general, can we make proofs about the model's function?

// These questions are central to both theoretical understanding and practical applications of deep learning.



== Generalization and grokking

#cite(<power2022>, form: "prose", style:"american-psychological-association") shows generalization can happen #quote(attribution: cite(<power2022>), "[...] well past the point of overfitting"), dubbing the phenomenon "grokking". The phenomenon is now well established @nanda2023, @humayun2024, @wang2024. #cite(<nanda2023>, form: "prose", style:"american-psychological-association") shows that a generalized circuit #quote(attribution: cite(<nanda2023>), "arises from the gradual amplification of structured mechanisms encoded in the weights,") rather than being a relatively sudden and stochastic encounter of an appropriate region of $cal(H)$. The important word of the quote is thus "gradual".

By regarding the series of gradients in time as a stochastic signal, #cite(<lee2024a>, form:"prose", style:"american-psychological-association") proposes decomposing the signal. Conceptually, #cite(<lee2024a>, form:"prose", style:"american-psychological-association") argues that in the case of gradient descent, the ordered sequence of gradient updates can be viewed as consisting of two components: _1)_ a fast varying overfitting component, and _2)_ a slow varying generalizing components. The general algorithm explaining the relationship between input and output is the same for all samples, whereas the weights that allow a given model to function are unique for all samples. Though not proven, this intuition bears out in that generalization is sped up fifty-fold in some cases.

This echoes the idea that generalized circuits go through _gradual_ amplification @nanda2023. To the extent that this phenomenon is widespread, it bodes well for generalizable DL in that the generalizing signal that one would want to amplify might exist long before the model is fully trained and could potentially be boosted in a targeted way by the method described by #cite(<lee2024a>, form: "prose", style:"american-psychological-association").

Perhaps the most widespread loss functions used in deep learning are mean cross-entropy @mce (for classification) and mean squared error @mse (for regression).

$
  L_("MCE") &= 1 / n sum_(i=1)^n sum_(j=1)^k y_p_(i j) ln(1 / hat(y)_p_(i j))#<mce> \
  L_("MSE") &= 1 / n sum_(i=1)^n (y_i - hat(y)_i)^2 #<mse>
$

These have various computational and mathematical properties that make them convenient to use, though they have been shown to struggle with generalizing out-of-distribution data @jeon2022, @yu2021.


== Multi-task learning in deep learning

// #cite(<baxter2011>, form:"prose", style:"american-psychological-association") further extends the notion of generalization and training to a multitask paradigm.
As stated, multi-task learning has been shown to have a regularizing effect @baxter2011, @maurer as the hypothesis $W$ that performs well across all of the hypothesis spaces $cal(H) in HH$ is more likely to be general. Viewed information theoretically, this concept is reminiscent of #cite(<shannon2001>, form:"prose", style:"american-psychological-association")'s asymptotic equipartition property @cover2006, or even more generally, the law of large numbers, which states that the more samples we have of a distribution, the closer our _estimates_ are to its underlying properties will align with the _true_ underlying properties.

In the context of $cal(T)_"miiii"$, multi-task learning is done by having the last layer output predictions for all tasks in parallel. Thus, whereas $cal(T)_("nanda")$ outputs a single one-hot $1 times 113$ vector for each of the potential remainders, $cal(T)_("miiii")$, as we shall see, outputs a $1 times q$ vector for each prime $q < p$ (i.e., 29 output-task vectors when $p=113$). The embeddings layer and the transformer block are thus shared for all tasks, meaning that representations that perform well across tasks are incentivized.

// Due to its prevalence, however, MCE is chosen in this paper. However, since we have multiple tasks, the MCE is modified as shown in


// Prime numbers, in particular, are an interesting domain for deep learning. A frequent feature of number theoretical problems is the ease with which they can be stated. This is true for trivial problems (such as proving there are infinitely many primes) and deceptive problems (such as "all even numbers can be expressed as the sum of two primes"). The latter, known as Goldbach's conjecture, remains unsolved.

== Transformer architecture

Transformers combine self-attention (a communication mechanism) with feed-forward layers (a computation mechanism).
The original transformer-block @vaswani2017 used extensive regularization—layer norm @ba2016,
dropout, weight decay, and residual connections are all integral components of the original architecture,
though recent years have seen simplifications yielding similar performance @he2023, @hosseini2024.

Input tokens are embedded into a $d$-dimensional space using learned token and positional embeddings:
$
  z = "TokenEmbed"(x) + "PosEmbed"("pos")
$<embed>

Each transformer block comprises multi-head attention:
$
  "Attention"(Q, K, V) = "softmax"(Q K^T / sqrt(d_k))V
$<attn>
where $Q$, $K$, and $V$ are linear projections of the input. Attention heads are combined through addition rather than concatenation (a transformer specific detail to align with #cite(<nanda2023>, form: "prose", style:"american-psychological-association")). This is followed by a feed-forward network with ReLU activation:
$
  "FFN"(z) = "ReLU"(z W_("in"))W_("out")
$<ffwd>
mapping from $d$ → $4d$ → $d$ dimensions, before finally:

$
  hat(y) = z W_("unembed")
$<output>

Each component includes residual connections and dropout.


// Algorithms, like gradient descent, are heuristics for finding optimal/optimized values of $cal(W)$ within $cal(H)$. $H$ itself is not modified by optimization algorithms of this level (i.e. optimizing $a x+b$ yields optimal $a "and" b$ values, but we might need a more complex model to describe given phenomena).



= Methods

How exactly a given model implements an algorithm is a non-trivial question—even modular addition is implemented in a relatively obscure way @nanda2023, as per @nanda_layer_1, @nanda_mul, and @nanda_last_layer.

This investigation probes the fundamental algorithmic structures internalized by a transformer model trained on a set of basic prime number-related modular arithmetic tasks with slight variations in complexity. This approach provides insights into how and why specific algorithmic patterns emerge from seemingly straightforward learning processes.

As stated, the setup here differentiates itself from $cal(T)_"nanda"$ in two crucial ways: _1)_ It is non-commutative; and _2)_ It is multitask.

// A deep learning model, $cal(M)$, consists of a set of model weights $cal(W)$ and a procedure on how to apply these to a given input $cal(X)$. Viewed in the context of the procedure, the set of potential values of $cal(W)$ can be thought of as a hypothesis space $cal(H)$ on the mapping between $cal(X)$ and $cal(Y)$, regarding a loss function $cal(L)$.

== Tasks

Stated plainly: the task $cal(T)_"miiii"$ predicts the remainder when dividing a two-digit base-$p$ number by each prime factor $q$ less than $p$. The set of prime factors we construct tasks for is thus ${q} = {q in PP : q < p}$.
For $p=113$, this yields 29 parallel tasks, one for each prime less than $p$. Each task predicts a remainder in the range $[0, q-1]$. This means smaller primes like 2 and 3 require binary and ternary classification, respectively, while the largest prime less than $p$, 109, requires predictions across 109 classes. The tasks thus naturally vary in difficulty: predicting $mod 2$ requires distinguishing odd from even numbers (which in binary amounts to looking at the last bit) while predicting $mod 109$ involves making a selection between many relatively similar classes. From an information-theoretical perspective, the expected cross entropy for an $n$-class problem is $ln(n)$, which has implications for the construction of the loss function, further discussed in @training.

Additionally, a baseline task $cal(T)_"basis"$ was constructed by shuffling the $y$-labels of $cal(T)_"miiii"$, and a task ablation test $cal(T)_"masked"$ was constructed by masking away the four simplest tasks $q in {2,3,5,7}$.



== Data

*Input Space ($X$)*
Each input $x in X$ represents a number in base $p$ using two digits, $(x_0,x_1)$, where the represented number is $x_0 p^0 + x_1 p^1$. For example, with $p=11$, the input space consists of all pairs $(x_0,x_1)$ where $x_0,x_1 < 11$, representing numbers up to $11^2-1 = 120$. This yields a dataset of 121 samples. @miiii_x_11 visualizes this input space, with each cell representing the value $x_0 p^0 + x_1 p^1$.

#figure(
  image("figs/x_11_plot.svg", width: 110%),
  caption: [Visualizing X (for a small dataset where $p=11$). Each cell represents the tuple $(x_0, x_1)$. The top left shows 0 $(0,0)$, and the bottom right shows 120 $(10,10)$—both in base-11],
)<miiii_x_11>

*Output Space ($Y$)*
For each input $x$, a vector $y in Y$ contains the remainder when dividing by each prime less than $p$. For $p=11$, this means predicting the remainder when dividing by 2, 3, 5, and 7. Each element $y_i$ ranges from $0$ to $q_i-1$ where $q_i$ is the $i$-th prime. @miiii_y_11 visualizes these remainders, with each subplot showing the remainder pattern for a specific prime divisor. For comparison, the rightmost plot shows the output space of @nanda2023's modular addition task.

#figure(
  image("figs/y_11_plot.svg", width: 120%),
  caption: [Visualizing tasks in Y (for $p=11$). $x_0$ and $x_1$ vary on the two axis, with the remainder modulo $q in {2,3,5,7}$ indicated by the square size. Note the innate periodicity of the modulo operator.],
)<miiii_y_11>

== Model

// Architectural decisions are made to align with #cite(<lee2024a>, form:"prose", style:"american-psychological-association") and #cite(<nanda2023>, form:"prose", style:"american-psychological-association").
The model follows the original transformer architecture @vaswani2017 with several key design choices aligned with recent work on mechanistic interpretability @nanda2023, @lee2024a: biases are disabled, and layer normalization is not used. The model consists of three main components: an embedding layer, transformer blocks, and an output layer. All weights are initialized following #cite(<he2015>, form: "prose", style:"american-psychological-association"). The model processes vectors of the kind seen in @vec, writing the eventual result to the last position.

// where $W_("out")$ projects to $sum_(i=1)^k f_i$ dimensions for $k$ prime factors, with $q_i$ being the $i"th"$ prime less than $p$.


$
  mat(delim:"[", quad x_0 quad x_1 quad hat(y) quad)
$<vec>

== Training<training>

Hyper parameter optimization was conducted using Optuna @akiba2019, searching over @hyper_param_search_space.


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

The model is trained using AdamW @loshchilov2019 with $beta_1=0.9$, $beta_2=0.98$ following #cite(<nanda2023>, form: "prose", style:"american-psychological-association"). To handle the varying number of classes across tasks (from 2 classes for mod 2 to 109 classes for mod 109), a modified (weighted) mean cross-entropy (@mce) loss is created, correcting for the difference in the expected loss within each task. Note that $EE[L_("MCE")] = ln(1/q)$, where $q$ is the number of classes within the task in question. Correcting for this, the loss function becomes as shown in @mmce.


$
  L_(cal(T)_"miiii") &= &&sum_(q in {q}) L_"MCE"_q / (ln(q)) \
  &=  &&sum_(q in {q}) (sum_(i=1)^n sum_(j=0)^(q-1) y_(q i j) ln(hat(y)_(q i j)) ) / ( n ln(q)) \
  &= &&sum_(q in {q})sum_(i=1)^n sum_(j=0)^(q-1) (y_(q i j)ln(hat(y)_(q i j)) ) / (n ln(q)) #<mmce>
$


To accelerate generalization, gradient filtering as per #cite(<lee2024a>, form:"prose", style:"american-psychological-association") is implemented and replicated.

$
  g_t = nabla_theta L + lambda(alpha e_(t-1) + (1-alpha)g_(t-1))
$<grad>

where $e_t$ is the exponential moving average of gradients with decay rate $alpha=0.98$, and $lambda$ controls the influence of the slow-varying component.

The training uses full batch gradient descent with the entire dataset of $p^2$ samples (#num(12769) when $p=113$). The model is evaluated on a held-out validation set after each epoch, tracking per-task accuracy and loss. As the setup used in $cal(T)_"nanda"$, training was done on thirty percent of the total dataset, with the remaining used for validation (1000 samples) and testing (remaining). Further, as $cal(T)_"miiii"$ involves the learning of 29 (when $p=113$) tasks rather than 1, and due to each task's non-commutativity, a larger hidden dimension of 256 was added to the hyper parameter search space, as well as the potential for 8 heads ($cal(T)_"nanda"$ was solved with a hidden dimension of 128, and 4 heads). The number of transformer blocks was kept at 1, as this ensures consistency with $cal(T)_"nanda"$ (and as full generalization was possible, as we shall see in the @results).

Training was done on a NVIDIA GeForce RTX 4090 GPU, with Python3.11 and extensive use of "JAX 0.4.35" and its associated ecosystem.
Neuron activations were calculated at every training step and logged for later analysis.

== Visualization

Much of the data worked with here is inherently high dimensional. For training, for example, we have $n$ steps, two splits (train/valid) about $p/ln(p)$ tasks, and two metrics (accuracy and loss). This, along with the inherent opaqueness of deep learning models, motivated the development of a custom visualization library, `esch`#footnote[https://github.com/syrkis/esch], to visualize attention weights, intermediate representations, training metrics, and more. To familiarize the reader with visualizing the inner workings of a trained model, an essential plot type for the reader to keep in mind is seen in @plot_example. As there are only #num(12769) samples when $p=113$, all samples can be fed at once to the model. Inspecting a specific activation thus yields a $1 times$ #num(12796) vector $v$, which can be reshaped as a $113 times 113$ matrix, with the two axes, $x_0$ and $x_1$, varying from 0 to 112, respectively. The top-left corner then shows the given value for the sample $(0 dot p^0 + 0 dot p^1)$, and so on.


#figure(
  align(
    horizon,
    stack(
    dir: ltr,
    spacing: 0pt, // Adjust this value to control the spacing
    image("figs/neurs_113_miiii_one.svg", width: 40%),
    image("figs/neurs_113_miiii_fft_one.svg", width: 46%),
  ),
  ),
  caption: [Plotting a neuron: (left) The activation of a particular neuron as $x_0$ and $x_1$ varies from $0$ to $p$. (right) The same processed with a fast Fourier transform to active frequencies ($omega$).],
)<plot_example>

Note that in `esch` plots, when appropriate, only the top leftmost $37 times 37$ slice is shown so as not to overwhelm the reader.

== Mechanistic interpretability process

Recall that a combination of linear products is itself a linear product. Therefore, as a mechanistic interpretability rule of thumb, one should look at the outputs of the non-linear transformations. In our case, that will be the attention weights and the intermediate representations within the transformer block's feed-forward output (which follows the ReLU activation).
Additionally, the embedding layers will be inspected using Fourier analysis and singular value decomposition. As mentioned in @mi, our interpretability approach combines activation visualization with frequency analysis to understand the learned algorithmic patterns. Following #cite(<nanda2023>, form: "prose", style:"american-psychological-association"), we analyze both the attention patterns and the learned representations through several lenses:

=== Attention visualization

Using `esch`, the custom visualization library, to visualize attention weights and intermediate representations. The library allows for the visualization of attention patterns across different layers, as well as the visualization of intermediate representations at each layer. These visualizations provide insights into the learned patterns and help identify potential areas of improvement.

=== The fast Fourier transform

As periodicity is established by #cite(<nanda2023>, form: "prose", style:"american-psychological-association") as a fundamental feature of the model trained on $cal(T)_"nanda"$, the fast Fourier transform (FFT) algorithm is used to detect which frequencies are in play.
Note that any square image can be described as a sum of 2d sine and cosine waves varying in frequency from 1 to the size of the image divided by 2 (plus a constant).
This is a fundamental tool used in signal processing. The theory is briefly outlined in @fft for reference.
This analysis helps identify the dominant frequencies in the model's computational patterns.
Recall that a vector can be described as a linear combination of other periodic vectors as per the discrete Fourier transform.

The default basis of the one-hot encoded representation of the input is thus the identity matrix. This can be projected into a Fourier basis by multiplying with the discrete Fourier transform (DFT) matrix visualized in @dft.



= Results and analysis<results>

== Hyper-parameter optimization

The best-performing hyper-parameters for training the model on $cal(T)_"miiii"$ are listed in @hyper_param_search_result. Notably, the model did not converge when $lambda = 0$, confirming the utility of the gradient amplification method proposed by #cite(<lee2024a>, form:"prose", style:"american-psychological-association") in the context of $cal(T)_"miiii"$.


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

    $1 / 10$, $1 / 2$, $1 / 3$, "256", $3 times 10^(-4)$, "4",
  ),
  caption: [Result of hyper-parameter search over $cal(T)_"miiii"$.],
)<hyper_param_search_result>


== Model Performance

@trainig_acc show the training and validation accuracy on $cal(T)_"miiii"$ over time. The model achieves a perfect accuracy of 1 on the validation set across all 29 tasks. The cross-entropy loss in @training_loss echoes this. In short—and to use the terminology of #cite(<power2022>, form:"prose", style:"american-psychological-association")—the model "grokked" on all tasks.
Interestingly, tasks corresponding to modulo 2, 3, 5, and 7 generalized in succession, while the remaining 25 tasks generalized around epoch #num(40000) in no particular order. This might suggest that the model initially learned solutions for the simpler tasks and later developed a more general computational strategy that allowed it to generalize across the remaining, more complex tasks.

#figure(
  stack(
    dir: ttb,
    image("figs/" + f_hash + "/acc_train_training.svg", width: 85%),
    image("figs/" + f_hash + "/acc_valid_training.svg", width: 85%),
  ),
  caption: [Accuracy training "curves": Training (top) and validation (bottom) accuracy over time ($x$-axis in log-scale). We see grokking occur on all tasks, first for $q in {2,3,5,7}$ in that order, and then the remaining 25 in no particular order.],
)<trainig_acc>

#figure(
  stack(
    dir: ttb,
    image("figs/" + f_hash + "/loss_train_training.svg", width: 85%),
    image("figs/" + f_hash + "/loss_valid_training.svg", width: 85%),
  ),
  caption: [Cross-entropy (@mce) loss on training (top) and validation (bottom) over time (note the log scale on the $x$-axis).],
)<training_loss>












== Embeddings<emb_section>

Positional embeddings play a crucial role in transformers by encoding the position of tokens in a sequence. @pos_emb compares the positional embeddings of models trained on $cal(T)_"nanda"$ and $cal(T)_"miiii"$.

For $cal(T)_"nanda"$, which involves a commutative task, the positional embeddings are virtually identical, with a Pearson correlation of 0.95, reflecting that the position of input tokens does not significantly alter their contribution to the task. In contrast, for $cal(T)_"miiii"$, the positional embeddings have a Pearson correlation of -0.64, indicating that the embeddings for the two positions are different. This difference is expected due to the non-commutative nature of the task, where the order of $x_0$ and $x_1$ matters ($x_0 dot p^0 != x_0 dot p^1$). This confirms that the model appropriately encodes position information for solving the tasks.

#figure(
  image("figs/pos_emb.svg"),
  caption: [Positional embeddings for $(x_0, x_1)$ for models trained on $cal(T)_"nanda"$ (top) and $cal(T)_"miiii"$ (bottom). Pearson's correlation is 0.95 and -0.64 respectively. This reflects the commutativity of $cal(T)_"nanda"$ and the lack thereof for $cal(T)_"miiii"$. Hollow cells indicate negative numbers.],
)<pos_emb>

// == Token Embeddings

Recall that a matrix $upright(bold(M))$ of size $m times n$ can be decomposed to its singular values $upright(bold(M)) = upright(bold(U))upright(bold(Sigma))upright(bold(V^T))$ (with the transpose being the complex conjugate when $upright(bold(M))$ is complex), where $upright(bold(U))$ is $m times m$, $upright(bold(Sigma))$ an $m times n$ rectangular diagonal matrix (whose diagonal is represented as a flat vector throughout this paper), and $upright(bold(V^T))$ a $n times n$ matrix. Intuitively, this can be thought of as rotating in the input space, then scaling, and then rotating in the output space.

@s displays the singular values of the token embeddings learned for $cal(T)_"nanda"$ and $cal(T)_"miiii"$. The singular values for $cal(T)_"miiii"$ are more diffuse, indicating that a larger number of components are needed to capture the variance in the embeddings compared to $cal(T)_"nanda"$. This suggests that the token embeddings for $cal(T)_"miiii"$ encode more complex information, reflecting the increased complexity of the multi-task learning scenario.

#figure(
  stack(
    image("figs/nanda_S.svg"),
    image("figs/miiii_S.svg"),
  ),
  caption: [First 83 of 113 singular values (truncated for clarity) of $upright(text(U))$ for $cal(T)_"nanda"$ (top) and $cal(T)_"miiii"$ (bottom). The ticks indicate the points where 50% and 90% of the variance is accounted for. We thus see that for $cal(T)_"miiii"$, the embedding space is much more crammed.],
)<s>

#figure(
  image("figs/nanda_U.svg"),
  caption: [$cal(T)_"nanda"$'s most significant (cutoff at 0.5 as per @s) singular vectors of $upright(text(U))$ from the singular value decomposition. Note this looks periodic!],
)<p_U>

#figure(
  image("figs/miiii_U.svg"),
  caption: [$cal(T)_"miiii"$'s most significant vectors of $upright(text(U))$. Note that, like in @p_U, we still observe periodicity, but there are more frequencies in play, as further explored in @miiii_fft.],
)<f_U>

@p_U and @f_U present the most significant singular vectors of $upright(text(U))$ for $cal(T)_"nanda"$ and $cal(T)_"miiii"$, respectively. Visual inspection shows periodicity in the top vectors for both models, but the $cal(T)_"miiii"$ model requires more vectors to capture the same amount of variance, consistent with the diffuse singular values observed in @s.


To further understand the structure of the token embeddings, we applied the Fast Fourier Transform (FFT). Only a few frequencies are active for $cal(T)_"nanda"$ as seen in @nanda_fft, consistent with the model implementing a cosine-sine lookup table as described in #cite(<nanda2023>, form: "prose", style:"american-psychological-association").

For the $cal(T)_"miiii"$ model, we observe a broader spectrum of active frequencies (@miiii_fft). This is expected due to the model having to represent periodicity corresponding to 29 primes.

Comparing with $cal(T)_"basis"$ in figure @baseline_fft, the periodicity is understood to be a structure inherent to the data picked up by the model.



#figure(
  stack(
    dir: ttb,
    image("figs/fourier_nanda_m.svg"),
    image("figs/fourier_nanda_f.svg"),
  ),
  caption: [$cal(T)_"nanda"$ tokens in Fourier basis: Note how all tokens are essentially linear combinations of the five most dominant Fourier basis vectors. The sparsity echoes the findings in @s that very few directions in the embedding space are used.],
)<nanda_fft>


#figure(
  stack(
    dir: ttb,
    image("figs/fourier_miiii_m.svg"),
    image("figs/fourier_miiii_f.svg"),
  ),
  caption: [The periodicity in the $cal(T)_"miiii"$ embeddings involves a much larger fraction of the Fourier basis, echoing the multiple tasks and their innate difference in frequency (recall that all tasks are performed on unique primes $q$).],
)<miiii_fft>


#figure(
  stack(
    image("figs/fourier_basis_m.svg"),
    image("figs/fourier_basis_f.svg"),
  ),
  caption: [Embeddings for $cal(T)_"basis"$ in Fourier basis have no periodicity. The periodicity is indeed an artifact of the modulo operator.],
)<baseline_fft>




== Analysis of Neuron Activations and Frequencies<neuron_section>

To understand the internal mechanisms developed by the model, we analyzed the neuron activations after the output weight matrix $W_"out"$ for the model trained on $cal(T)_"miiii"$. @miiii_neurons shows that these activations exhibit periodic patterns with respect to $(x_0, x_1)$. This periodicity aligns with the modular arithmetic nature of the tasks, mirroring #cite(<nanda2023>, form:"prose", style:"american-psychological-association") ($cal(T)_"nanda"$).

#figure(
  stack(
    dir: ttb,
    image("figs/neurs_113_miiii_three.svg"),
    image("figs/neurs_113_miiii_fft_three.svg"),
  ),
  caption: [We plot the activation of the first three neurons of the activations immediately following ReLU in @ffwd as $x_0$ and $x_1$ vary (top). Note we only show the top $37 times 37$ corner of the full $133 times 113$ sample matrix. Here too we see periodicity, confirmed by a Fourier transform (bottom). Neurons are reactive to highly particular frequencies in their input domains.],
)<miiii_neurons>

For comparison, @basis_neurons shows the neuron activations for a model trained on $cal(T)_"basis"$. These activations do _not_ exhibit periodicity, confirming that the observed periodic patterns in the models trained for $cal(T)_"miiii"$ and $cal(T)_"nanda"$, too, are indeed a result of the modulo operations inherent in the tasks.

#figure(
  stack(
    dir: ttb,
    image("figs/neurs_113_basis.svg"),
    image("figs/neurs_113_basis_fft.svg"),
  ),
  caption: [Neuron activations for model trained on $cal(T)_"basis"$. As in @baseline_fft, no periodicity is observed for the baseline.],
)<basis_neurons>

The analysis of active frequencies _through training_ using the Fast Fourier Transform (FFT) is illustrated in @finding, with the core findings showing a spike in frequency activation around epoch #num(16384) visible in @tableeeee. The top plot shows the different frequencies of the transformer block's feed-forward neurons evolving as the model learns. The bottom plot displays the variance of frequency activations and the number of frequencies exceeding a significance threshold $omega > mu + 2 sigma$ (i.e., which spots like the ones of the bottom row of @miiii_neurons are active). Initially, a handful of frequencies become dominant as the model generalizes on the first four tasks. As training progresses and the model begins to generalize on the remaining tasks, more frequencies become significant, suggesting that the model is developing more complex internal representations to handle the additional tasks.

#figure(
  stack(
    dir: ttb,
    image("figs/miiii_large_finding.svg"),
    image("figs/miiii_small_finding.svg"),
  ),
  caption: [Top: Frequency dominance is averaged over neurons through training (average activation of frequencies as shown in @miiii_neurons). We see four distinct phases: _1)_ no significant frequencies, _2)_ frequencies gradually emerging, _3)_ a wall of frequencies, and _4)_ frequency count similar to phase 2. Bottom: total number of active frequencies at the corresponding time step. A frequency $omega$ is active when $omega > mu + 2 sigma$ (a frequently used signal processing default).],
)<finding>


#figure(
  table(
    columns: (1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: center,
    table.header(
      "epoch",
      "256",
      "1024",
      "4096",
      "16384",
      "65536",
    ),

    $|omega|$, $0$, $0$, $10$, $18$, "10",
  ),
  caption: [Number of active frequencies on $cal(T)_"miiii"$ over epochs.],
)<tableeeee>

// However, we observe that significant frequencies appear as generalization has occurred, which may suggest the presence of another phase following grokking in the context of multi-task learning. This could be indicative of circuit merging or the integration of task-specific circuits into a more general solution.

@l2_norms shows the L2 norms of gradients through time for the different weight matrices of the model trained on $cal(T)_"miiii"$. The gradient norms provide insights into how different parts of the model are being updated during training. Like with #cite(<nanda2023>, form:"prose", style:"american-psychological-association"), the attention layer converges quickly, echoing their finding that it does not contribute much to solving their modular arithmetic task.

#figure(
  image("figs/grads_norms_miiii.svg"),
  caption: [L2 norms of gradients over time for the different weight matrices of the model trained on $cal(T)_"miiii"$.
    Row order corresponds to how deep in the model the weight is used.
    This shows when during training what parts of the model are updated. `e.*` are embeddings @embed, `a.*` attention layer weights @attn, and `w.*` weights of the feed-forward module @ffwd (`e.u` being the final un-embedding layer.)],
)<l2_norms>

These results demonstrate that more frequencies are involved when training on $cal(T)_"miiii"$ compared to $cal(T)_"nanda"$. The increased frequency components reflect the need for the model to encode multiple periodic patterns corresponding to the various modular arithmetic tasks.

Combining the analysis of embeddings and the transformer block neurons, we see that:
1. A lot more frequencies are in play for $cal(T)_"miiii"$ than in $cal(T)_"nanda"$.
2. Neurons remain highly reactive to a very small set of frequencies.
3. The periodicity is an artifact of the modulo group by analysis of $cal(T)_"basis"$


== Attention Patterns

@attention_heads shows that, in contrast to the model trained on $cal(T)_"nanda"$, where attention heads may focus jointly on both input tokens in a periodic fashion, the attention heads for the $cal(T)_"miiii"$ model focus exclusively on one digit or the other. This behavior could be due to the non-commutative nature of the task, where the position of each digit significantly affects the outcome. #cite(<nanda2023>, form: "prose", style:"american-psychological-association") concludes that the attention mechanism does not contribute significantly#footnote[Neel Nanda has also stated (in a YouTube video) that a multi layer perceptron rather than a transformer-block would probably have been more appropriate for his setup. The transformer is used here due to the non-commutativity of $cal(T)_"miiii"$, and to stay close to Nanda's work.] to the solving of $cal(T)_"nanda"$, and will thus also not be explored further here.

#figure(
  stack(
  image("figs/miiii_wei.svg", width: 110%),  // TODO insert nanda attention, and comment that this might be due to comutativty.
  image("figs/nanda_wei.svg", width: 110%),  // TODO insert nanda attention, and comment that this might be due to comutativty.
),
  caption: [Attention from $hat(y)$ to $x_0$ for the four attention heads in the $cal(T)_"miiii"$ model. The attention heads tend to focus on one digit, reflecting the non-commutative nature of the task.],
)<attention_heads>

Overall, our results demonstrate that the model effectively learns to solve multiple modular arithmetic tasks by developing internal representations that capture the periodic nature of these tasks. The analysis of embeddings and neuron activations provides insights into how the model generalizes from simpler to more complex tasks, possibly through the reuse and integration of learned circuits. Interestingly, as will be discussed in @discussion, there are four significant shifts in the number of frequencies active in the neurons.


Lastly, as can be seen in @bad_training_acc, when dropout was disabled, the model's performance diverged on the validation set (overfitting), even with other kinds of regularization (multi-task, l2, etc.). @masked_tasks shows the accuracy through training for $cal(T)_"masked"$. The masking of $q in {2,3,5,7}$, does _not_ delay grokking on the remaining tasks notably, the spiking after generalization to easy tasks ($q in {11, 13, 17, 19}$) remains.


= Discussion<discussion>

Recall that, when viewed individually, the sub-tasks of $cal(T)_"miiii"$ differ from $cal(T)_"nanda"$ only in commutativity (making the task harder) and in prime since $q<p$ (making the task easier for smaller $q$'s like ${2, 3, 5, 7}$—though less so as $q$ approaches $p$). @pos_emb indicates that the model learns to account for the commutativity by using positional embeddings.

During training, the changes in the number of active neuron frequencies $omega$ in the feed-forward layer (see @finding) echo the loss and accuracy progression seen in @trainig_acc and @training_loss.
Further, as the model groks on the primes $q in {2, 3, 5, 7}$, a handful of frequencies become dominant,
similar to the original model trained on $cal(T)_"nanda"$.

We thus have that: _1)_ the model learns to correct for commutativity, and _2)_ the model is marred by periodicity as per both the token embedding (@emb_section) and feed-forward layer analysis (@neuron_section). Combining these facts, we can assume the learned mechanism to be extremely similar (and perhaps identical when ignoring commutativity) to the one outlined by #cite(<nanda2023>, form:"prose", style:"american-psychological-association") in @nanda_layer_1, @nanda_mul, and @nanda_last_layer.

As the remaining 25 tasks are learned, we see a temporary spike in the number of active frequencies, disappearing again as the model generalizes fully (reaches perfect accuracy).
The observation that the number of active frequencies before and after the remaining 25 tasks are learned are the same indicates a reuse of circuitry.
However, the fact of the spike suggests that additional circuits form during the generalization process.
Viewed in the context of #cite(<lee2024a>, form:"prose", style:"american-psychological-association")'s method for boosting slow-varying generalizing components a question emerges: are there circuits that only facilitate the development of generalization, but are not present in the generalized mechanisms? Real life gives plenty of examples of phenomena that make itself obsolete (e.g., a medicine that fully eradicates an illness and is thus no longer needed). Viewed this way, the spike suggests we might divide the set of circuits in two: _1)_; those useful for the mechanism, and _2)_, those useful for _learning_ the mechanism.


= Future work

A logical next step would be to explore the validity of the notion that some circuits help learning and others help solve the problem. This might yield insight on how to improve #cite(<lee2024a>, form: "prose", style:"american-psychological-association")'s grokking speedup heuristic. Aspects of the circuit discovery workflow could be automated with the methods outlined by #cite(<conmy2023>, form:"prose", style:"american-psychological-association").


Additionally, making variations on $cal(T)_"miiii"$ is also likely to be a good avenue for discovery: Divisibility rather than remainder could be predicted; Experiments training for more epochs could be conducted with larger values of $p$; A more in depth mapping of the shared circuitry could be done, for example attempting to see what types of ablations break which tasks—for example, how can performance be degraded on one grokked task, without affecting the others?.

The code associated with this paper is available as a PyPI package (`pip install miiii`) to facilitate the exploration of these questions (as well as replication of the findings at hand).


= Conclusion

This paper explores the impact of multi-task learning on mechanistic interpretability by training a transformer model on a non-commutative, multi-task modular arithmetic problem $cal(T)_"miiii"$. The model successfully generalizes across all tasks, learning complex internal representations that capture the unique periodic nature of modular arithmetic across multiple primes. Analysis reveals that while the model reuses and integrates circuits for simpler tasks, additional circuits may form during training to facilitate generalization to more complex tasks.

These findings highlight that multi-task learning influences the emergence and complexity of internal mechanisms, posing challenges for mechanistic interpretability but also offering insights into how models internalize algorithms. Understanding these dynamics is important for advancing interpretability and reliability in deep learning systems. Future work includes exploring the distinction between circuits that aid in learning versus those that contribute to the final mechanism/solution and investigating how variations in task design impact the development of internal representations. Advancing the understanding of how deep learning models handle multiple tasks contributes to the broader goal of making these models more interpretable and reliable.

#bibliography("zotero.bib")

#pagebreak()

#appendix[
  #heading(level: 1, "Appendix", numbering: none)

  = First four tasks masked<masked_tasks>

  The phenomenon of learning 4 tasks, then spiking, and then learning all tasks is also present when masking away $q in {2,3,5,7}$, though more subtly.

  #figure(
    stack(
      dir: ttb,
      image("figs/" + masks_hash + "/acc_train_training.svg"),
      image("figs/" + masks_hash + "/acc_valid_training.svg"),
    ),
    caption: [Accuracy when ablating $q in {2,3,5,7}$. Note the model now seems to learn tasks $q in {11, 13, 17, 19}$ first.],
  )
  #figure(
    image("figs/masks_large_finding.svg"),
    caption: [Active frequencies $omega$ during training. We see that the spiking is more subtle and that the model now learns tasks $q in {11, 13, 17, 19}$, before then generalizing to all tasks],
  )


  #pagebreak()



  // = Training plots<training_plots>

  // // Training plot of model trained on $cal(T)_"miiii"$ without dropout.

  // == Training loss
  // #figure(
  //   stack(
  //     dir: ttb,
  //     image("figs/" + f_hash + "/loss_train_training.svg"),
  //     image("figs/" + f_hash + "/loss_valid_training.svg"),
  //   ),
  //   caption: [Representation of training and validation loss ($x$-axis is in log scale).],
  // )



  // #figure(
  //   stack(
  //     image("figs/41e20b3a4790402f8a5be458/loss_train_training.svg"),
  //     image("figs/41e20b3a4790402f8a5be458/loss_valid_training.svg"),
  //   ),
  //   caption: [Loss for run with dropout disabled],
  // )


  // #pagebreak()


  = Fast Fourier Transform<fft>

  The inner product between two vectors $bold(v) "and" bold(u)$ of length $n$ can be written as per @inner_product.

  $
    sum_i^n bold(v)[i] bold(u)[i]
  $<inner_product>

  We can extend the meaning of inner products to functions $f "and" g$ over the interval $[a;b]$ with @inner_product_function.

  $
    integral_a^b f(x)g(x) d x
  $<inner_product_function>

  A function $f(x)$, can be written as a sum of cosine and sine terms plus a constant as per:

  $
    f(x) =A_0 / 2 + sum_(k=1)^(infinity) (A_k cos(k x) + B_k sin(k x))
  $

  Where $A_k$ and $B_k$ are the normalized inner products $angle.l f(x), cos(k x) angle.r$ and $angle.l f(x), sin(k x) angle.r$ respectively
  #footnote[Note the pointy brackets denote inner product]. These are explicitly written out in @AB_k.

  $
    A_k = 1 / pi integral_(-pi)^pi f(x) cos(k x) d k, quad
    B_k = 1 / pi integral_(-pi)^pi f(x) sin(k x) d k
  $<AB_k>

  This can be similarly extended for that grid, which is the basis for the two-dimensional FFT.

  #pagebreak()
  == Discrete Fourier transform (DFT) matrix<dft>

  #figure(
    image("figs/real_dft.svg"),
    caption: [DFT matrix],
  )

  #pagebreak()

  // = Ablating dropout and tasks



  = Dropout disabled<bad_training_acc>
  #figure(
    stack(
      image("figs/" + nodro_hash + "/acc_train_training.svg"),
      image("figs/" + nodro_hash + "/acc_valid_training.svg"),
    ),
    caption: [Disabling dropout: Without dropout, the model over-fits to all but the very simplest ($q in {2, 3}$) tasks.],
  )

  #pagebreak()

  = Sub-symbolic implementation of $f(x, y)$<subsymbolic>

  Compute $f(x)$ for ${(a,b) in NN^2 : 0 <= a,b < 113)$, by adding the two rows of $W_E_"pos"$ in @embeds to a one-hot encoded $a$ and $b$, and then multiplying by $W_E_"tok"$. Then multiply by $W_k, W_q$ and $W_v$ as per the operation described in @attn, and then add to the output of the embedding operations. Send that through the feed-forward network with the weights in @ffwd_fun. The reader is asked to confirm visually that the weight in the figures indeed computes $f(x, y) = cos (a x) + sin (b x)$ when applied in the order described above.

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
      image("figs/ffwd_w_in_prime.svg", height: 85%),
      image("figs/ffwd_w_out_prime.svg", height: 85%),
    ),
    caption: [$W_"in"$ and $W_"out"^T$],
  )<ffwd_fun>
  #figure(
    image("figs/unbeds_prime.svg"),
    caption: [$W_U$],
  )<unbeds>

  //   #pagebreak()

  //   = Training curves<ugly_training_curves>

  //   #figure(
  //     stack(
  //       dir: ttb,
  //       image("aim/metrics-19_50_32-27-Nov-24.svg"),
  //       image("aim/metrics-19_45_37-27-Nov-24.svg"),
  //       image("aim/metrics-19_51_06-27-Nov-24.svg"),
  //       image("aim/metrics-19_45_23-27-Nov-24.svg"),
  //     ),
  //     caption: [Training curves. Green and red have $lambda = 0$, and do not converge in training time, vindicating @lee2024a. This is the most obvious validation accuracy plot (bottom).],
  //   )
]
