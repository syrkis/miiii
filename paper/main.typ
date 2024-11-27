#import "@preview/unequivocal-ams:0.1.2": ams-article, theorem, proof
#import "@preview/equate:0.2.1": equate // <- for numbering equations
#import "@preview/unify:0.6.1": num // <- for making numbers look nice


#let f_hash = "0aacbd66fd4a49da86574cc3"
#let s_hash = "7c2a10494ff64e66a9af2731"
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
    This paper investigates how neural networks learn to solve multiple related mathematical tasks simultaneously, through the lens of mechanistic interpretability (MI). A transformer model is trained on 29 parallel tasks, each requiring the prediction of remainders when dividing two-digit base-113 numbers—as has been the domain of previous MI work @nanda2023—by all potential prime factors less than 113. This setup naturally creates a spectrum of task complexity, from binary classification (division by 2) to 109-way classification (division by 109). Analysis of the model's learned representations indicates that after independently solving the first four tasks (mod 2, 3, 5, and 7), the model develops a shared computational strategy that enables rapid generalization to the remaining 25 tasks.
    Additionally, findings @lee2024a that show amplifying slow-moving gradients significantly accelerates this generalization process, are reproduced. Our results provide insights into how neural networks discover and implement mathematical algorithms, particularly when learning multiple related tasks of varying complexity.
    Project repo is https://github.com/syrkis/miiii.
  ],
)


#let appendix(body) = {
  set heading(numbering: "A", supplement: [Appendix])
  counter(heading).update(0)
  body
}

// body ///////////////////////////////////////////////////////////////////////

= Introduction

Recent years have seen deep learning (DL) models achieve remarkable proficiency in complex computational tasks, including protein structure prediction @jumper2021, strategic reasoning @dinan2022, and natural language generation @radford2018a—areas previously thought to be the exclusive domain of human intelligence. Traditional (symbolic) programming allows functions like $f(x, y) = cos(a dot x) + sin(b dot y)$ to be implemented in code with clear typographical isomorphism—meaning the code's structure directly mirrors the mathematical notation. For example, in Haskell: `f x y = cos(a * x) + sin(b * y)`. In contrast, DL models are inherently sub-symbolic, meaning that the models' atomic constituents (often 32-bit floating-point numbers centered around 0) are meaningless when viewed directly. For reference, @subsymbolic shows a DL-based implementation of the affore mentioned function. Indeed, the increasing prevalence of DL can be understood as a transition from symbolic to sub-symbolic algorithms.

Precursors to modern DL methods learned how to weigh human-designed features @shannon1950, with later works learning to create features from data to then weigh @tesauro1993, @silver2017—in combination with tree search strategies, in the case of games @browne2012. Very recent DL work has even eliminated tree search in the case of chess, mapping directly from observation space to action space @ruoss2024. Pure DL methods are thus becoming
ubiquitous but remain largely inscrutable, with recent works still attempting to define what interpretability even means in the DL context @lipton2018. Given the breadth @cybenko1989 of tasks that DL models can be (and are) trained to solve—along with their sub-symbolic nature—it is, however, hardly a surprise that their interpretation remains difficult.



Mathematically, DL refers to a set of methods that combine linear maps (matrix multiplications) with non-linearities (activation functions). Formally, all the potential numerical values of a given model's weights $W$ can be thought of as a hypothesis space $cal(H)$. Often, $cal(H)$ is determined by human decisions (number of layers, kinds of layers, sizes of layers, etc.). $cal(H)$ is then navigated using some optimization heuristic, such as gradient descent, in hope of finding a $W$ that "performs well" (i.e., successfully minimizes some loss $cal(L)$ often computed by a differentiable function) on whatever training data is present. This vast, sub-symbolic hypothesis space, while enabling impressive performance and the solving of relatively exotic#footnote[Try manually writing a function in a language of your choice that classifies dogs and cats from images.] tasks, makes it challenging to understand how any one particular solution actually works.

// #figure(
//   stack(
//     dir: ttb,
//     image("figs/" + f_hash + "/acc_valid_training.svg"),
//     // image("figs/" + f_hash + "/acc_valid_training.svg"),
//   ),
//   caption: [Visualization of the validation accuracy for all 29 tasks through training (log) time. Notice the first (top) four tasks are generalized in succession, with the rest occurring around the same time, as if by phase-transition.],
// )<trainig_loss_tio>


The ways in which a given model can minimize $cal(L)$ can be placed on a continuum: on one side, we have overfitting, remembering the training data, (i.e. functioning as an archive akin to lossy and even lossless compression); and on the other, we have generalization, learning the rules that govern the relationship between input and output (i.e. functioning as an algorithm).

When attempting to give a mechanistic of a given DL model's behavior, it entails the exsistence of a mechanism.
MI assumed this mechanism to be general, thus mkaing generalization a necessary (though insufficient) condition. Generalization ensures that there _is_ an algorithm present to be uncovered (necesity); however, it is possible for that algorithm to be so obscurely implemented that reverse engineering, for all intents and purposes, is impossible (insufficiceny). Various forms of regularization are used to incentivize the emergence of algorithmic (generalized) rather than archiving (overfitted) behavior @ba2016, @krizhevsky2017, @krogh1991.

// As will be covered in @related_works, the mechanistic interpretability (MI) literature has, despite its nascent state, already established some conventions and successes. Circuits solving basic algorithmic tasks have been successfully reverse-engineered @nanda2023, and aspects of this workflow have been automated @conmy2023.

// kinde insecure about this
As of yet, no MI work has explored the effect of multitask learning, the focus of this paper. Multitask learning also has a regularizing effect @baxter2011. Formally, the set of hyptheses spaces for each task of a set of tasks (often called environment) is denoted by $cal(H) in HH$. When minising the losses across all tasks in parallel, generalising $W$'s are thus incentivised, as these help lower loss across tasks (in contract to memorizing $W$'s that lower loss for one task). A $W$ derived from a multi-task training process can thus be thought of as the intersection of the high-performing areas of all $cal(H) in HH$.

// This empirical approach to understanding neural networks makes MI more akin to botany than theoretical computer science: while finding an interesting specimen (training a working model on an original task) is relatively straightforward—like strolling a botanical garden, looking for an unstudied flower—carefully dissecting it to understand its internal mechanisms remains challenging and labor-intensive.


// weights $W$ in $cal(H)$ that perform well across tasks are more likely to be general. #cite(<baxter2011>, form:"prose") refers to the set of hypotheses spaces for the different tasks in a given environment of tasks as $cal(H) in HH$. A $W$ performing well across tasks can thus be thought of as the intersection of the hypotheses spaces across $HH$..

In this spirit, the present paper builds on the work of #cite(<nanda2023>, form:"prose", style:"american-anthropological-association"), which trains a transformer @vaswani2017 model to perform modular addition, as seen in @nanda_task, as task denoted as $cal(T)_"nanda"$ throughout the paper.

$
  (x_0 + x_1) mod p, quad forall x_0, x_1 < p, quad p = 113
$<nanda_task>

#figure(
  image("figs/polar.svg", width: 120%),
  caption: [
    Periodic patterns in polar coordinates $(n, n mod tau)$ for numbers less than #num(12769). Left: numbers with remainder 0 mod 17 or 23 (see the two spirals). Middle: numbers with remainder 0 mod 11. Right: prime numbers.
  ],
)<nats>


The task of this paper focuses on predicting remainders modulo all primes $q$ less than $p$, where $x$ is interpreted as $x_0 p^0 + x_1 p^1$, formally shown in @miiii_task, and is referred to as $cal(T)_("miiii")$:



// $
// mat(delim: "|", 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47; 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113)
// $

$
  (
    x_0 p^0 + x_1 p^1
  ) mod q, quad forall x_0, x_1 < p, quad forall q < p, quad p = 113
$<miiii_task>

$cal(T)_("miiii")$ differentiates itself from $cal(T)_("nanda")$ in two significant ways: _1)_ it is non-commutative, and _2)_ it is, as mentioned, multitask. These differences present unique challenges for mechanistic interpretation, as the model must learn to handle both the order-dependent nature of the inputs and develop shared representations across multiple modular arithmetic tasks. Further, as $cal(T)_("miiii")$ is harder than $cal(T)_("nanda")$ the model can be expected to generalize slower when trained on the former. Therefore, #cite(<lee2024a>, form:"prose", style:"american-anthropological-association")'s recent work on speeding up generalization, by positing the model parameters gradients through time can be viewed as the sum a slow varying generalizing component (which is boosted) and a quick varying overfitting component (which is suppressed), was (successfully) replicated to make training tractable.

More generally, modular arithmetic on primes is a particularly useful task for MI as it ensures uniformity among the output classes, allows for comparison with other MI work @nanda2023, and, from a number-theoretic point of view, primes contain mysteries ranging from the trivially solved—are there an infinite number of primes?—to the deceptively difficult—can all even numbers larger than 4 be described as the sum of two primes? The latter, known as Goldbach's Conjecture, remains unsolved after centuries. The choice of using every prime less than the square root of the largest number of the dataset, also serves the following purpose: to test if a given natural number is prime, it suffices to test that it is not a multiple of any prime less than its square root—the set of tasks trained for here, can thus be viewed in conjunction as a single prime detection task (primes are the only samples whose target vector is the zero vector).

To provide insight into the periodic structure of these remainders mod $1$ for natual numbers less than #num(12769) (and motivate thinking in rotational terms), @nats visualizes various modular patterns in polar coordinates ($n, n mod 2 pi$). One could imagine tightening and loosening the spiral by multiplying $tau$ by a constant, to align multiples of a given number in a straight line (imagining this is encouraged).

// Lastly, the reader is asked to accept the inspection of a DL model transitioning from archive to algorithm on multiple simultaneous tasks as inherently interesting, independent of the current literature's scarcity on the subject.

// Lastly, the fact that MI lags so far behind the cutting edge of DL means that the models in which interesting MI is performed are relatively simple to train. The MI workflow is thus perhaps more similar to botany than theoretical computer science. While the models (the specimens) are easy to cultivate, dissecting them to uncover the principles governing their function remains a challenging endeavor. This paper aims to contribute to this effort by exploring the mechanistic interpretability of models trained on multitask modular arithmetic.

= Background and related work<related_works>

// Mechanistic Interpretability as a field is relatively new, though the objects of its study have been seen widespread adoption in the last decade. And indeed, many reverse engineering methods from other fields, such as neuroscience or signal processing, have their uses here. The following sections outline these fields and their use for the task at hand.

== Generalization and grokking

#cite(<power2022>, form: "prose", style:"american-anthropological-association") shows generalization can happen #quote(attribution: cite(<power2022>), "[...] well past the point of overfitting"), dubbing the phenomenon "grokking". The phenomenon is now well established @nanda2023, @humayun2024, @wang2024. #cite(<nanda2023>, form: "prose", style:"american-anthropological-association") shows that a generalized circuit #quote(attribution: cite(<nanda2023>), "arises from the gradual amplification of structured mechanisms encoded in the weights,") rather than being a relatively sudden and stochastic encounter of an appropriate region of $cal(H)$. The important word of the quote is thus "gradual". Further, by regarding the series of gradients in time as a stochastic signal, #cite(<lee2024a>, form:"author", style:"american-anthropological-association") proposes decomposing the signal into two components: a fast-varying overfitting component and a slow-varying generalization component. They show that amplification of the slow-varying component accelerates grokking substantially (more than fifty-fold in some cases). This echoes the idea that generalized circuits go through gradual amplification @nanda2023. To the extent that this phenomenon is widespread, it bodes well for generalizable deep learning, in that the generalizing signal that one would want to amplify exists long before the model is fully trained, and might be boosted in a targeted way my the method described by #cite(<lee2024a>, form: "author", style:"american-anthropological-association").

Conceptually, #cite(<lee2024a>, form:"author", style:"american-anthropological-association") argues that in the case of gradient descent, the ordered sequence of gradient updates can be viewed as consisting of two components: _1)_ a fast varying overfitting component, and _2)_ a slow varying generalizing components. The general algorithm explaining the relationship between input and output is the same for all samples, whereas the weights that allow a given model to function are unique for all samples. Though not proven, this intuition bears out in that generalization is sped up fifty-fold in some cases.


== Mechanistic Interpretability (MI)

=== Foundations and definitions
MI is a relatively new field focused on reverse-engineering the internal mechanisms of neural networks. #cite(<lipton2018>, form: "prose", style:"american-anthropological-association") contrasts MI with other forms of interpretability, such as feature importance analysis. While feature importance measures correlations between inputs and outputs (e.g., red pixels correlating with "rose" classifications), MI aims to understand how the model actually processes information.

=== Current methods and tools
Methods and tools used so far in MI include: Activation visualization across large datasets; Singular value decomposition of weight matrices; Ablation studies to identify critical circuits; Circuit discovery automation #cite(<conmy2023>) #cite(<weiss2021>, form: "prose", style:"american-anthropological-association")'s RASP language demonstrates how architectural constraints can be made explicit, helping researchers "think like a transformer" by expressing computation in terms of the architecture's native operations (attention as reduction, MLP as mapping). Current research is being done (CITE GROW AI?) into how to grow and merge neural network models, indicating a potential for composition of networks, with "modules" fulfilling certain properties.

=== Case study: modular addition
#cite(<nanda2023>, form: "prose", style:"american-anthropological-association")'s analysis of a transformer trained on modular addition ($cal(T)_("nanda")$) exemplifies MI methodology. They discovered that:
- The embedding layer learns trigonometric lookup tables
- The feed-forward network combines these through multiplication
- The final layer performs the equivalent of argmax
This implementation exploits the commutative property of modular addition (@commutative):
$
  (x_0 + x_1) mod p = (x_1 + x_0) mod p
$<commutative>

=== Theoretical context
While MI provides concrete insights into specific models, broader theoretical understanding of deep learning remains elusive. Different frameworks compete to explain the successes and limitations (and underlying theory) of DL.
- Information theory @yu2021
- Geometric approaches @bronstein2021
- Category theory @gavranovic2024

This theoretical uncertainty leads MI researchers to focus on simple algorithmic tasks with known solutions and architectures using ReLU activation functions, which favor interpretable orthogonal representations @nanda2023.

=== Open questions
The observation that networks can transition from memorization to algorithmic solutions raises several questions:
1. Can we bypass the memorization phase?
2. What determines which algorithms emerge?
3. How does multi-task learning affect algorithm discovery?
4. How can memorization and algorithmic computation coexist?
5. What other tricks than slow gradient boosting @lee2024a can speed up grokking?
6. If a circuit is general, can we make proofs about the model;s function?

These questions are central to both theoretical understanding and practical applications of deep learning.


=== Circuits

The term circuit is frequently used in MI. It refers to a distinct subset of the network, performing a specific task. As of yet, the notions of circuits in neural networks are relatively informal (i.e., the algebra of circuits from electrical engineering and neuroscience has not yet been rigorously applied).

== Multi-task learning in DL

As stated, multitask learning has been shown to have a regularizing effect @baxter2011, @maurer as the hypothesis *W* that performs well across the of hypothesis spaces $HH$ is more likely to be general. Viewed information theoretically, this concept is reminiscent of #cite(<shannon2001>, form:"prose", style:"american-anthropological-association")'s asymptotic equipartition property @cover2006, or even more generally, the law of large numbers, which states that the more samples we have of a distribution, the closer our estimates of its underlying properties will align with the true underlying property.

In the DL context, multitask learning is done by having the last layer output predictions for multiple tasks independently. Thus, whereas $cal(T)_("nanda")$ outputs a single $1 times 113$ vector for each of the potential remainders, $cal(T)_("miiii")$, as we shall see, outputs one vector for each prime $f < p$ (29 when $p=113$), each of which has shape $1 times f$. The embeddings layer and the transformer block is thus shared for all tasks, meaning that representations that perform well across tasks are incentivized.

== Loss functions and training dynamics

Perhaps the most widespread loss functions used in deep learning are mean cross-entropy @mce (for classification) and mean squared error @mse (for regression).

$
  L_("MCE") &= 1 / n sum_(i=1)^n sum_(j=1)^k y_p_(i j) ln(1 / hat(y)_p_(i j))#<mce> \
  L_("MSE") &= 1 / n sum_(i=1)^n (y_i - hat(y)_i)^2 #<mse>
$

These have various computational and mathematical properties that make them convenient to use, while they, however, struggle to generalize @jeon2022. Due to its prevalence, however, MCE is chosen in this paper. However, since we have multiple tasks, the MCE is modified as shown in

== Deep number theory

Multiple papers describe the use of deep learning to detect prime numbers @egri2006, @lee2024, @wu2023a.
None are particularly promising as prime detection algorithms, as they do not provide speedups, use more memory, or are less accurate than traditional methods.
However, in exploring the foundations of deep learning, the task of prime detection is interesting, as it is a simple task that is difficult to learn, and is synthetic, meaning that the arbitrary amounts of data are generated by a simple algorithm.

Prime numbers, in particular, are an interesting domain for deep learning. A frequent feature of number theoretical problems is the ease with which they can be stated. This is true for trivial problems (such as proving there are infinitely many primes) and deceptive problems (such as "all even numbers can be expressed as the sum of two primes"). The latter, known as Goldbach's conjecture, remains unsolved. There are about $n/ln(n)$ primes less than $n$. To test if a given number $n$ is prime, it is sufficient to test if it is divisible by any prime less than $sqrt(n)$ (Sieve of Eratosthenes), of which there are about $sqrt(n)/ln(sqrt(n))$.

== Transformer architecture

Various modifications/simplifications have been made to the transformer block @he2023, @hosseini2024.
Transformers combine self-attention (a communication mechanism) with feed-forward layers (a computation mechanism).
Importantly, transformers tend to rely on residual streams (I will elaborate).
I am currently using the original transformer block, but I want to switch to @he2023's block, as it is simpler and more interpretable—but there is not much research on it yet.

Traditional loss functions like cross-entropy and mean squared error,
have been shown to not generalize well to out-of-distribution data @yu2021.
Indeed, additional regularization techniques are a hallmark of many modern architectures,
the most extreme example of which is perhaps the original transformer @vaswani2017—layer norm @ba2016,
dropout, weight decay, residual connections, are all integral components of the original architecture,
though recent years have seen simplifications yielding similar performance @he2023.
Importantly, deep learning architectures can function both as archives—overfitting to training data—and as generalized algorithms @power2022.


= Methods

However, how exactly a given model implements an algorithm is a non-trivial question—as we shall see, even modular addition is implemented in an obscure way @nanda2023.
This investigation probes the fundamental algorithmic structures internalized by a transformer model trained on a set of basic prime number-related modular arithmetic tasks, with slight variations in complexity. This approach provides insights into how and why specific algorithmic patterns emerge from seemingly straightforward learning processes.

My setup thus differentiates itself from $cal(T)_"nanda"$ in two crucial ways:

1. Mine is non-commutative.
2. It is multitask.

A model deep learning model, $cal(M)$, consists of a set of model weights $cal(W)$ and a procedure on how to apply these to a given input $cal(X)$. Viewed in the context of the procedure, the set of potential values of $cal(W)$ can be thought of as a hypothesis space $cal(H)$ on the mapping between $cal(X)$ and $cal(Y)$, regarding a loss function $cal(L)$. Algorithms, like gradient descent, are heuristics for finding optimal / optimized values of $cal(W)$ within $cal(H)$. $H$ itself is not modified by optimization algorithms of this level (i.e. $a x+b$ yields optimal $a "and" b$ values, but we might need an $x^2$ term to describe the given phenomena).

#cite(<baxter2011>, form:"prose", style:"american-anthropological-association") further extends the notion of generalization and training to a multitask paradigm.

== Tasks

Stated plainly: the task predicts the remainder when dividing a two-digit base-$p$ number by each prime factor $q$ less than $p$. The set of prime factors we construct tasks for is thus $F = {f in PP : f < p}$
For $p=113$, this yields 29 parallel tasks, one for each prime less than $p$. Each task predicts a remainder in the range $[0, f-1]$. This means smaller primes like 2 and 3 require binary and ternary classification, respectively, while the largest prime less than $p$, 109, requires predictions across 109 classes. The tasks thus naturally vary challenged: predicting $mod 2$ requires distinguishing odd from even numbers (which in binary amounts to looking at the last bit), while predicting $mod 109$ involves making a selection between many relatively similar classes. From an information-theoretical perspective, the expected cross entropy for an $n$-class problem is $ln(n)$, which has implications for the construction of the loss function, further discussed in @training.


== Data

*Input Space ($X$)*
Each input $x in X$ represents a number in base $p$ using two digits, $(x_0,x_1)$, where the represented number is $x_0 p^0 + x_1 p^1$. For example, with $p=11$, the input space consists of all pairs $(x_0,x_1)$ where $x_0,x_1 < 11$, representing numbers up to $11^2-1 = 120$. This yields a dataset of 121 samples. @miiii_x_11 visualizes this input space, with each cell representing the value $x_0 p^0 + x_1 p^1$.

#figure(
  image("figs/x_11_plot.svg", width: 110%),
  caption: [Visualization of input space $X$ for $p=11$. Each cell $(x_0,x_1)$ represents the number $x_0 p^0 + x_1 p^1$. The top left shows 0 $(0,0)$, and the bottom right shows 120 $(10,10)$—both in base-11],
)<miiii_x_11>

*Output Space ($Y$)*
For each input $x$, a vector $y in Y$ contains the remainder when dividing by each prime less than $p$. For $p=11$, this means predicting the remainder when dividing by 2, 3, 5, and 7. Each element $y_i$ ranges from $0$ to $f_i-1$ where $f_i$ is the $i$-th prime. @miiii_y_11 visualizes these remainders, with each subplot showing the remainder pattern for a specific prime divisor. For comparison, the rightmost plot shows the output space of @nanda2023's modular addition task.

#figure(
  image("figs/y_11_plot.svg", width: 120%),
  caption: [Output space $Y$ for $p=11$. The first four plots show remainders when dividing by 2, 3, 5, and 7 respectively. The rightmost plot shows the output space of the modular addition task for comparison.],
)<miiii_y_11>

== Model

// Architectural decisions are made to align with #cite(<lee2024a>, form: "prose") and #cite(<nanda2023>, form: "prose").
The model follows the original transformer architecture @vaswani2017 with several key design choices aligned with recent work on mechanistic interpretability @nanda2023, @lee2024a: biases are disabled, and layer normalization is not used. The model consists of three main components: an embedding layer, transformer blocks, and an output layer. All weights are initialized following #cite(<he2015>, form: "prose", style:"american-anthropological-association").

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
where $W_("out")$ projects to $sum_(i=1)^k f_i$ dimensions for $k$ prime factors, with $q_i$ being the $i"th"$ prime less than $p$.


$
  mat(delim:"[", quad x_0 quad x_1 quad \_ quad)
$

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

The model is trained using AdamW @loshchilov2019 with $beta_1=0.9$, $beta_2=0.98$ following @nanda2023. To handle the varying number of classes across tasks (from 2 classes for mod 2 to 109 classes for mod 109), a modified (weighted) mean cross-entropy (@mce) loss is created, correcting for the difference in expected loss within each task. Note that $EE[L_("MCE")] = ln(1/k)$, where $k$ is the number of classes within the task in question. Correcting for this, the loss function becomes as shown in @mmce.


$
  L_(cal(T)_"miiii") &= - &&sum_(q in Q) L_"MCE"_f / (-ln(f)) \
  &= - &&sum_(q in Q) (sum_(i=1)^n sum_(j=1)^(f) y_(k_f i j) ln(hat(y)_(k_f i j)) ) / (- n ln(f)) \
  &= &&sum_(q in Q)sum_(i=1)^n sum_(j=1)^(f) (y_(k_f i j)ln(hat(y)_(k_f i j)) ) / (n ln(f)) #<mmce>
$

To accelerate generalization, gradient filtering as per #cite(<lee2024a>, form: "prose") is implemented and replicated.

$
  g_t = nabla_theta L + lambda(alpha e_(t-1) + (1-alpha)g_(t-1))
$<grad>

where $e_t$ is the exponential moving average of gradients with decay rate $alpha=0.98$, and $lambda=2$ controls the influence of the slow-varying component.

Training uses full batch gradient descent with the entire dataset of $p^2$ samples (#num(12769) when $p=113$). The model is evaluated on a held-out validation set after each epoch, tracking per-task accuracy and loss. As the setup used in $cal(T)_"nanda"$, training was done on thirty percent of the total dataset, with the remaining used for validation (1000 samples) and testing (remaining). Further as $cal(T)_"miiii"$ involves the learning of 29 (when $p=113$) tasks rather then 1, and due to each task's non-commutativity, a larger hidden dimension of 256 was added to the hyper parameter search space, as well as the potential for 8 heads ($cal(T)_"nanda"$ was solved with a hidden dimension of 128, and 4 heads). The number of transformer blocks was kept at 1, as this ensures consistency with $cal(T)_"nanda"$ (and as full generalization was possible, as we shall see in the results).


== Visualization

Much of the data worked with here is inherently high dimensional. For training, for example, we have $n$ steps, two splits (train/valid) about $p/ln(p)$ tasks, and two metrics (accuracy, and loss). This, along with the inherent opaqueness of deep learning models, motivated the developed custom visualization library, `esch`#footnote[https://github.com/syrkis/esch] to visualize attention weights, intermediate representations, training metrics, and more. The most important plot type for the reader to keep in mind is seen in @plot_type. As there are only #num(12769) samples when $p=113$, all samples can be fed at once to the model. Inspecting a specific activation thus yields a $1 times$ #num(12796) vector $v$, which can be reshaped at a $113 times 113$ matrix, with the two axes varying $x_0$ and $x_1$ from 0 to 112, respectively. The top-left corner then shows the given value for the sample $(0 dot p^0 + 0 dot p^1)$, and so on.

#figure(
  stack(
    dir: ltr,
    image("tmp.svg", width: 100%),
  ),
  caption: [Top left $37 times 37$ slice of the attention pattern from $hat(y)$ to $x_0$ in the first attention head of all $(x_0, x_1)$ pairs, for a model trained on $cal(T)_"nanda"$. Note that each square of the plot represents a unique sample and is thus entirely independent of one another. The periodicity is thus a function of the model learning an order of the natural numbers in question.],
)<plot_type>

// TODO: ADD FOURIER PLOTS

Note that that in `esch` plots, when appropriate, only the top leftmost $37 times 37$ slice is shown, to not overwhelm the reader. Visualizations are available in the Appendix.

== Mechanistic interpretability

A combination of linear products is itself a linear product. As a mechanistic interpretability rule of thumb, one should look at the outputs of the non-linear transformations. In our case, that will be the attention weights and the intermediate representations with each transformer block's MLP (which follows a ReLU activation).
Additionally, the embedding layers will be inspected. blah blah.

Our interpretability approach combines visualization techniques with frequency analysis to understand the learned algorithmic patterns. Following @nanda2023, we analyze both the attention patterns and the learned representations through several lenses:

=== Attention visualization

Using `esch`, the custom visualization library, to visualize attention weights and intermediate representations. The library allows for the visualization of attention patterns across different layers, as well as the visualization of intermediate representations at each layer. These visualizations provide insights into the learned patterns and help identify potential areas of improvement.

=== The fast Fourier transform

As periodicity is established by #cite(<nanda2023>, form: "prose", style:"american-anthropological-association") as a fundamental feature of the model trained on $cal(T)_"nanda"$, the fast Fourier transform (FFT) algorithm is used to detect which frequencies are in play.
Note that any square image, can be described as a sum of 2d sine and cosine waves varying in frequency from 1 to the size of the image divided by 2 (plus a constant).
This is a fundamental tool used in signal processing. The theory is briefly outlined in @fft for reference.
This analysis helps identify the dominant frequencies in the model's computational patterns.

The default basis of the one-hot encoded representation of the input is thus the identity matrix. This can be projected into a Fourier basis by multiplying with the discrete Fourier transform (DFT) matrix visualized in @dft.


// We track how representations evolve through the network by:
// - Visualizing activation matrices at each layer
// - Computing correlation matrices between different positions and features
// - Analyzing the residual stream contributions

// These analyses are performed across different input patterns to understand how the model distinguishes between prime and composite numbers.

// == Evaluation

// The model is trained using the AdamW optimizer with a learning rate of $10^(-3)$,
// and weight decay of $1.0$. Dropout is used with a rate of $0.5$.
// A hidden size of 128, a batch size of $|cal(D)|$, and a maximum of 6 000 epochs.
// The MLP layers map the input from 128 to 512 to 128.
// Layer normalization is also used.
// The gradients were modified in accordance with the method described by #cite(<lee2024a>, form: "prose"),
// to speed up generalization.
// 3 transformer layers are used. The model is trained on a single Apple M3 Metal GPU with JAX @jax2018github.
// Optax @deepmind2020jax was used for the optimizer.

// $cal(T)$ is harder to solve than the modular addition task by #cite(<nanda2023>, form: "prose"),
// as it consists of multiple modular multiplication sub-tasks shown in @task_1.

// $
// forall p < sqrt((|X|)/ln(|X|)) and n!=p [n mod p != 0]
// $ <task_1>

// $T_1$ is being strictly harder than $T_2$,
// might merit and increase in the number of layers, heads, and hidden size,
// which I am currently investigating (update for Anders).



// #figure(
//   image("figs/attention_layer_0.svg"),
//   caption: [Example of attention from $hat(y)$ to $x_0$],
// )

// #figure(
// image("figs/attention_layer_1.svg"),
// caption: [Example of attention from $hat(y)$ to $x_0$],
// )

// The frequency of a positive sample in task $i$ is used as the weight for the focal loss during training.
// Furthermore, a one-hot vector is used to mask tasks to shield the model from a particular signal during training.

= Results

#figure(
  stack(
    dir: ttb,
    image("figs/" + f_hash + "/acc_train_training.svg"),
    image("figs/" + f_hash + "/acc_valid_training.svg"),
  ),
  caption: [Representation of training and validation acc ($x$-axis is in log scale).],
)<trainig_acc>

The hyperparameters performing best on $cal(T)_"miiii"$ can be seen in @hyper_param_search_result.
Notably, the model never converges when $lambda = 0$, confirming the utility of #cite(<lee2024a>, form:"prose", style:"american-anthropological-association")'s amplification of slow varying gradients, in the context of $cal(T)"miiii"$.
Setting dropout to 0 results in performance on the validation set to diverge (overfitting) in spite of the heavy regularization used.

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

    "0.1", $1 / 2$, $1 / 3$, "256", $3 times 10^(-4)$, "4",
  ),
  caption: [Reslt of hyperparameter search over $cal(T)_"miiii"$.],
)<hyper_param_search_result>


As seen in figures @trainig_acc and @training_loss, the model grokked on all 29 tasks, achieving perfect accuracy on the validation and test sets. Note that tasks 2, 3, 5 and 7 generalize in succession of one another, while the remaining 25 tasks generalize around epoch #num(40000) in no particular order. This could indicate that a more general solution has been found, allowing for a sort of phase transition for the remaining tasks by reusing circuitry developed through the first four.

#figure( // should we have plot of lambda = 0 instead?
  stack(
    image("figs/41e20b3a4790402f8a5be458/acc_train_training.svg"),
    image("figs/41e20b3a4790402f8a5be458/acc_valid_training.svg"),
  ),
  caption: [Representation of training and validation acc with dropout disabled],
)<bad_training_acc>

#figure(
  stack(
    dir: ttb,
    image("omega-series-1.svg", width: 110%),
    image("figs/omega.svg", width: 110%),
    // image("omega-series-2.svg", width: 110%),
  ),
  caption: [Representation of active frequencies (as per the FFT) of the transformer block neurons throught training (top). Variance of frequency activations, and number of frequencies above a threshold of $omega > mu + 2 sigma$ (bottom)],
)<finding>

#figure(
  stack(
    dir: ttb,
    image(f_hash + "_tmp_1.svg", width: 110%),
    image(f_hash + "_astrid.svg", width: 110%),
  ),
  caption: [Representation of active frequencies (as per the FFT) of the transformer block neurons throught training (top). Variance of frequency activations, and number of frequencies above a threshold of $omega > mu + 2 sigma$ (bottom)],
),

#figure(
  stack(
    dir: ttb,
    image(s_hash + "_tmp_1.svg", width: 110%),
    image(s_hash + "_astrid.svg", width: 110%),
  ),
  caption: [Representation of active frequencies (as per the FFT) of the transformer block neurons throught training (top). Variance of frequency activations, and number of frequencies above a threshold of $omega > mu + 2 sigma$ (bottom)],
)


#figure(
  image("figs/grads_norms.svg"),
  caption: [L2 norm of graidents through time for the different weight matricies of a model trained on $cal(T)_"miiii"$],
)<l2_norms>


== Positional embeddings


@pos_emb shows the positional embeddings of the $cal(T)_"nanda"$ to be virtually identical (Person correlation of 0.95), which is to be expected due to the tasks' commutativity (a given value at $x_0$ or $x_1$ contributes the same to the task). The same measure for a model trained on $cal(T)_"miiii"$ is -0.64, translating the embeddings differently for the two positions. This is to be expected, as by the task's non-commutativity, $x_0 dot p ^ 0 != x_0 dot p^1$. Inspecting the positional embeddings confirms the obvious: position matters.

#figure(
  image("figs/pos_emb.svg"),
  caption: [Positional embeddings for the network trained on @nanda_task (top) and @miiii_task (bottom)],
)<pos_emb>


== Token embeddings


Recall that a matrix $upright(bold(M))$ of size $m times n$ can be decomposed to its singular values $upright(bold(M)) = upright(bold(U))upright(bold(Sigma))upright(bold(V^T))$ (with the transpose being the complex conjugate when $upright(bold(M))$ is complex), where $upright(bold(U))$ is $m times m$, $upright(bold(Sigma))$ an $m times n$ rectangular diagonal matrix (whose diagonal is represented as a flat vector throughout this paper), and $upright(bold(V^T))$ a $n times n$ matrix. Intuitively, this can be thought of rotating in the input space, then scaling, and then rotating in the output space.
@s shows us that the singular values of the the token embeddings learned for $cal(T)_"miiii"$ to be much more diffuse than those for $cal(T)_"nanda"$ (with the ticks indicating at what point 50 % and 90 % of the variance is accounted for). As stated, the embedding layer of the $cal(T)_"nanda"$ trained models represents a look table for the sine and cosine values of the tokens—hance the periodicity of the most significant singular vectors @p_U. Visual inspection of the top most vectors of @f_U indeed shows periodicity, but a much larger fraction of the vectors is required to capture the same amount of variance @s.

#figure(
  image("figs/S.svg"),
  caption: [First 83 of 113 (truncated for clarity) singular values of $upright(bold(U))$ for $cal(T)_("nanda")$ (top) and $cal(T)_("miiii")$ (bottom)],
)<s>

#figure(
  image("figs/p_U.svg"),
  caption: [Most significant singular vectors of $upright(bold(U))$ for $cal(T)_("nanda")$],
)<p_U>

#figure(
  image("figs/f_U.svg"),
  caption: [Most significant singular vectors of $upright(bold(U))$ for $cal(T)_("miiii")$],
)<f_U>



To further understand the underlying structure of the token embeddings, the fast Fourier transform (FFT) algorithm is used. @p_f shows the five particularly active frequencies for the $cal(T)_"nanda"$-model. For the $cal(T)_"miiii"$-model we see a much broader spectrum of frequencies is active, though comparing to a randomly initialized baseline, the periodicity remains apparent. This is to be expected if the network too implements the cosine-sine look table @nanda2023, as each task relates to a particular prime $q$—no point is hit twice when rotating through $CC$ with $q$ steps for very $q in Q$.


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




// The fact of periodicity in @f_f despite the presence of multiple tasks with unique rotational steps around the circle, the non-commutative nature of the task, is further @nanda2023 indication that trigonometric tables are a reliably used representation of the architecture.

//#figure(
//stack(
//dir: ttb,
// image("figs/p_f.svg"),
//   image("figs/p_f_norm.svg"),
//),
// caption: [Frequencies of $W_(E_(cal(T)_("nanda")))$ in Fourier space],
//)<p_f>

== Attention patterns

Unlike @nanda_task, our attention heads focus on one digit or the other. This could be due to the non-commutative nature of the task.


== Feed-forward

// #figure(
//   stack(
//     dir: ttb,
//     image("figs/train_acc.svg", width: 100%),
//     image("figs/valid_acc.svg", width: 100%),
//   ),
//   caption: [Training and validation accuracy],
// )<generalization_levels_>


// We see that the positional embeddings are orthogonal. The token embeddings of $x_0$ and $x_1$ are offset, allowing for the cosine and sine table to be learned for both.
// NOTE: They might not be orthogonal, but rather pointing in opposite directions (we only have two vectors, so orthogonality is not needed).

// #figure(
//   image("figs/pos_emb.svg", width: 100%),
//   caption: [Positional embeddings for the first $37$ for a model trained on @nanda_task (top) and @miiii_task (bottom). The low information contained in the positional encoding of @nanda_task is to be expected as the task is commutative, while @miiii_task is not—$(x_0 + x_1) mod p = (x_1 + x_0) mod p$ but $((x_0 p^0 + x_1 p^1) mod p) != ((x_1 p^1 + x_0 p^0) mod p)$..],
// )<generalization_levels_>

// - sin/cos lookup tables in embedding layer.
// - does pos not matter for this task? No, cos it is not commutative. (a + b) mod p = (b + a) mod p -> Nanda. But (a p^1 + b p^0) mod p != (b p^1 + a p^0) mod p.

#figure(
  image("figs/weis_miiii_113_slice_23.svg", width: 110%),
  caption: [Attention from $hat(y)$ to $x_0$ for the four attention heads.],
)


#figure(
  image("figs/ffwd_miiii_113_4_neurons_slice_23.svg", width: 110%),
  caption: [Slice of first four neurons in the feed forward network for $p=113$ trained for $cal(T)_"miiii"$. Notice the periodicity.],
)

#figure(
  image("figs/miiii_113_F_neuron_0.svg"),
  caption: [Neuron 0 for sample (0, 0) in Fourier space. We see that frequencies responding to the given sample are extremely sparse.],
)

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

= Discussion

The evolution of the frequencies $omega$ in the neuron space shown in @finding echoes the generalization phases shown in @trainig_acc. As the model groks on the primes 2, 3, 5, and 7 in the first fifth of the training run a handful of frequencies become dominant, #cite(<nanda2023>, form:"prose")'s findings. As the model improves training loss with respect to the rest of the tasks, we see a more uniform distribution of active frequencies, with no single frequenci cross the frequently used signal processing threshold of $mu + 2 omega$.

As the reamining 25 tasks are solved, we do see a significant number of frequencies cross the significance threshold. The gully generalized sollution to all 29 tasks has 9 frequencies above the threshold, while the soultuion to the first four tasks has 4. This is indication of circuit reuse and generality, though more work needs to be done to fully confirm this.

One possible alternative explenation is that the standard $mu + 2 sigma$ threshold is too high, and that multiple frequencies are present and doing useful work, but there remains too much noise.

Interstingly, the number of active frequencies for the first four tasks is similar to that of a model trained on $cal(T)_"nanda"$.

However, inspecting when generalization happens, and when significant frequencies appear in the neuron space, we see that frequencies appear AFTER generalization. This could be an indication that there exists another phase after grokking in the case of multi task learning, i.e. task circuit merging. A sort of syzygy of circuits.

= Further work

Several promising directions emerge from this work. First, our observation that circuits developed for initial tasks are later reused suggests a novel approach to accelerating learning: by identifying and amplifying these emergent circuits early in training, we might extend #cite(<lee2024a>, form:"prose", style:"american-anthropological-association")'s gradient-based acceleration techniques. While their method amplifies slow-moving components of the gradient signal uniformly, targeted amplification of specific emerging circuits could provide even greater speedups.

// Second, the relationship between model capacity and prime size remains unexplored. How does a model trained on remainders modulo primes less than $p$ perform when tested on primes less than some larger $q$? This question connects to fundamental issues in neural network generalization and could provide insights into how these models encode mathematical concepts.

Second, the choice between predicting remainders versus binary divisibility presents an interesting trade-off. While remainder prediction requires more output neurons, it might provide richer gradient signals during training. A systematic comparison of these approaches could yield practical insights for training mathematical neural networks.

Lastly, this work raises questions about the relationship between circuit reuse and task difficulty. Are simpler tasks always learned first, and do their circuits necessarily form building blocks for more complex tasks? More specifically: how related to a simpler task $a$ does a more complex task $b$ for circuits to reusable between the two. Understanding these relationships could inform better training strategies for multi-task learning more broadly.

The code associated with this paper is available as a PyPI package (`pip install miiii`) to facilitate exploration of these questions (as well as replication of the findings at hand).

= Conclusion

This paper presents several key findings about how neural networks learn multiple related mathematical tasks. First, we observed a striking pattern in the model's learning trajectory: after independently solving four fundamental tasks (remainders modulo 2, 3, 5, and 7), the model rapidly generalized to 25 additional tasks. This suggests the emergence of a general computational strategy, rather than task-specific solutions. Our ablation studies, where masking the initial four tasks prevented generalization within feasible training time, further support this hierarchical learning pattern.

Second, our analysis of the model's internal representations revealed that multi-task learning promotes the development of more robust and general algorithmic solutions. The periodic patterns observed in the model's embeddings and attention mechanisms, while more complex than those found in single-task modular arithmetic, demonstrate how the model learns to handle multiple periodic relationships simultaneously.

These findings contribute to our understanding of mechanistic interpretability in two ways: they demonstrate how multi-task learning can guide networks toward more general solutions, and they show how the complexity of these solutions can be systematically analyzed even as the number of tasks increases. Future work might explore how these insights could be applied to accelerate learning in other domains where multiple related tasks must be solved simultaneously.

#bibliography("zotero.bib")

#pagebreak()

#appendix[
  #heading(level: 1, "Appendix", numbering: none)

  = Training plots<training_plots>

  // Training plot of model trained on $cal(T)_"miiii"$ without dropout.

  == Training loss <training_loss>
  #figure(
    stack(
      dir: ttb,
      image("figs/" + f_hash + "/loss_train_training.svg"),
      image("figs/" + f_hash + "/loss_valid_training.svg"),
    ),
    caption: [Representation of training and validation loss ($x$-axis is in log scale).],
  )



  #figure(
    stack(
      image("figs/41e20b3a4790402f8a5be458/loss_train_training.svg"),
      image("figs/41e20b3a4790402f8a5be458/loss_valid_training.svg"),
    ),
    caption: [Loss for run with dropout disabled],
  )


  #pagebreak()


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

  Where $A_k$ and $B_k$ are the normalized inner products $angle.l f(x), cos(k x) angle.r$ and $angle.l f(x), sin(k x) angle.r$ respectively
  #footnote[Note the pointy brackets denote inner product]. These are explicitly written out in @AB_k.

  $
    A_k = 1 / pi integral_(-pi)^pi f(x) cos(k x) d k, quad
    B_k = 1 / pi integral_(-pi)^pi f(x) sin(k x) d k
  $<AB_k>

  This can be similarly extended for that grid, which is the basis for the two-dimensional FFT.

  #pagebreak()

  = Discrete Fourier transform (DFT) matrix<dft>

  #figure(
    image("figs/real_dft.svg"),
    caption: [DFT matrix],
  )


  #pagebreak()

  = Sub-symbolic implementation of $f(x, y)$<subsymbolic>

  Compute $f(x)$ for ${(a,b) in NN^2 : 0 <= a,b < 113}$, by adding the two rows of $W_E_"pos"$ in @embeds to a one-hot encoded $a$ and $b$, and then multiplying by $W_E_"tok"$. Then multiply by $W_k, W_q$ and $W_v$ indecently in performing the operation described in @attn, and then add to the output of the embedding operations. Send that through the feed-forward network with the weights in @ffwd_fun. The reader is asked to confirm visually that the weight in the figures indeed computes $f(x, y) = cos (a x) + sin (b x)$ when applied in the order described above.

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

  #pagebreak()

  = Training curves

  #figure(
    stack(
      dir: ttb,
      image("aim/metrics-19_50_32-27-Nov-24.svg"),
      image("aim/metrics-19_45_37-27-Nov-24.svg"),
      image("aim/metrics-19_51_06-27-Nov-24.svg"),
      image("aim/metrics-19_45_23-27-Nov-24.svg"),
    ),
    caption: [Training curves. Green and red have $lambda = 0$, and do not converge in training time, vindicating @lee2024a. This is most obivous validation accuracy plot (bottom).],
  )
]
