#import "@preview/unequivocal-ams:0.1.2": ams-article, theorem, proof
#import "@preview/equate:0.2.1": equate // <- for numbering equations


#let f_hash = "7ddd799ee00349b9b94acd5d"
#let p_hash = "7ddd799ee00349b9b94acd5d"
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
    This paper investigates the emergence and mechanistic nature of algorithmic learning in transformer models through the lens of modular arithmetic and prime factorization. Building upon recent work in mechanistic interpretability, a transformer model is trained to predict remainders when dividing base-p numbers by prime factors. For numbers represented as $x_0 p^0 + x_1 p^1$ (where $x_0,x_1 < p$), the model must learn distinct strategies for each prime factor, with task difficulty scaling naturally with the size of the factor. Setting $p=113$ yields 29 parallel tasks and 12,769 samples, allowing for the study of how the model develops different computational strategies for tasks of varying complexity. Analysis of learned representations and attention patterns reveals distinct periodicities in the model's internal representations, suggesting the emergence of trigonometric basis functions similar to those found in simpler modular arithmetic tasks. This work contributes to our understanding of how neural networks discover and implement mathematical algorithms, particularly in settings with multiple related tasks of varying complexity. As a second contribution, this paper reproduces #cite(<lee2024a>, form:"prose")'s finding that amplifying slow moving gradients, can significantly speed up generalization.
    #footnote[https://github.com/syrkis/miiii].
  ],
)


#let appendix(body) = {
  set heading(numbering: "A", supplement: [Appendix])
  counter(heading).update(0)
  body
}
// #set page(margin: (x: 5em, y: 5cm))

// body ///////////////////////////////////////////////////////////////////////


= Introduction

Recent years have seen deep learning (DL) models achieve remarkable proficiency in complex computational tasks,
including protein structure prediction @jumper2021, strategic reasoning @dinan2022,
and natural language generation—areas previously thought to be the exclusive domain of human intelligence.
In contrast to traditional (symbolic) programming in which functions like $f(x, y) = cos(a dot x) + sin(b dot y)$ can be implemented with clear typographical isomorphism—meaning the code's structure directly mirrors the mathematical notation. This is evident in the case of Haskell: `f x y = cos(a * x) + sin(b * y)`. DL models, however, are inherently subsymbolic (@subsymbolic shows an equivalent DL based implementation of $f$).

Indeed, the development of DL can be understood as a transition from symbolic to subsymbolic algorithms: the gradual subsuming of computational tasks, with precursors to modern methods learning how to weigh human-designed features @shannon1950, and later works learning to create features from data to then weigh @tesauro1993, @silver2017 (in combination with search tree strategies, in the case of games). Recent DL work has even gotten rid of search trees, mapping directly from a game state to an action @ruoss2024. These methods are thus increasingly prevalent, and almost equally inscrutable, with recent works still attempting to define what interpretability even means in this context @lipton2018. Given the breadth @cybenko1989 of tasks that DL models can be trained to solve—along with their subsymbolic nature—it is, however, hardly a surprise that their interpretation remains difficult.

Mathematically, DL refers to a set of methods that combine linear maps (matrix multiplications) with non-linearities (activation functions).
Formally, all the potential numerical values of a given model's weights $W$ can be thought of as a hypothesis space $cal(H)$. Often $cal(H)$ is thus determined by human decisions (number of layers, kinds of layers, sizes of layers, etc). $cal(H)$ is then navigated using some optimization heuristic, such as gradient descent, in hope of finding a $W$ that "performs well" (i.e. successfully minimizes some loss $cal(L)$) on whatever training data we have. This vast hypothesis space, while enabling impressive performance, makes it challenging to understand how any particular solution actually works.

The ways in which a given model can minimize $cal(L)$ can be placed on a continuum: on one side we have overfitting (remembering the training data, or functioning as an archive akin to lossy and even lossless compression) and on the other we have generalizing (learning the rules that govern the relationship between input and output, or functioning as algorithm).

When describing a mechanistic explanation for a given DL model, generalization is a necessary (though insufficient) condition. Generalization ensures that there _is_ an algorithm present to be uncovered (necessary), while it is possible for that algorithm to be so obscurely implemented that reverse engineering for all intents and purposes is impossible. Various tricks, known as "regularization" exists to incentivize the emergence of the algorithmic, rather than the archiving behavior @ba2016, @krizhevsky2017, @krogh1991. As will be covered in @related_works the mechanistic interpretability (MI) literature has, despite its nascent state, already established some conventions and successes. Circuits solving basic algorithm tasks have been successfully reverse engineered, and aspects of this workflow have been automated @conmy2023. However, as of yet, no MI work has explored the effect of multitask learning.

The present paper builds on the work of #cite(<nanda2023>, form:"prose"), which trains a transformer @vaswani2017 model to perform modular addition as seen in @nanda_task.

$
  (x_0 + x_1) mod p, quad forall x_0, x_1 < p, quad p = 113
$<nanda_task>

This is referred to as $cal(T)_("nanda")$. The task of this paper, focusing on predicting remainders mod all primes less than $p$, where $x$ is interpreted as $x_0 p^0 + x_1 p^1$, formally shown in @miiii_task, is referred to as $cal(T)_("miiii")$.

$
  (
    x_0 p^0 + x_1 p^1
  ) mod f, quad forall x_0, x_1 < p, quad forall f < p, quad p = 113
$<miiii_task>

$cal(T)_("miiii")$ thus differentiates itself from $cal(T)_("nanda")$ in two significant ways: It is non-commutative, and it is multitask. These differences present unique challenges for mechanistic interpretation, as the model must learn to handle both the order-dependent nature of the inputs and develop shared representations across multiple modular arithmetic tasks.


= Related works<related_works>


Therefor the mechanistis interpretability literature tends to focus on simple algorithmic tasks, for which we ourselves can write a clear, concice algorithms, as well using the ReLU acitvation function (which for mathematical reasons favors a privlidged bases, i.e. orthogonality) @nanda2023, @conmy2023.

The loss functions perhaps most frequently used are cross-entropy and mean squared error, both of which has been shown to favor memorization rather than generalizaition @jeon2022.




Conceptually, #cite(<lee2024a>, form:"prose") argues that in the case of gradient decent, the ordred sequence of gradient updates can be viewed as consisting of two components: _1)_ a fast varying overfitting component, and _2)_ a slow varying generalizing components. The general algorithm exaplining the realtionship between input and outout is the same for all samples, whereas the weights that allow a given model to function is archive are unique for all samples. Though not proven, this intuition bears out in that generealiazation is sped up fifty fold in some cases.

Recent work shows that in practice, this continuum is gradually traversed @nanda2023.



Whereas the first machine learning methods of the 1950s can be summarized as "machines learning how to weigh human crafted features" @shannon1950, already the 1980s saw the feature crafting swallowed up by the machine learning @tesauro1993.


However, inscrutability remains a pervasive issue in DL models, often overshadowing their task proficiency. Defining what it means for a model to be interpretable, rather than inscrutable, is still an ongoing challenge @lipton2018. Interpretability indeed refers to severals distinct qualities:



Inscruitability is
Theory, however, is far behind practice when it comes to DL. Is DL best understood from and information theoretic @yu2021, a geometric @bronstein2021, or a category theretic @gavranovic2024 perspective.
The success of DL has, however, not brought much theoretical understanding.

---

In the archival mode, models exhibit overfitting by memorizing specific patterns in the training data, thereby failing to generalize to unseen data. Conversely, in the algorithmic mode, models abstract underlying principles from the training data, enabling them to generalize effectively to new, unseen data. This paper investigates the transition between these modes, demonstrating that a model can simultaneously exhibit both archival and algorithmic behaviors, particularly when trained on multiple tasks. From an information-theoretical perspective, this duality can be understood through the lens of model capacity and the trade-off between bias and variance. We build on the foundational work of Nanda @nanda2023 and Lee @lee2024a, who have shown that generalizing circuits begin to form early in the training process, suggesting that the capacity for generalization is inherent even in the initial stages of model training.




Nanda and Lee @nanda2023, @lee2024a have shown that the formation of generalizing circuits begins early in the training process. This early formation suggests that even at the initial stages of training, DLMs start developing the capacity to generalize, which later evolves as training progresses. Understanding this transition and the coexistence of both modes is crucial for advancing our theoretical understanding of DLMs and improving their practical applications.


Multi-task learning extends the capabilities of DLMs by training them on multiple OFTEN related tasks simultaneously. This approach not only improves generalization across tasks but also helps in discovering shared representations and biases that are beneficial for all tasks in the environment. Baxter @baxter2011 highlights the importance of finding a suitable bias that can generalize well across multiple tasks, thereby enhancing the overall learning process.


Understanding the internal mechanisms of DLMs remains a significant challenge. Traditional loss functions like cross-entropy and mean squared error often fail to generalize well to out-of-distribution data @yu2021. To address this, modern architectures incorporate various regularization techniques, such as layer normalization @ba2016, dropout, weight decay, and residual connections @vaswani2017. Despite these advancements, the theoretical understanding of how DLMs transition from archives to algorithms is still limited.

Nanda's task provides a valuable framework for probing the internal workings of DLMs. By reverse-engineering a simple transformer model trained to solve modular arithmetic tasks, Nanda's work sheds light on how these models implement specific algorithms. Our study builds on this by introducing two key differences: our setup is non-commutative and involves multiple tasks, providing a richer environment for understanding the generalization capabilities of DLMs.

To formalize our investigation, we consider a deep learning model \( \mathcal{M} \) consisting of a set of model weights \( \mathcal{W} \) and a procedure for applying these weights to a given input \( \mathcal{X} \). The set of potential values of \( \mathcal{W} \) constitutes a hypothesis space \( \mathcal{H} \), which defines the mapping between \( \mathcal{X} \) and \( \mathcal{Y} \) with respect to a loss function \( \mathcal{L} \). Optimization algorithms like gradient descent are used to find optimal values of \( \mathcal{W} \) within \( \mathcal{H} \), but the hypothesis space itself remains unchanged.

---


Recent years have seen deep learning models (DLMs) demonstrate remarkable proficiency in solving complex computational tasks, from language generation to protein structure prediction. From an information theoretical perspective, DLMs can perform both lossless and lossy compression @yu2021, enabling them to distill essential patterns from noisy data. Traditional compressors, like `gzip`, have even shown to outperform DLMs in classification tasks under certain conditions @jiang2023a.

This compression capability, combined with their ability to learn generative and generalized models @kingma2022, @goodfellow2014, makes understanding their internal mechanisms particularly interesting.



Traditional loss functions like cross-entropy and mean squared error,
have been shown to not genrealize well to out of distribution data @yu2021.
Indeed, additional regularization techniques are a hallmark of many modern architectures,
the most extreme example of which is perhaps the original transformer @vaswani2017—layernorm @ba2016,
dropout, weight decay, residual connections, are all integral components of the original architecture,
though recent years have seen simplifications yielding similar performance @he2023.
Importantly, deep learning architectures can function both as archives—overfitting to training data—and as generalized algorithms @power2022.

A system capable of transitioning from archive to algorithm presents intriguing questions:
Why not skip the archiving step and directly learn algorithms? What types of algorithms does it learn, and how reliably?
Can the learning process be expedited? How does the presence of multiple tasks affect the learning process?
What specific algorithm has been learned by a given system?
How can it exist as an archive and an algorithm simultaneously?
Addressing these questions is essential for advancing the theoretical understanding of deep learning and enhancing its practical applications.

In deep learning, however, theory often lags behind practice, limiting our ability to mechanistically explain basic models that have generalized on even relatively simple, synthetically generated tasks. Exploring the mechanics of deep learning models is perhaps more akin to studying biology or botany than traditional computer science. This paper, for example, reverse-engineers a simple transformer model trained to solve modular arithmetic tasks. The simplicity of this training can be likened to discovering an intriguing plant in a botanical garden (easy), while understanding its mechanics is akin to dissecting the plant to uncover the principles governing its growth and function (hard).

Prime numbers, in particular, are an interesting domain for deep learning. A frequent feature of number theoretical problems is the ease with which they can be stated. This is true for trivial problems (such as proving there are infinitely many primes) and deceptive problems (such as "all even numbers can be expressed as the sum of two primes"). The latter, known as Goldbach's conjecture, remains unsolved. There are about $n/ln(n)$ primes less than $n$. To test if a given number $n$ is prime, it is sufficient to test if it is divisible by any prime less than $sqrt(n)$ (Sieve of Eratosthenes), of which there are about $sqrt(n)/ln(sqrt(n))$.

However, how exactly a given model implements an algorithm is a non-trivial question—as we shell see, even modular addition is implemented in an obscure way @nanda2023.
This investigation probes the fundamental algorithmic structures internalized by a transformer model trained on a set of basic prime number-related modular arithmetic tasks, with slight variations in complexity. This approach provides insights into how and why specific algorithmic patterns emerge from seemingly straightforward learning processes.

My setup thus differentiates itself from Nanda's in two crucial ways:

1. Mine is non-commutative.
2. It is multi-task.

A model deep learning model, $cal(M)$, consits of a set of model weights $cal(W)$ and a procedure on how to apply these to a given input $cal(X)$. Viewed in the context of the procedure, the set of potential valuesues of $cal(W)$ can be thought of as a hypothesis space $cal(H)$ on the mapping between $cal(X)$ and $cal(Y)$, with respect to a loss function $cal(L)$. Algorithms like gradient decent, are heiristics for finsing optimal / optimised values of $cal(W)$ within $cal(H)$. $H$ itself is not modified by optimization algorithms of this level (i.e. $a x+b$ yield optimal $a "and" b$ values, but we might need a $x^2$ term to describe the given phenomen.

#cite(<baxter2011>, form:"prose") further extends the notion of generaliation and training to a multi-task paradigm.

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
  L_("ce") = sum_(t in T)1 / ln(t) 1 / p^2 sum_(i=0)^(N) sum_(j=0)^(t)ln(p_t)
$


where $alpha_i = 1/ln(f_i)$ accounts for the varying difficulty across tasks with different prime factors $f_i$.

To accelerate generalization, gradient filtering as per #cite(<lee2024a>, form: "prose") is replicated:

$
  g_t = nabla_theta L + lambda(alpha e_(t-1) + (1-alpha)g_(t-1))
$<grad>

where $e_t$ is the exponential moving average of gradients with decay rate $alpha=0.98$, and $lambda=2$ controls the influence of the slow-varying component.

Training uses full batch gradient descent with the entire dataset of $p^2$ samples. The model is evaluated on a held-out validation set after each epoch, tracking per-task accuracy and loss.


== Visualization

Much of the data worked with here is inherently high dimensional. For training, for example, we have $n$ steps, two splits (train/valid) about $p/ln(p)$ tasks, and two metrics (accuracy, and loss). This, along with the inherent opaqueness of deep learning models, motivated the developed custom visualization library, `esch`#footnote[https://github.com/syrkis/esch] to visualize attention weights, intermediate representations, training metrics, and more.

== Mechanistic Interpretability

A combination of linear products is itself a linear product. As a mechanistic interpretability rule of thumb, one should look at the outputs of the non-linear transformations. In our case that will be the attention weights, and the intermediate representations with each transformer block's MLP (which follows a ReLU activation).
Additionally, the embeddings layers will be inspected. blah blah.

Our interpretability approach combines visualization techniques with frequency analysis to understand the learned algorithmic patterns. Following @nanda2023, we analyze both the attention patterns and the learned representations through several lenses:

*Attention Visualization*
Using `esch`, the custom visualization library, to visualize attention weights and intermediate representations. The library allows for the visualization of attention patterns across different layers, as well as the visualization of intermediate representations at each layer. These visualizations provide insights into the learned patterns and help identify potential areas of improvement.

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
  caption: [Attention from $\_$ to $x_0$ in the first attention head for all $(x_0, x_1)$-pairs.],
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
    image("figs/fourier_f_m.svg"),
    image("figs/fourier_f_f.svg"),
  ),
  caption: [$W_(E_(cal(T)_b))$ in Fourier space (norm below)],
)<f_f>

As is apparent in @f_f and @p_f a lot more frequencies are in play when training for $cal(T)_b$ than $cal(T)_a$. This is to be expected if the network too implements the cosine-sine look table @nanda2023, as each task is prime related, and thus there is no common steps hit when rotating around the unit circle in the complex plane. Comparing @f_f and @r_f we see that the frequencies, though cluttered, are far from random.

#figure(
  stack(
    dir: ttb,
    image("figs/fourier_r_m.svg"),
    image("figs/fourier_r_f.svg"),
  ),
  caption: [Untrained token embeddings @he2015 in Fourier space (norm below)],
)<r_f>


#figure(
  stack(
    dir: ttb,
    image("figs/fourier_p_m.svg"),
    image("figs/fourier_p_f.svg"),
  ),
  caption: [Token embeddings ($W_E_(cal(T)_a)$) in Fourier space (norm below)],
)<p_f>


The fact of periodicity in @f_f despite the presence of multiple tasks with unique rotational steps around the circle, the non commutative nature of the task, is further @nanda2023 indication that trigonometric tables are a reliably used representation of the architecture.

//#figure(
//stack(
//dir: ttb,
// image("figs/p_f.svg"),
//   image("figs/p_f_norm.svg"),
//),
// caption: [Frequencies of $W_(E_(cal(T)_a))$ in Fourier space],
//)<p_f>

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
//
= Further work

The mysteries of primes and deep learning are both plentiful, and there are many fundamental questions to be answered in mixing the two. How does training a model on $p$ affect it performance on a $q > p$. How does predicting divisibility directly, compare to predicting remainders (both have been explored in this setup). In this spirit, the code associated with this paper is available as a pypi package, and can be installed with `pip install miiii`.

= Conclusion

The sudden learning of 25 tasks, after having generalized independently to a joint solution to the first four, indicates that.
there is indeed an assiting effect to having multiple tasks in the development of these circuits. Masking away those four tasks delays grokking beyond the epochs feasible to train for within the experiment at hand.


#bibliography("zotero.bib")

#pagebreak()

#appendix[
  #heading(level: 1, "Appendix", numbering: none)
  = Subsymbolic implementation of $f(x, y)$<subsymbolic>

  #figure(
    image("figs/4a98603ba79c4ed2895f9670/acc_train_training.svg"),
  )


]
