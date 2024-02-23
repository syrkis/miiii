---
title: Mechanistic Interpretability on Irreducible Unsigned-Integer Identifiers
author: Noah Syrkis
type: paper
---

\begin{abstract}
We apply the mechanistic interpretability framework to a transformer model trained on a dataset of irreducible integers. We show that the model has learned to perform modular addition, and we reverse-engineer the model to understand how it does so.
\end{abstract}

# Introduction

<!-- why is deep learning theoreitically hard -->

Deep learning is the modelling of high dimensional probability density functions by fitting piece-wise linear functions [@prince2023].
Examples of linear functions are $f(X)=XW+b$ or $f(X)=X\star W + b$
(the later of which represents convolution, or multiplication in the frequency domain).
The output of linear function is fed as input to another via non-linear activation functions like [@eq:sigmoid] or [@eq:relu].
The deep learning process is thus, at core not understood.

<!-- why is deep learning practically hard -->

Similar to its process, the deep learning product, the models, are almost always as mysterious and inscruitable as the process that birthed it. The inscruitability is further complicated by the us not 

<!--- -->

$$\sigma(x)=\frac{1}{1-e^{-x}}$${#eq:sigmoid}

$$\text{ReLU}(x)=\max(0, x)$${#eq:relu}

For some reason (and this as deep an answer as science currently affords) this structure has the ability to generelise.
As @mitchell1997 puts it "A computer program is said to learn from experience E with respect to some class of tasks T, and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."
Why does it learn? What does it learn?



Often what is reported about a given model is a like accuracy or mean-squared-error (MSE).

Both are single members of the set $\mathbb{R}$,

A symptom of this is that very fundamental things willb esuprisgin, like the fact of @sohl-dickstein2024's discovery that the landscape of trainable hyper paramters is fractal.


Reverse-engineerings deep neural networks (DNN) is a relatively new field, but has already shown success. For example, reverse engineers a transformer model to understand how it performs modular addition. attempts to automate the reverse-engineering process, and is somewhat successful.

Mechanistic interpretability (MI) posits that deep neural networks (DNN) are circuits that can be reverse-engineered to understand their inner workings. MI is a relatively new field, but has already shown success. For example, reverse engineers a transformer model to understand how it performs modular addition. attempts to automate the reverse-engineering process, and is somewhat successful.

Mechanistic interpretability (MI) posits that deep neural networks (DNN) are circuits that can be reverse-engineered to understand their inner workings. MI is a relatively new field, but has already shown success. For example, reverse engineers a transformer model to understand how it performs modular addition. attempts to automate the reverse-engineering process, and is somewhat successful.

# Background

Mechanistic interpretability is said to be a young field because the object of it study is new. The idea of reverse engineering circuits is however not new, and many methods from neuroscience and electrical engineering can be applied here without much modification.

Symbolic and sybsymbolic models. There are models whose building blocks exist on the same level of abstraction as the model as a whole does.

```
function xor(a, b, c)
    not (a and b and c) and
    not (not a and not b and not c)
end
```

does the same as

$$
f(x) = \sigma\left(\begin{bmatrix}0 & -1 \\ 1 & 1\end{bmatrix}\begin{bmatrix}0 & -1 \\ 1 & 1\end{bmatrix}x\right)> 0
$$


Neural networks are universal function aproximators, operating sub-symbolically,
(presumably) learning intersting and sometimes even conseice algorithms,
while representing these in a largely inscruitable way.

@lee2024 shows transformer models applied to prime number analysis.


## Transformer models

The transformer model introdcued by @vaswani2017 is behind much of the recent success in natural langauge processing (NLP). This fact makes its interpretability important.
From a geometric deep learning point of view [@bronstein2021], attention—which is the value proposition of the transformer architecture—is, as [@eq:attn] shows, relatively simple, on par with convulutions ([@eq:conv]) and recurrence ([@eq:rec])

$$
A(X) = KQ^t/V
$${#eq:attn}

$$
C(X) = X \star K
$${#eq:conv}

$$
R(X_i, (R(X_{i-1}))) = f(X, X)
$${#eq:rec}

And yet, @vaswani2017's now famous transformer block diagram is nutoriously cluttered, containing risidual streams, normalization layers, projections, concatenations, and more, in addition to the attention mechanism itself.
Each of these are impressive addition that boost performance, for reasons that are not deeply understood.
@he2023a does away with *both* the residual stream and the normalization layers, without much degredation in performance, and @he2023, furhter publishes the simplified transformer block.


As artificial intelligence systems 
Mechanistic interpretability (MI) posits that deep neural networks (DNN) are circuits that can be reverse-engineered to understand their inner workings. MI is a relatively new field, but has already shown success. For example, @nanda2023 reverse engineers a transformer model [@vaswani2017] to understand how it performs modular addition.
@cover2006 attempts to automate the reverse-engineering process, and is somewhat successful.
However, the process is still largely manual and requires a deep understanding of the model's architecture and training process. Leveraging the transformer model's attention mechanism, @conmy2023 attempts to automate the reverse-engineering process, and is somewhat successful. 
@conmy2023 attempts to automate the reverse-engineering process, and is somewhat successful, while @belcak2022 diconfirms this.

In this paper, we apply the MI framework to a transformer model trained on a dataset of irreducible integers. We show that the model has learned to perform modular addition, and we reverse-engineer the model to understand how it does so.

## Mechanistic interpretability

Mechanistic interpretability (MI) posits that deep neural networks (DNN) are circuits that can be reverse-engineered to understand their inner workings. MI is a relatively new field, but has already shown success. For example, @nanda2023 reverse engineers a transformer model [@vaswani2017] to understand how it performs modular addition.

## Valid sequences

As the tokenization is done on the digit level the model is efectively asked "is the sequence 1,0,1 (101) valid prime pattern".
It is a simple yes or no question, but it's answer depends on the 99 preceding natural numbers.
Similar to @hofstadter2000's MIII^[Primes are refered to as "irriducible intergers" to have the title have the MIII acronym.] puzzle, something something blah blah bullshit.

# Methodology

Our methodology consists of the following steps:

## Data

Here's a table:

\atable{data.csv}{Dataset}{data}

The dataset consists of four-digit integers and their labels. The labels are 1 if the integer is irreducible, and 0 otherwise. The dataset is generated by taking all four-digit integers and checking if they are irreducible. The dataset is then split into a training set and a test set.

## Model

The model is a transformer model in the style of @hu2023. It is trained on the dataset above.

## Reverse-engineering

We reverse-engineer the model by analyzing the attention weights. We show that the model has learned to perform modular addition.
We reverse-engineer the model by analyzing the attention weights. We show that the model has learned to perform modular addition.
We reverse-engineer the model by analyzing the attention weights. We show that the model has learned to perform modular addition.
We reverse-engineer the model by analyzing the attention weights. We show that the model has learned to perform modular addition.
We reverse-engineer the model by analyzing the attention weights. We show that the model has learned to perform modular addition.

\atable{data.csv}{Dataset}{data}{wide}

# Results

We reverse-engineer the model by analyzing the attention weights. We show that the model has learned to perform modular addition.
We reverse-engineer the model by analyzing the attention weights. We show that the model has learned to perform modular addition.
We reverse-engineer the model by analyzing the attention weights. We show that the model has learned to perform modular addition.

## Circuits

We see these circuits:

- Circuit 1
- Circuit 2
- Circuit 3


## Attention weights

We see these attention weights:

- Attention weight 1
- Attention weight 2


## Modular addition

We show that the model has learned to perform modular addition.
We show that the model has learned to perform modular addition.
We show that the model has learned to perform modular addition.
We show that the model has learned to perform modular addition.
We show that the model has learned to perform modular addition.
We show that the model has learned to perform modular addition.
We show that the model has learned to perform modular addition.
We show that the model has learned to perform modular addition.
We show that the model has learned to perform modular addition.
We show that the model has learned to perform modular addition.
We show that the model has learned to perform modular addition.
We show that the model has learned to perform modular addition.
We show that the model has learned to perform modular addition.
We show that the model has learned to perform modular addition.
We show that the model has learned to perform modular addition.
We show that the model has learned to perform modular addition.
We show that the model has learned to perform modular addition.
We show that the model has learned to perform modular addition.
We show that the model has learned to perform modular addition.

# Analysis

Lorem Lorem ipsum dolor sit amet, consectetur adipisci elit, sed eiusmod tempor incidunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur. Quis aute iure reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint obcaecat cupiditat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. 

\afigure{figs/attn.png}{Attention}{attn}

## Interpretability

Lorem ipsum dolor sit amet, consectetur adipisci elit, sed eiusmod tempor incidunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur. Quis aute iure reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint obcaecat cupiditat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. 

## Generalization

Lorem ipsum dolor sit amet, consectetur adipisci elit, sed eiusmod tempor incidunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur. Quis aute iure reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint obcaecat cupiditat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. 

# Conclusion

Lorem ipsum dolor sit amet, consectetur adipisci elit, sed eiusmod tempor incidunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur. Quis aute iure reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint obcaecat cupiditat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. 