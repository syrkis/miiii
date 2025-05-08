// import /////////////////////////////////////////
#import "@preview/touying:0.6.1": *
#import "@local/lilka:0.0.0": *
#show: lilka

// head ///////////////////////////////////////////
#let title = "Mechanistic Interpretability on (multi-task) Irreducible Integer Identifiers"
#let info = (author: "Noah Syrkis", date: datetime.today(), title: title)
#show: slides.with(config-info(..info), config-common(handout: false))
#metadata((title: title, slug: "miiii"))<frontmatter>


#let miiii_hash = "50115caac50c4fbfa6bce4cc"

// body //////////////////////////////////////////
#title-slide()


= Mechanistic Interpretability


#focus-slide[
  "This disgusting pile of matrices is actually just an astoundingly poorly written, elegant and consice algorithm" — Neel Nanda#footnote[Not verbatim, but the gist of it]
]

#slide[
  - Sub-symbolic nature of deep learning obscures model mechanisms #pause
  - No obvious mapping from the weights of a trained model to math notation #pause
  - MI is about reverse engineering these models, and looking closely at them #pause
  - Many low hanging fruits / practical botany phase of the science #pause
  - How does a given model work? How can we train it faster? Is it safe?
]


== Grokking

#slide[
  - Grokking @power2022 is "sudden generalization" #pause
  - MI (often) needs a mechanism #pause
  - Grokking is thus convenient for MI
  // - #cite(<lee2024a>, form: "prose", style:"american-psychological-association") speeds up grokking by boosting slow gradients as per @grokfast
  // - For more see @svd
][
  #meanwhile
  #figure(
    stack(
      image("figs/" + miiii_hash + "/acc_train_training.svg", width: 100%),
      image("figs/" + miiii_hash + "/acc_valid_training.svg", width: 100%),
    ),
    caption: [Grokking],
  )<grokking>
  // $
  // h(t) &= h(t-1) alpha + g(t)(1-alpha)\
  // hat(g)(t) &= g(t) + lambda h(t)
  // $<grokfast>
]


// #focus-slide[

// #figure(
// image("figs/neuroscope.svg", width: 75%),
// caption: [Shamleless plug: visit `github.com/syrkis/esch` for more esch plots],
// )
// ]

= Modular Arithmetic

#slide[
  - "Seminal" MI paper by #cite(<nanda2023>, form: "prose", style: "american-psychological-association") focuses on modular addition ($cal(T)_"nanda"$)
  // #footnote[Nanda worked at Anthropic under the great Chris Olah, and popularized
  // #footnote[To the extent there is such a thing as popularity in this niece a subject] MI]
  - Their final setup trains on $p=113$
  - They train a one-layer transformer
  // #footnote[MLP would have been better / simpler according to Nanda]
  - We call their task $cal(T)_"nanda"$#pause
  - And ours we call $cal(T)_"miiii"$
][
  #meanwhile
  $
    cal(T)_"nanda" = (x_0 + x_1) mod p, forall x_0, x_1 \
    cal(T)_"miiii" = (x_0 p^0 + x_1 p^1) mod q, forall q < p
  $<miiii_task>
]



#slide[
  - $cal(T)_"miiii"$ is non-commutative ...
  - ... and multi-task: $q$ ranges from 2 to 109#footnote[Largest prime less than $p=113$]
  - $cal(T)_"nanda"$ use a single layer transformer
  - Note that these tasks are synthetic and trivial to solve with conventional programming
  - They are used in the MI literature to turn black boxes opaque
]


#focus-slide[
  #figure(
    image("figs/polar.svg"),
    caption: [$NN < p^2$ multiples of 13 or 27 (left) 11 (mid.) or primes (right)],
  )
]




// #focus-slide[
//   - Modular addition is a severely solved problem
//   - The foucs is this solution—not the problem.
// ]


= Grokking on $cal(T)_"miiii"$

#slide[
  // - MI needs creativity ... #pause but there are tricks: #pause
  - For two-token samples, plot them varying one on each axis (@mi_tricks)
  - When a matrix is periodic use Fourier
  - Singular value decomposition
  // - Deep learning purposely phrases the solution space in a way that makes it incomprehensiblo
][
  #meanwhile
  #figure(
    stack(image("figs/nanda_U.svg", width: 90%), stack(
      dir: ltr,
      image("figs/neurs_113_nanda_one.svg", width: 45%),
      image("figs/neurs_113_nanda_fft_one.svg", width: 50%),
    )),
    caption: [Top singular vectors of $bold(upright(U))_W_E_cal(T)_"nanda"$ (top), varying $x_0$ and $x_1$ in sample (left) and freq. (right) space in $W_"out"_cal(T)_"miiii"$],
  )<mi_tricks>
]

#slide[

  - The model groks on $cal(T)_"miiii"$ (@training_acc)
  - Needed GrokFast #cite(<lee2024a>) on compute budget
  - Final hyperparams are seen in @hyper_param_search_result

  #figure(
    table(
      columns: (1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
      inset: 15pt,
      table.header("rate", $lambda$, "wd", $d$, "lr", "heads"),
      $1 / 10$, $1 / 2$, $1 / 3$, "256", $3 / 10^4$, "4",
    ),
    caption: [Hyperparams for $cal(T)_"miiii"$],
  )<hyper_param_search_result>
][
  #figure(
    stack(image("figs/" + miiii_hash + "/acc_train_training.svg"), image(
      "figs/" + miiii_hash + "/acc_valid_training.svg",
    )),
    caption: [Training (top) and validation (bottom) accuracy during training on $cal(T)_"miiii"$],
  )<training_acc>
]

= Embeddings

How the embedding layer deals with the difference between $cal(T)_"nanda"$ and $cal(T)_"miiii"$


== Correcting for non-commutativity

#slide[
  - The position embs. of @pos_emb reflects that $cal(T)_"nanda"$ is commutative and $cal(T)_"miiii"$ is not #pause
  - Maybe: this corrects non-comm. of $cal(T)_"miiii"$?
  - Corr. is $0.95$ for $cal(T)_"nanda"$ and $-0.64$ for $cal(T)_"miiii"$
][
  #meanwhile
  #figure(
    image("figs/pos_emb.svg", width: 100%),
    caption: [Positional embeddings for $cal(T)_"nanda"$ (top) and $cal(T)_"miiii"$ (bottom).],
  )<pos_emb>
]

== Correcting for multi-tasking

#slide[
  #meanwhile
  - For $cal(T)_"nanda"$ token embs. are essentially linear combinations of 5 frequencies ($omega$)
  - For $cal(T)_"miiii"$ more frequencies are in play
  - Each $cal(T)_"miiii"$ subtask targets unique prime
  - Possibility: One basis per prime task
][
  #figure(
    stack(spacing: 1em, image("figs/fourier_nanda_m.svg"), image(
      "figs/fourier_miiii_m.svg",
    )),
    caption: [$cal(T)_"nanda"$ (top) and $cal(T)_"miiii"$ (bottom) token embeddings in Fourier basis],
  )<tok_emb>
]

== Sanity-check and task-mask

#slide[
  - Masking $q in {2,3,5,7}$ yields we see a slight decrease in token emb. freqs.
  - Sanity check: $cal(T)_"baseline"$ has no periodicity
  - The tok. embs. encode a basis per subtask?
][
  #figure(
    stack(
      spacing: 1em,
      // image("figs/fourier_basis_m.svg"),
      image("figs/fourier_basis_m.svg"),
      image("figs/fourier_miiii_m.svg"),
      image("figs/fourier_masks_m.svg"),
    ),
    caption: [$cal(T)_"baseline"$ (top), $cal(T)_"miiii"$ (middle) and $cal(T)_"masked"$ (bottom) token embeddings in Fourier basis],
  )<tok_emb>
]

= Neurons

#slide[
  - @neurs shows transformer MLP neuron activations as $x_0$, $x_1$ vary on each axis
  - Inspite of the dense Fourier basis of $W_E_cal(T)_"miiii"$ the periodicity is clear
][
  #figure(
    stack(image("figs/neurs_113_nanda.svg"), image("figs/neurs_113_miiii.svg")),
    caption: [Activations of first three neurons for $cal(T)_"nanda"$ (top) and $cal(T)_"miiii"$ (bottom)],
  )<neurs>
]

#slide[
  - (Probably redundant) sanity check: @fft_neurs confirms neurons are periodic
  - See some freqs. $omega$ rise into significance
  - Lets log $|omega > mu_omega + 2 sigma_omega|$ while training
][
  #figure(
    stack(image("figs/neurs_113_nanda_fft.svg"), image(
      "figs/neurs_113_miiii_fft.svg",
    )),
    caption: [FFT of Activations of first three neurons for $cal(T)_"nanda"$ (top) and $cal(T)_"miiii"$ (bottom)],
  )<fft_neurs>
]


#focus-slide[
  #figure(
    stack(image("figs/neurs_113_basis.svg", width: 50%), image(
      "figs/neurs_113_basis_fft.svg",
      width: 50%,
    )),
    caption: [Neurons as archive for $cal(T)_"basline"$],
  )<ava>
]

#focus-slide[
  #figure(
    stack(image("figs/neurs_113_miiii.svg", width: 50%), image(
      "figs/neurs_113_miiii_fft.svg",
      width: 50%,
    )),
    caption: [Neurons as algorithm $cal(T)_"miiii"$],
  )<algorithm>
]

#focus-slide[
  #figure(
    stack(
      // image("figs/neurs_113_miiii_fft.svg"),
      image("figs/miiii_large_finding.svg", width: 100%),
    ),
    caption: [Number of neurons with frequency $omega$ above the theshold $mu_omega + 2 sigma_omega$],
  )<spike>
]


= The $omega$-Spike

#slide[
  // - We reach our central finding
  - Neurs. periodic on solving $q in {2,3,5,7}$
  - When we generalize to the reamining tasks, many frequencies activate (64-sample)
  // - Quickly after generalization $omega$'s merge
  - Those $omega$'s are not useful for memory and not useful after generalization
  #figure(
    table(
      columns: (1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
      align: center,
      inset: 10pt,
      table.header([time], "256", "1024", "4096", "16384", "65536"),
      $bold(|omega|)$, $0$, $0$, $10$, $18$, "10",
    ),
    caption: [active $omega$'s through training],
  )<tableeeee>
][
  #figure(
    stack(image("figs/miiii_large_finding.svg", width: 96.5%), image(
      "figs/" + miiii_hash + "/acc_valid_training.svg",
      width: 100%,
    )),
    caption: [@spike (top) and validation accuracy from @training_acc (bottom)],
  )
]

#slide[
  - GrokFast @lee2024a shows time gradient sequences is (arguably) a stocastical signal with:
    - A fast varying overfitting component
    - A slow varying generealizing component
  - My work confirms this to be true for $cal(T)_"miiii"$ ...
  - ... and observes a strucutre that seems to fit _neither_ of the two
]

#slide[
  - Future work:
    - Modify GrokFast to assume a third stochastic component
    - Relate to signal processing literature
    - Can more depth make tok-embedding sparse?
]


#focus-slide[
  TAK
]

#[
  #show heading.where(level: 1): set heading(numbering: none)
  = References <touying:unoutlined>
  #set align(top)
  #pad(y: 2em, bibliography("zotero.bib", title: none, style: "ieee"))
]

#appendix[

  = Stochastic Signal Processing

  We denote the weights of a model as $theta$. The gradient of $theta$ with respect to our loss function at time $t$ we denote $g(t)$.
  As we train the model, $g(t)$ varies, going up and down. This can be thought of as a stocastic signal.
  We can represent this signal with a Fourier basis. GrokFast posits that the slow varying frequencies contribute to grokking.
  Higer frequencies are then muted, and grokking is indeed accelerated.

  = Discrete Fourier Transform

  Function can be expressed as a linear combination of cosine and sine waves.
  A similar thing can be done for data / vectors.

  = Singular Value Decomposition <svd>

  An $n times m$ matrix $M$ can be represented as a $U Sigma V^*$, where $U$ is an $m times m$ complex unitary matrix, $Sigma$ a rectangular $m times n$ diagonal matrix (padded with zeros), and $V$ an $n times n$ complex unitary matrix. Multiplying by $M$ can thus be viewed as first rotating in the $m$-space with $U$, then scaling by $Sigma$ and then rotating by $V$ in the $n$-space.
]
