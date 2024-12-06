// head /////////////////////////////////////////////////////////////////////////
#import "@preview/touying:0.5.3": *
#import "@local/esch:0.0.0": *
#import "@preview/equate:0.2.1": equate // <- for numbering equations


// HASHES
#let miiii_hash = "50115caac50c4fbfa6bce4cc"


#let title = [Mechanistic Interpretability on multi-task Irreducible Integer Identifiers]
#show: escher-theme.with(
  aspect-ratio: "16-9",
  config-info(author: "Noah Syrkis", date: datetime.today(), title: title),
  config-common(handout: true),   // <- for presentations
)

#show: equate.with(breakable: true, sub-numbering: true)
#set math.equation(numbering: "(1.1)", supplement: "Eq.")

#show figure.caption: emph

// body /////////////////////////////////////////////////////////////////////////
#cover-slide()

#focus-slide[
  #figure(
    image("figs/polar.svg"),
    caption: [$NN < p^2$ multiples of 13 or 27 (L), 11 (M), or primes (R)],
  )
]

= Mechanistic Interpretability (MI)


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
  - MI neceitates a mechanism #pause
  - Grokking is thus convenient for MI #pause
  - #cite(<lee2024a>, form: "prose", style:"american-psychological-association") speeds up grokking by boosting slow gradients as per @grokfast
][
  // #meanwhile
  $
    h(t) &= h(t-1) alpha + g(t)(1-alpha)\
    hat(g)(t) &= g(t) + lambda h(t)
  $<grokfast>
]

== Visualizing

#slide[
  - MI use creativity ... #pause but there are tricks: #pause
    - For two-token samples, plot them varying one on each axis (@mi_tricks) #pause
    - When a matrix is periodic use Fourier
    - Singular value decomp. ($upright(bold(M)) = upright(bold(U)) upright(bold(Sigma))upright(bold(V^*))$)
][
  #meanwhile
  #figure(
    stack(
      image("figs/nanda_U.svg", width: 90%),
      stack(
        dir: ltr,
        image("figs/neurs_113_nanda_one.svg", width: 45%),
        image("figs/neurs_113_nanda_fft_one.svg", width: 50%),
      ),
    ),
    caption: [Top singular vectors of $bold(upright(U))_W_E_cal(T)_"nanda"$ (top), varying $x_0$ and $x_1$ in sample (left) and freq. (right) space in $W_"out"_cal(T)_"miiii"$],
  )<mi_tricks>
]

= Modular Arithmetic

#slide[
  - "Seminal" MI paper by #cite(<nanda2023>, form: "prose", style:"american-psychological-association") focus on modular additon (@nanda_task)
    // #footnote[Nanda worked at Anthropic under the great Chris Olah, and popularized
    // #footnote[To the extent there is such a thing as popularity in this niece a subject] MI]
  - Their final setup trains on $p=113$
  - They train a one-layer transformer
  // #footnote[MLP would have been better / simpler according to Nanda]
  - We call their task $cal(T)_"nanda"$#pause
  - And ours, seen in @miiii_task, we call $cal(T)_"miiii"$
][
  #meanwhile
  $
    (x_0 + x_1) mod p, quad forall x_0, x_1
  $<nanda_task>
  #pause
  $
    (x_0 p^0 + x_1 p^1) mod q, quad forall q < p
  $<miiii_task>
]


// #focus-slide[
//   - give architecutre
//   - What is a transformer?
// ]


= Grokking on $cal(T)_"miiii"$

#slide[

  - The model groks on $cal(T)_"miiii"$ (@training_acc)
  - Needed GrokFast #cite(<lee2024a>) on compute budget
  - Final hyperparams are seen in @hyper_param_search_result

  #figure(
    table(
      columns: (1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
      inset: 15pt,
      table.header(
        "rate",
        $lambda$,
        "wd",
        $d$,
        "lr",
        "heads",
      ),

      $1 / 10$, $1 / 2$, $1 / 3$, "256", $3 / 10^4$, "4",
    ),
    caption: [Hyperparams for $cal(T)_"miiii"$],
  )<hyper_param_search_result>
][
  #figure(
    stack(
      image("figs/" + miiii_hash + "/acc_train_training.svg"),
      image("figs/" + miiii_hash + "/acc_valid_training.svg"),
    ),
    caption: [Training (top) and validation (bottom) accuracy during training on $cal(T)_"miiii"$],
  )<training_acc>
]

= Embeddings

#slide[
  - The pos. embs. of @pos_emb shows $cal(T)_"nanda"$ is commutative and $cal(T)_"miiii"$ is not
  - Pearsons correlation is $0.95$ for $cal(T)_"nanda"$ and $-0.64$ for $cal(T)_"miiii"$
  - Conjecture: The pos. embs. correct for non-commutativity of $cal(T)_"miiii"$
][
  #figure(
    image("figs/pos_emb.svg", width: 100%),
    caption: [Positional embeddings for $cal(T)_"nanda"$ (top) and $cal(T)_"miiii"$ (bottom).],
  )<pos_emb>
]

== Token Embeddings

#slide[
  - For $cal(T)_"nanda"$ token embs. are essentially linear combinations of 5 frequencies ($omega$)
  - For $cal(T)_"miiii"$ more frequencies are in play
  - Each $cal(T)_"miiii"$ subtask targets unique prime
  - Possibility: One basis per prime task
][
  #figure(
    stack(
      spacing: 1em,
      // image("figs/fourier_basis_m.svg"),
      image("figs/fourier_nanda_m.svg"),
      image("figs/fourier_miiii_m.svg"),
    ),
    caption: [$cal(T)_"nanda"$ (top) and $cal(T)_"miiii"$ (bottom) token embeddings in Fourier basis],
  )<tok_emb>
]

#slide[
  - Masking $q in {2,3,5,7}$ yields we see a slight decrease in token emb. freqs. #pause
  - Sanity check: $cal(T)_"bline"$ has no periodicity
  - Conjecture: The tok. embs. encode a basis per subtask
][
  #figure(
    stack(
      spacing: 1em,
      // image("figs/fourier_basis_m.svg"),
      image("figs/fourier_masks_m.svg"),
      image("figs/fourier_basis_m.svg"),
    ),
    caption: [$cal(T)_"nanda"$ (top) and $cal(T)_"miiii"$ (bottom) token embeddings in Fourier basis],
  )<tok_emb>
]

= Neurons

#slide[
  - @neurs shows transformer MLP neuron activations as $x_0$, $x_1$ vary on each axis
  - Inspite of the dense Fourier basis of $W_E_cal(T)_"miiii"$ the periodicity is clear
][
  #figure(
    stack(
      image("figs/neurs_113_nanda.svg"),
      image("figs/neurs_113_miiii.svg"),
    ),
    caption: [Activations of first three neurons for $cal(T)_"nanda"$ (top) and $cal(T)_"miiii"$ (bottom)],
  )<neurs>
]

#slide[
  - (Probably redundant) sanity check: @fft_neurs confirms neurons are periodic
  - See some freqs. $omega$ rise into significance
  - Lets log $|omega > mu_omega + 2 sigma_omega|$ while training
][
  #figure(
    stack(
      image("figs/neurs_113_nanda_fft.svg"),
      image("figs/neurs_113_miiii_fft.svg"),
    ),
    caption: [FFT of Activations of first three neurons for $cal(T)_"nanda"$ (top) and $cal(T)_"miiii"$ (bottom)],
  )<fft_neurs>
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


= $omega$-Spike

#slide[
  // - We reach our central finding
  - Neurs. periodic on solving $q in {2,3,5,7}$
  - When we generalize to the reamining tasks, many frequencies activate
  // - Quickly after generalization $omega$'s merge
  - Those $omega$'s are not useful for memory and not useful after generalization
  #figure(
    table(
      columns: (1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
      align: center,
      inset: 10pt,
      table.header(
        [*epoch*],
        "256",
        "1024",
        "4096",
        "16384",
        "65536",
      ),

      $bold(|omega|)$, $0$, $0$, $10$, $18$, "10",
    ),
    caption: [active $omega$'s through training],
  )<tableeeee>
][
  #figure(
    stack(
      image("figs/miiii_large_finding.svg", width: 96.5%),
      image("figs/" + miiii_hash + "/acc_valid_training.svg", width: 100%),
      // image("figs/" + miiii_hash + "/acc_train_training.svg", width: 75%),
    ),
    caption: [@spike (top) and validation accuracy from @training_acc (bottom)],
  )
]



// #v(2em)
// #pause
// #figure(
// stack(
// image("figs/nanda_u.svg", width: 80%),
// pause,
// image("figs/fourier_nanda_m.svg", width: 80%),
// image("figs/neurs_113_nanda_fft_three.svg", width: 70%),
// ),
// caption: [Top 50 % vectors of $upright(bold(U))$],
// )
// #pause
// #v(2em)
// Behold, periodicity!
// #pagebreak()
// #image("figs/fourier_nanda_m.svg")


// #slide[
// #figure(
// image("figs/fourier_nanda_m.svg"),
// caption: [All 128-d embeddings are a combination of only five periodic functions!],
// )
// ]

// #slide[
// - $W_E$ is a sine and cosine table
// - Sends periodic representation to transformer block layer
// ][
// $
// x_0 &-> sin(w x_0), cos(w x_0) \
// x_1 &-> sin(w x_1), cos(w x_1) \
// $
//
// ]

// #focus-slide[
// #figure(
// stack(
// image("figs/neurs_113_nanda_three.svg", width: 70%),
// v(1em),
// image("figs/neurs_113_nanda_fft_three.svg", width: 65%),
// ),
// caption: [Neurons in sample (T) and frquency (B) space],
// )
// ]


// #focus-slide[
// - Months of #cite(<nanda2023>, form: "author", style: "american-psychological-association") probing, plotting and ablating
// -

// periodicity

// Practising RASP is a good start #cite(
// <weiss2021>,
// form: "prose",
// style: "american-psychological-association",
// )

// ]



// #slide[
//   // - periodicity #pause
//   $
//     sin(w(x_0 + x_1)) &= sin(w x_0) cos(w x_0) + cos(w x_0) sin(w x_1) \
//     cos(w(x_0 + x_1)) &= cos(w x_1) cos(w x_1) - sin(w x_0) sin(w x_1) \
//     &= cos(w(x_0 + x_1)) cos(w c) + sin(w(x_0 + x_1)) sin(w c) \
//     "Logit"(c) &prop cos(w(x_0 + x_1 - c)) \
//   $<nanda_last_layer>
// ]


// = Multi-Task (and Modulo) Learning

// #slide[
// - My work extends $cal(T)_"nanda"$ with $cal(T)_"miiii"$
// - $cal(T)_"miiii"$ computes two-digit base-$p$ modulo remainders for all primes $q<p$
// - Multi-task and non-commutative (@non_commut)
// ][
//
// $
// (x_0 p^0 + x_1 p^1) != (x_1 p^0 + x_0 p^1)
// $<non_commut>
// ]
//
// #focus-slide[
// #figure(
// image("figs/y_11_plot.svg", width: 100%),
// caption: [Representation of target when $p=11$],
// )
// ]
//
//
// = Model Architecture / Training
//
//
// #slide[
// - One layer transformer model
// - No bias or layernorm (as #cite(<nanda2023>, form: "author", style: "american-psychological-association"))
// - Trained for 60k+ full batch epochs
// - Gradient boost as per #cite(<lee2024a>, form: "prose", style: "american-psychological-association")
//
// ][
// ]
//
//
// #figure(
// image("figs/" + miiii_hash + "/acc_train_training.svg"),
// caption: [$NN < p^2$ multiples of 13 or 27 (L), 11 (M), or primes (R)],
// )
//
// = Embedding Representations
//
// #slide[
// - Non-commutativity is picked up
// - But what about periodicity?#pause
// - $cal(T)_"miiii"$ focus on 29 primes#pause
// - No overlaping steps when rotating in $CC$.
// #meanwhile
// ][
//
// #figure(
// stack(
// ),
// caption: [$cal(T)_"miiii"$ is non-commutative],
// )
// ]
//
// #slide[
// #figure(
// stack(
// image("figs/miiii_u.svg"),
// ),
// caption: [many relevant#footnote[and possibly periodic?] singular values in $upright(bold(u))_cal(t)_"miiii"$],
// )
// ]
//
// #focus-slide[
// #figure(
// stack(
// image("figs/fourier_miiii_m.svg", width: 90%),
// v(1em),
// image("figs/fourier_basis_m.svg", width: 90%),
// ),
// caption: [$"DFT" dot W_E_cal(T)_"miiii"$ (top) and $"DFT"dot W_E_cal(T)_"bline"$ (bottom)],
// )
// ]
//
// #focus-slide[
// $W_E_cal(T)_"miiii"$ is periodic, but a lot more frequencies in play
// ]
// - Each prime $q<p$ reaches unique points around the unit circle
//
//
//
// = Neuron Periodicity
//
// #slide[
// - $W_"out"$ periodic in $cal(T)_"miiii"$
// - Not so for $cal(T)_"bline"$
// ][
//
// #figure(
// stack(
// image("figs/neurs_113_miiii_one.svg", width: 70%),
// image("figs/neurs_113_miiii_fft_one.svg", width: 60%),
// ),
// caption: [$cal(T)_"miiii"$ (periodic)],
// )
// ][
// #figure(
// stack(
// image("figs/neurs_113_basis_one.svg", width: 70%),
// image("figs/neurs_113_basis_fft_one.svg", width: 60%),
// ),
// caption: [$cal(T)_"bline"$ (no period)],
// )
//
// ]
//
// #focus-slide[
// #figure(
// image("figs/neurs_113_miiii_three.svg"),
// caption: [More periodic $cal(T)_"miiii"$ neuron activations],
// )
// ]
//
// = Embryology of Neural Circuits
//
// #slide[
// - Freqs. $omega$ in $W_"out"_cal(T)_"miiii"$ follows loss
// 1. Performace increases
// 2. Afterwards, circuits proliferate
// 3. Finally frequencies decrease (merge)
// ][
// #figure(
// stack(
// image("figs/miiii_large_finding.svg", width: 100%),
// image("figs/miiii_large_finding.svg", width: 130%),
// image("figs/" + miiii_hash + "/acc_valid_training.svg", width: 104%),
// ),
// caption: [Spike and generalization],
// )
// ]
//
// #focus-slide[
// #figure(
// image("figs/miiii_small_finding.svg"),
// caption: [Active frequencies in $W_"out"_cal(T)_"miiii"$ in time],
// )
// #v(2em)
// #figure(
// table(
// columns: (1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
// align: center,
// inset: 15pt,
// table.header(
// "epoch",
// "256",
// "1024",
// "4096",
// "16384",
// "65536",
// ),
//
// $|omega|$, $0$, $0$, $10$, $18$, "10",
// ),
// caption: [Active frequencies in $W_E_cal(T)_"miiii"$ in time],
// )<tableeeee>
// ]

#set align(top)
#show heading.where(level: 1): set heading(numbering: none)
= References <touying:unoutlined>
#bibliography("zotero.bib", title: none, style: "ieee")
