// head /////////////////////////////////////////////////////////////////////////
#import "@preview/touying:0.5.3": *
#import "@preview/lovelace:0.3.0": *
#import "@local/esch:0.0.0": *
#import "@preview/gviz:0.1.0": * // for rendering dot graphs
#import "@preview/finite:0.3.0": automaton // for rendering automata
#import "@preview/fletcher:0.5.1" as fletcher: diagram, node, edge, shapes
#import "@preview/equate:0.2.1": equate // <- for numbering equations
#import "@preview/plotst:0.2.0": axis, plot, graph_plot, overlay


// HASHES
#let miiii_hash = "50115caac50c4fbfa6bce4cc"


#let title = [Mechanistic Interpretability on multi-task Irreducible Integer Identifiers]
#show: escher-theme.with(
  aspect-ratio: "16-9",
  config-info(author: "Noah Syrkis", date: datetime.today(), title: title),
  config-common(handout: false),   // <- for presentations
)
#show raw.where(lang: "dot-render"): it => render-image(it.text)
#show: equate.with(breakable: true, sub-numbering: false)
#set math.equation(numbering: "(1.1)", supplement: "Eq.")

#show figure.caption: emph

// body /////////////////////////////////////////////////////////////////////////
#cover-slide()




= Mechanistic Interpretability (MI)

#focus-slide[
  "This disgusting pile of matrices is actually just an astoundingly poorly written, elegant and consice algorithm" — Neel Nanda#footnote[Not verbatim, but the gist of it]
]


#slide[
  - Sub-symbolic nature of deep learning obscures model mechanisms
  - MI is about reverse engineering these
]

#slide[
  - #cite(<nanda2023>, form: "prose", style:"american-psychological-association") perform MI on a model doing modular additon (@nanda_task)
  - Their final setup trains on $p=113$
  - We call this task $cal(T)_"nanda"$

][

  #v(3em)
  $
    (x_0 + x_1) mod p, quad forall x_0, x_1
  $<nanda_task>
]

#focus-slide[
  #v(2em)
  Recall a matrix $upright(bold(M))$ = $upright(bold(U))upright(bold(Sigma))upright(bold(V^*))$
  #v(2em)
  #pause
  #figure(
    stack(
      image("figs/nanda_u.svg", width: 80%),
      // pause,
      // image("figs/fourier_nanda_m.svg", width: 80%),
      // image("figs/neurs_113_nanda_fft_three.svg", width: 70%),
    ),
    caption: [Top 50 % vectors of $upright(bold(U))$],
  )#pause
  #v(2em)
  Behold, periodicity!
  // #pagebreak()
  // #image("figs/fourier_nanda_m.svg")
]


#focus-slide[
  #figure(
    image("figs/fourier_nanda_m.svg"),
    caption: [All 128-d embeddings are a combination of only five periodic functions!],
  )
]

#slide[
  - $W_E$ is a sine and cosine table
  - Sends periodic representation to transformer block layer
][
  $
    x_0 &-> sin(w x_0), cos(w x_0) \
    x_1 &-> sin(w x_1), cos(w x_1) \
  $

]

#focus-slide[
  #figure(
    stack(
      image("figs/neurs_113_nanda_three.svg", width: 70%),
      // pause,
      v(1em),
      image("figs/neurs_113_nanda_fft_three.svg", width: 65%),
      // meanwhile,
    ),
    caption: [Neurons in sample (T) and frquency (B) space],
  )
]


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


= Multi-Task (and Modulo) Learning

#slide[
  - My work extends $cal(T)_"nanda"$ with $cal(T)_"miiii"$
  - $cal(T)_"miiii"$ computes two-digit base-$p$ modulo remainders for all primes $q<p$ (@miiii_task)
  - Multi-task and non-commutative (@non_commut)
][
  $
    (x_0 p^0 + x_1 p^1) mod q, quad forall q < p
  $<miiii_task>
  $
    (x_0 p^0 + x_1 p^1) != (x_1 p^0 + x_0 p^1)
  $<non_commut>
]

#focus-slide[
  #figure(
    image("figs/y_11_plot.svg", width: 100%),
    caption: [Representation of target when $p=11$],
  )
]

#focus-slide[
  #figure(
    image("figs/polar.svg"),
    caption: [$NN < p^2$ multiples of 13 or 27 (L), 11 (M), or primes (R)],
  )
]

// - **Setup**: Multi-task learning with modular arithmetic tasks.
// - **Tasks**: Predict remainders for two-digit base-113 numbers by primes less than 113.
// - **Complexity**: Varies from binary classification (mod 2) to 109-way classification (mod 109).
// - **Benefits**: Encourages generalization, reduces overfitting.

= Model Architecture / Training


#slide[
  - One layer transformer model
  - No bias or layernorm (as #cite(<nanda2023>, form: "author", style: "american-psychological-association"))
  - Trained for 60k+ full batch epochs
  - Gradient boost as per #cite(<lee2024a>, form: "prose", style: "american-psychological-association")

][
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
    caption: [Final hyper-parameters for $cal(T)_"miiii"$],
  )<hyper_param_search_result>
]


#figure(
  image("figs/" + miiii_hash + "/acc_train_training.svg"),
  caption: [$NN < p^2$ multiples of 13 or 27 (L), 11 (M), or primes (R)],
)

= Embedding Representations

#slide[
  - Non-commutativity is picked up
  - But what about periodicity?#pause
  - $cal(T)_"miiii"$ focus on 29 primes#pause
  - No overlaping steps when rotating in $CC$.
  #meanwhile
][

  #figure(
    stack(
      image("figs/pos_emb.svg", width: 120%),
    ),
    caption: [$cal(T)_"miiii"$ is non-commutative],
  )
]

#slide[
  #figure(
    stack(
      image("figs/miiii_u.svg"),
    ),
    caption: [many relevant#footnote[and possibly periodic?] singular values in $upright(bold(u))_cal(t)_"miiii"$],
  )
]

#focus-slide[
  #figure(
    stack(
      image("figs/fourier_miiii_m.svg", width: 90%),
      v(1em),
      image("figs/fourier_basis_m.svg", width: 90%),
    ),
    caption: [$"DFT" dot W_E_cal(T)_"miiii"$ (top) and $"DFT"dot W_E_cal(T)_"bline"$ (bottom)],
  )
]

#focus-slide[
  $W_E_cal(T)_"miiii"$ is periodic, but a lot more frequencies in play
]
// - Each prime $q<p$ reaches unique points around the unit circle



= Neuron Periodicity

#slide[
  - Like $cal(T)_"nanda"$, $W_"out"$ neurons is periodic for $cal(T)_"miiii"$
  - Not so for $cal(T)_"bline"$
][

  #figure(
    stack(
      image("figs/neurs_113_miiii_one.svg", width: 70%),
      image("figs/neurs_113_miiii_fft_one.svg", width: 60%),
    ),
    caption: [$cal(T)_"miiii"$ (periodic)],
  )
][
  #figure(
    stack(
      image("figs/neurs_113_basis_one.svg", width: 70%),
      image("figs/neurs_113_basis_fft_one.svg", width: 60%),
    ),
    caption: [$cal(T)_"bline"$ (no period)],
  )

]

#focus-slide[
  #figure(
    image("figs/neurs_113_miiii_three.svg"),
    caption: [More periodic $cal(T)_"miiii"$ neuron activations],
  )
]

= Embryology of Neural Circuits

#slide[
  // - Freqs. $omega$ in $W_"out"_cal(T)_"miiii"$ follows loss
  1. Performace increases
  2. Afterwards, circuits proliferate
  3. Finally frequencies decrease (merge)
][
  #figure(
    stack(
      image("figs/miiii_large_finding.svg", width: 100%),
      // image("figs/miiii_large_finding.svg", width: 130%),
      image("figs/" + miiii_hash + "/acc_train_training.svg", width: 104%),
    ),
    caption: [],
  )
]

#focus-slide[
  #figure(
    image("figs/miiii_small_finding.svg"),
    caption: [Active frequencies in $W_"out"_cal(T)_"miiii"$ in time],
  )
  #v(2em)
  #figure(
    table(
      columns: (1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
      align: center,
      inset: 15pt,
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
    caption: [Active frequencies in $W_E_cal(T)_"miiii"$ in time],
  )<tableeeee>
]

//
// #slide[
// - Deep model mapping $X -> Y$
// - MI work @nanda2023 on modular addition (@modular_addition)
// - A general algorithm was grokked @power2022 ...
// - ... and reverse engineered.
// ][
// $
// (x_0 + x_1) mod p &= y, quad forall x_0, x_1 < p #<modular_addition>\
// $
// ]
//
//
//
//
// #slide[
// - A next step: new task? multi task?
// - Approximately $n / ln(n)$ primes less than n
// - A number $n$ not a multiple of any prime less than $sqrt(n)$ is prime
// - Test remainder on primes $p'<p$ (@new_task) for all two digit base-$p$ numbers.
// ][
// $
// (x_0 p^0 + x_1 p^1) mod p' = y, quad forall p' < p #<new_task>\
// $
// ]
//
// #focus-slide[
// $
// [
// X|Y
// ] &= mat(
// augment: #(vline: 2),
// delim:"[",
// 0, 0, 0, dots.h, 103;
// 0, 1, 1, dots.h, 103;
// dots.v, dots.v, dots.v, dots.down, dots.v;
// 112, 112, 111, dots.h, 103;
// )#<nanda_matrix>
// $
// ]
//
// #focus-slide[
// #figure(
// stack(
// dir: ttb,
// image("figs/ds_miiii_11_x.svg", width: 100%),
// image("figs/y_11_plot.svg", width: 100%),
// ),
// caption: [Representation of $X$ (top) and $Y$ (bottom) for $p=11$],
// )
// ]
//
// = Mechanistic Interpretability
//
// = The Embryology of Circuits
//
// = MIIIII
//
// #focus-slide[
// #figure(
// image("figs/ffwd_37_4.svg"),
// caption: [Top left $37 times 37$ slice of first four $W_("out")$ neurons],
// )
// ]
//
//
#set align(top)
#show heading.where(level: 1): set heading(numbering: none)
= References <touying:unoutlined>
#bibliography("zotero.bib", title: none, style: "ieee")
//
