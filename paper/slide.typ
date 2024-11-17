// head /////////////////////////////////////////////////////////////////////////
#import "@preview/touying:0.5.3": *
#import "@preview/lovelace:0.3.0": *
#import "@local/esch:0.0.0": *
#import "@preview/gviz:0.1.0": * // for rendering dot graphs
#import "@preview/finite:0.3.0": automaton // for rendering automata
#import "@preview/fletcher:0.5.1" as fletcher: diagram, node, edge, shapes
#import "@preview/equate:0.2.1": equate // <- for numbering equations
#import "@preview/plotst:0.2.0": axis, plot, graph_plot, overlay


#let title = [Mechanistic Interpretability and Implementability of Irreducible Integer Identifiers (MIIIII)]
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


#focus-slide[
  "This disgusting pile of matrices is actually just an astoundingly poorlu written, elegant and consice algorithm" — Neel Nanda#footnote[Not verbatim, but the gist of it]
]


= Task Design

#slide[
  - Deep model mapping $X -> Y$
  - MI work @nanda2023 on modular addition (@modular_addition)
  - A general algorithm was grokked @power2022 ...
  - ... and reverse engineered.
][
  $
    (x_0 + x_1) mod p &= y, quad forall x_0, x_1 < p #<modular_addition>\
  $
]




#slide[
  - A next step: new task? multi task?
  - Approximately $n / ln(n)$ primes less than n
  - A number $n$ not a multiple of any prime less than $sqrt(n)$ is prime
  - Test remainder on primes $p'<p$ (@new_task) for all two digit base-$p$ numbers.
][
  $
    (x_0 p^0 + x_1 p^1) mod p' = y, quad forall p' < p #<new_task>\
  $
]

#focus-slide[
  $
    [
      X|Y
    ] &= mat(
      augment: #(vline: 2),
      delim:"[",
      0, 0, 0, dots.h, 103;
      0, 1, 1, dots.h, 103;
      dots.v, dots.v, dots.v, dots.down, dots.v;
      112, 112, 111, dots.h, 103;
  )#<nanda_matrix>
  $
]

#focus-slide[
  #figure(
    stack(
      dir: ttb,
      image("figs/ds_miiii_11_x.svg", width: 100%),
      image("figs/y_11_plot.svg", width: 100%),
    ),
    caption: [Representation of $X$ (top) and $Y$ (bottom) for $p=11$],
  )
]

= Mechanistic Interpretability

= The Embryology of Circuits

= MIIIII

#focus-slide[
  #figure(
    image("figs/ffwd_37_4.svg"),
    caption: [Top left $37 times 37$ slice of first four $W_("out")$ neurons],
  )
]


#set align(top)
#show heading.where(level: 1): set heading(numbering: none)
= References <touying:unoutlined>
#bibliography("zotero.bib", title: none, style: "ieee")
