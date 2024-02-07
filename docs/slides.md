---
title: Mechanistic Interpretability on Irreducible Integers
type: slides
author: Noah Syrkis
---

# Mech. interp. (MI)

- Reverse-engineering neural network circuits.
- @nanda2023 shows MI modular addition transformer.
- There are (allegedly) low hanging fruits in MI.

# Grokking

- Grokking is when a model suddenly generalises.
- @nanda2023 shows grokking in a transformer.
- Grokking means the weights represents an algorithm...
- ... rather than a weired data base.

\framebreak

- Since MI is about reverse-engineering circuits...
- ... grokking is a good sign for MI ...
- ... as it means circuits are _there_.

# $\mathbb{Z}$-sequences

- @belcak2022 shows that transformers can sequences $\in\mathbb{Z}$.
- They work in thousands of squences from OEIS [@sloane2003].
- They have four tasks: (1) sequence classification, (2) sequence comparission, (3) sequence continuation, and (4) sequence unmasking.
- Each task is strictly harder than the previous one.

\framebreak

- Though $\mathbb{Z}$-sequences are simple to see, some can be hard to impossible to understand.
- $1, 2, 3, ..., 100$ is easy, while the busy beaver sequence [@aaronson2020] is hard/impossible.
- Complexity ranges from trivial to fuck-off-forever.


# MIII

- I want to explore the use of MI on $\mathbb{Z}$-sequences.
- Initially, I want to explore the classification task...
- ... with possibility of moving up in task complexity.