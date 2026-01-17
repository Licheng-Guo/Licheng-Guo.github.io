---
title: "Understanding Rotary Position Embedding (RoPE)"
date: 2026-01-05
tags: ["machine-learning", "transformers", "attention"]
math: true
---

## The Problem: Vanilla Attention Ignores Position

In vanilla GPT-2, we compute how much influence a prior token should have on a later token by taking the dot product of a query vector with a key vector:

$$\text{score} = q \cdot k$$

However, this approach produces the same score regardless of the distance between two tokens. The same pair of tokens will end up with the same correlation score whether they are right next to each other or 100 tokens apart. In other words, the weight computed this way does not encode any information about relative position.

![Vanilla attention ignores position](/images/rope/vanilla-attention.png)

## The Core Idea: Twist the Dot Product by Distance

The idea behind RoPE is essentially asking: can we "twist" how we compare two vectors based on how far apart they are?

Instead of a single dot product, can we create a "distance-2 dot product" when two tokens are two positions apart, a "distance-3 dot product" when three apart, and so on? If we can do this, the positional information will be reflected in the computed attention weight, and the model can learn to use this information during training.

![Twist dot product by distance](/images/rope/distance-aware-dot-product.png)

## The Solution: Rotate Vectors by Their Position

Recall that for two vectors $\vec{a}$ and $\vec{b}$, the dot product is:

$$\vec{a} \cdot \vec{b} = |\vec{a}||\vec{b}|\cos\phi$$

where $\phi$ is the angle between the two vectors.

RoPE's clever trick is to **rotate each vector according to its position** before computing the dot product. If we rotate $\vec{a}$ (at position $m$) by angle $m\theta$ and rotate $\vec{b}$ (at position $n$) by angle $n\theta$, then the angle between the rotated vectors becomes:

$$\phi_{\text{new}} = \phi + (n - m)\theta$$

The resulting dot product is:

$$\vec{a}_{\text{rot}} \cdot \vec{b}_{\text{rot}} = |\vec{a}||\vec{b}|\cos\big(\phi + (n-m)\theta\big)$$

Now the dot product depends on the **relative distance** $(n - m)$, not the absolute positions. Different distances produce different scores, giving the transformer a way to distinguish them.

![Rotate vectors by position](/images/rope/vector-rotation.png)

### Why Relative Position Emerges

This deserves emphasis. If query $q$ is at position $m$ and key $k$ is at position $n$:

$$q_m = R(m\theta) \cdot q, \quad k_n = R(n\theta) \cdot k$$

where $R(\alpha)$ denotes a rotation matrix by angle $\alpha$. The dot product becomes:

$$q_m^T k_n = q^T R(-m\theta) R(n\theta) k = q^T R\big((n-m)\theta\big) k$$

The absolute positions $m$ and $n$ cancel out, leaving only the difference $(n - m)$. This is the mathematical heart of RoPE.

## The Aliasing Problem: Single Frequency Fails

The naive implementation has a critical limitation: the cosine function is periodic.

Say we choose $\theta = 30°$ per position. Then:

|Distance|Rotation Difference|Cosine Value|
|---|---|---|
|1|30°|0.866|
|13|390° = 30°|0.866|

Distance 1 and distance 13 produce identical scores — they become indistinguishable. This is an **aliasing** problem.

One idea might be to make $\theta$ very small, say $1°$, so the period becomes 360 positions. But that creates a different problem:

|Distance|Rotation Difference|Cosine Value|
|---|---|---|
|1|1°|0.99985|
|2|2°|0.99939|

Now distance-1 and distance-2 pairs are almost indistinguishable — the signal is too weak.

We face a dilemma:

- **High frequency (large θ):** Nearby positions are distinct, but distant positions collide
- **Low frequency (small θ):** Distant positions are distinct, but nearby positions blur together

No single frequency works for all distance scales.

![Aliasing problem](/images/rope/aliasing-problem.png)

## The Multi-Frequency Solution

To solve this, RoPE rotates **different segments of the vector at different rates**.

### Partitioning into Pairs

A $d$-dimensional vector is split into $d/2$ consecutive pairs. Each pair is treated as an independent 2D vector:

$$\vec{a} = \underbrace{[a_0, a_1]}_{\text{Pair 0}}, \underbrace{[a_2, a_3]}_{\text{Pair 1}}, \ldots, \underbrace{[a_{d-2}, a_{d-1}]}_{\text{Pair } d/2-1}$$

Why pairs of 2? Because 2D is the minimum dimension where rotation is meaningful. A 1D "rotation" would just be a sign flip, which cannot encode continuous position information.

### Different Frequencies per Pair

Each pair $j$ is assigned its own rotation frequency:

$$\theta_j = 10000^{-2j/d}$$

This creates a geometric progression:

|Pair|Frequency|Wavelength (positions per full rotation)|
|---|---|---|
|0 (first)|$\theta_0 = 1$|~6 positions|
|1|$\theta_1 \approx 0.1$|~63 positions|
|...|...|...|
|$d/2-1$ (last)|$\theta_{d/2-1} \approx 0.0001$|~63,000 positions|

**Early pairs (low index) rotate fast** — they're sensitive to nearby positions but alias quickly.

**Later pairs (high index) rotate slowly** — they distinguish distant positions but blur nearby ones.

Together, they cover all distance scales without tradeoffs.

### The Clock Analogy

This is exactly how a clock encodes time:

- **Second hand (high frequency):** Distinguishes moments within a minute, but 1:00:30 and 1:01:30 have the same second hand position
- **Minute hand (medium frequency):** Distinguishes times within an hour
- **Hour hand (low frequency):** Distinguishes times across the day

No single hand can uniquely identify all times, but together they form an unambiguous representation.

### Computing the Rotated Vector

At position $m$, each pair rotates by its own angle:

$$\text{Pair } j \text{ rotates by } m \times \theta_j$$

The rotation is applied independently to each pair:

$$\begin{pmatrix} a'_{2j} \\ a'_{2j+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_j) & -\sin(m\theta_j) \\ \sin(m\theta_j) & \cos(m\theta_j) \end{pmatrix} \begin{pmatrix} a_{2j} \\ a_{2j+1} \end{pmatrix}$$

The full transformation is a block-diagonal rotation matrix:

$$R_m = \begin{pmatrix} R(m\theta_0) & & & \\ & R(m\theta_1) & & \\ & & \ddots & \\ & & & R(m\theta_{d/2-1}) \end{pmatrix}$$

### The Final Dot Product

When we compute the dot product of two rotated vectors, each pair contributes independently:

$$q'_m \cdot k'_n = \sum_{j=0}^{d/2-1} \Big[ |q_j||k_j|\cos\big(\phi_j + (n-m)\theta_j\big) \Big]$$

where $\phi_j$ is the original angle between the $j$-th pair of $q$ and $k$.

This is a sum of cosines at different frequencies — analogous to a Fourier series. The combination of many frequencies creates a unique "fingerprint" for each relative distance, solving the aliasing problem completely.

## Summary

|Concept|Explanation|
|---|---|
|**Problem**|Vanilla attention ignores token positions|
|**Solution**|Rotate query/key vectors by their position before dot product|
|**Why it works**|Rotation changes the angle between vectors; the change depends only on relative position|
|**Aliasing problem**|A single rotation frequency causes distant positions to collide|
|**Multi-frequency fix**|Different dimension pairs rotate at different speeds, covering all distance scales|
|**Frequency formula**|$\theta_j = 10000^{-2j/d}$, giving fast rotation for early pairs, slow for later pairs|

The elegance of RoPE lies in encoding relative position through rotation — a geometric operation that naturally produces position-dependent similarity without adding any learnable parameters. The model learns to _use_ this position information through its existing $W_Q$ and $W_K$ projection matrices.
