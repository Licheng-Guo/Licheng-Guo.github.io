---
title: "Fourier Transformation is Simply Changing Basis"
date: 2026-01-17
tags: ["Intuitive-Math"]
math: true
---

*This post explains DFT and DCT from a linear algebra perspective. The note has been polished by AI based on the original learning notes of the author.*

When I tried to learn Fourier transformation in undergrad, I never really figured out why the formula is the way it is. I rigidly and brute-forcefully memorize the formulas to answer exam questions and then totally forgot everything two weeks after the semester. Almost 10 years after I originally took the course, I had an in-depth discussion with AI what exactly is Fourier Transformation and was fortunate to have an aha moment.

Fourier transformation is just a change of basis from a linear algebra perspective.

This blog will starts with Discrete Fourier Transformation (DFT), and then extends the idea to Discrete Cosine Transformation (DCT). It was a breathtaking moment for me to realize that, behind the mysterious transformation formula, how simple and intuitive they are.

## DFT as Change of Basis

Textbooks often describe DFT as "mapping signals from the time domain to the frequency domain". There is nothing fancy about a signal. Now, to describe any vector, you need a **basis** — a set of N linearly independent vectors that can combine to create any point in that space. A real signal $\mathbf{x}$ of N samples is just a vector in $\mathbb{R}^N$ with the standard basis $[1, 0, 0, ...]^T$, $[0, 1, 0, ...]^T$, etc. (We'll work in $\mathbb{C}^N$ since DFT uses complex exponentials as basis vectors.)

The DFT process simply says, now I have a different set of $N$ basis vectors as follows, what is the coordinate of the signal vector expressed with this new set of basis vectors? We will explain where these set of basis vectors come from, but for now let's just assume they are provided.

$$
\mathbf{b}_0 = [e^{i \cdot 0 \cdot \frac{2\pi \cdot 0}{N}}, e^{i \cdot 0 \cdot \frac{2\pi \cdot 1}{N}}, ..., e^{i \cdot 0 \cdot \frac{2\pi \cdot (N-1)}{N}}]^T = [1, 1, ..., 1]^T
$$
$$
\mathbf{b}_1 = [e^{i \cdot 1 \cdot \frac{2\pi \cdot 0}{N}}, e^{i \cdot 1 \cdot \frac{2\pi \cdot 1}{N}}, ..., e^{i \cdot 1 \cdot \frac{2\pi \cdot (N-1)}{N}}]^T
$$
$$
\mathbf{b}_2 = [e^{i \cdot 2 \cdot \frac{2\pi \cdot 0}{N}}, e^{i \cdot 2 \cdot \frac{2\pi \cdot 1}{N}}, ..., e^{i \cdot 2 \cdot \frac{2\pi \cdot (N-1)}{N}}]^T
$$
$$
\vdots
$$
$$
\mathbf{b}_{N-1} = [e^{i \cdot (N-1) \cdot \frac{2\pi \cdot 0}{N}}, e^{i \cdot (N-1) \cdot \frac{2\pi \cdot 1}{N}}, ..., e^{i \cdot (N-1) \cdot \frac{2\pi \cdot (N-1)}{N}}]^T
$$

In general: $\mathbf{b}_k[n] = e^{i \cdot 2\pi k n / N}$ for $k, n = 0, 1, ..., N-1$.

**Physical meaning:** The index k represents frequency — basis vector $\mathbf{b}_k$ completes exactly k full rotations around the unit circle over the N samples. So $\mathbf{b}_0$ is constant (DC), $\mathbf{b}_1$ oscillates once, $\mathbf{b}_2$ oscillates twice, etc. Projecting onto $\mathbf{b}_k$ measures "how much does my signal oscillate at frequency k?"

For example, for $N=4$, the basis vector $\mathbf{b}_1$ (k=1) is:
$$
\begin{aligned}
\mathbf{b}_1 &= [e^{i \cdot 2\pi \cdot 0/4}, e^{i \cdot 2\pi \cdot 1/4}, e^{i \cdot 2\pi \cdot 2/4}, e^{i \cdot 2\pi \cdot 3/4}]^T \\
&= [e^{0}, e^{i\pi/2}, e^{i\pi}, e^{i \cdot 3\pi/2}]^T \\
&= [1, i, -1, -i]^T
\end{aligned}
$$

Note that the DFT bases are **orthogonal** — each basis vector is perpendicular to all others. This is crucial so that projections are independent and reconstruction is trivial.

To compute the coefficients $X_k$ of signal vector $\mathbf{x}$ on the basis vector $\mathbf{b_k}$, we perform the inner product between them— a projection. You're asking: "How much does my signal point in the direction of this basis vector?" 

If your signal is perfectly aligned with basis vector k (say, a pure cosine at that frequency), the projection is large. If your signal is perpendicular to it, the projection is zero.

The collection of all these projections ${X₀, X₁, ..., X_{N-1}}$ is just your original vector **expressed in the new coordinate system**.

Note the conjugate operation since we are in complex domain.
$$
\begin{aligned}
X_k &= \langle \mathbf{x}, \mathbf{b_k} \rangle \\
&= \sum_{n=0}^{N-1} x[n] \cdot \overline{b_k[n]} \\
&= \sum_{n=0}^{N-1} x[n] \cdot \overline{e^{i \cdot k \cdot \frac{2\pi \cdot n}{N}}} \\
&= \sum_{n=0}^{N-1} x[n] \cdot {e^{-i \cdot k \cdot \frac{2\pi \cdot n}{N}}} \\
\end{aligned}
$$
Therefore, when we express the same vector $\mathbf{x}$ through this new magic set of basis vectors.

$$ \begin{bmatrix} \langle \mathbf{x}, \mathbf{b_0} \rangle \\ \langle \mathbf{x}, \mathbf{b_1} \rangle \\ ... \\ \langle \mathbf{x}, \mathbf{b_{N-1}} \rangle \end{bmatrix}$$

Does that look familiar? That's exactly the formula to compute DFT. 

Note that to get the true coordinate (coefficient) in the new basis, we need to divide $\langle \mathbf{x}, \mathbf{b_k} \rangle$ by $\|\mathbf{b}_k\|^2$. For this specific set of basis, $\|\mathbf{b}_k\|^2 = N$ for each basis vector. As is the convention for DFT, we skip this division in the forward transform and instead adjust by $1/N$ when we reconstruct the signal.

## Inverse DFT as Reconstruction

The inverse DFT is just a weighted sum of basis vectors. It is just the definition of what a basis means:

$$\text{signal} = \sum_{k=0}^{N-1} X_k \cdot (\text{basis vector } k)$$

Each coefficient $X_k$ tells you "how much" of basis vector k to include. Add them all up, you get your original vector back. This is exactly like saying:

$$\vec{v} = v_x \hat{x} + v_y \hat{y} + v_z \hat{z}$$

Same idea, just with sinusoids instead of $x̂$, $ŷ$, $ẑ$.


But here's the catch: our basis vectors aren't **unit length**. As we mentioned, to get the true coefficients we should have divided $X_k$ by $\|\mathbf{b}_k\|^2 = N$, but by convention DFT skips this normalization. So we need to apply the $1/N$ factor in the reconstruction process.

$$
\mathbf{x} = \frac{1}{N}\sum_{k=0}^{N-1} X_k \cdot \mathbf{b}_k
$$
Thus
$${x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X_k \cdot e^{i \cdot 2\pi k n / N}}$$

That's the inverse DFT. The **positive sign** in the exponent appears because we're summing the basis vectors themselves (not their conjugates). The **1/N** is just normalization by the basis vector norms.

## Summary of DFT and Inverse DFT

Fundamentally, DFT is really just changing basis. Pick a basis, project onto it, get coefficients. Use coefficients to reconstruct. All the formulas are just this linear algebra dressed up in trigonometric clothing. The DFT formula:

$$X_k = \sum_{n=0}^{N-1} x_n \cdot e^{-i 2\pi kn/N}$$

looks mysterious until you realize it's saying:

$$X_k = \langle \mathbf{x}, \mathbf{b}_k \rangle$$

"Project my signal onto the k-th sinusoidal basis vector."

That's it. Everything else — the negative sign, the 1/N, the complex exponentials — is just the mechanics of working in a complex vector space with a specific choice of orthogonal basis.


---

## DCT: Same Idea, Different Basis

Once we view DFT as changing basis, other transforms in the family become equally transparent.

DFT uses complex exponentials, which is elegant but means real signals produce complex coefficients. For applications like image compression, we'd prefer real coefficients. The DCT achieves this by using **cosines only** — a real-valued orthogonal basis for real signals. This is why JPEG uses DCT, not DFT.

| | DFT | DCT |
|---|---|---|
| **Basis** | $e^{i 2\pi k n / N}$ | $\cos(\pi k (n+0.5) / N)$ |
| **Coefficients** | Complex | Real |
| **Use case** | General signal analysis | Image/audio compression |

Both are just different choices of orthogonal basis for the same N-dimensional space.


## The Basis Vectors

First, we define our basis. The k-th DCT basis vector has components:

$$\mathbf{b}_k[n] = \cos\left(\frac{\pi k (n + 0.5)}{N}\right) \quad \text{for } n = 0, 1, ..., N-1$$

**Why (n + 0.5)?** This half-sample shift samples the cosine at the *center* of each interval rather than at the boundary. This makes the basis vectors orthogonal and avoids boundary discontinuities — important for compression where we process signals in blocks.

So for N=4, basis vector k=1 would be:

$$\mathbf{b}_1 = \begin{bmatrix} \cos(\pi \cdot 1 \cdot 0.5/4) \\ \cos(\pi \cdot 1 \cdot 1.5/4) \\ \cos(\pi \cdot 1 \cdot 2.5/4) \\ \cos(\pi \cdot 1 \cdot 3.5/4) \end{bmatrix}$$

We have N such vectors, one for each frequency k = 0, 1, ..., N-1.


## Forward DCT: Projection

The forward transform asks: "What are the coordinates of my signal in this basis?"

For orthogonal bases, each coordinate is just the **inner product** of your signal with that basis vector:

$$X_k = \langle \mathbf{x}, \mathbf{b}_k \rangle = \sum_{n=0}^{N-1} x_n \cdot \mathbf{b}_k[n]$$

Substituting our basis definition:

$$\boxed{X_k = \sum_{n=0}^{N-1} x_n \cos\left(\frac{\pi k (n + 0.5)}{N}\right)}$$

That's the DCT formula. It's just N inner products — one projection per basis vector.


## Inverse DCT: Reconstruction

The inverse asks: "Given coordinates, reconstruct the vector."

For any basis, reconstruction is:

$$\mathbf{x} = \sum_{k=0}^{N-1} c_k \cdot \mathbf{b}_k$$

But here's the catch: our basis vectors aren't **unit length**. The projection $X_k$ gives us a raw inner product, not the final coefficient $c_k$. We need to normalize.

For orthogonal (non-orthonormal) bases:

$$c_k = \frac{X_k}{|\mathbf{b}_k|^2}$$

The norms of our DCT basis vectors are:

- $|\mathbf{b}_0|^2 = N$ (the k=0 basis vector is all 1s)
- $|\mathbf{b}_k|^2 = N/2$ for k > 0

So the reconstruction coefficients are:

$$c_0 = \frac{X_0}{N}, \quad c_k = \frac{X_k}{N/2} = \frac{2X_k}{N} \text{ for } k > 0$$

Plugging into the reconstruction formula:

$$x_n = c_0 \cdot \mathbf{b}_0[n] + \sum_{k=1}^{N-1} c_k \cdot \mathbf{b}_k[n]$$

$$x_n = \frac{X_0}{N} \cdot 1 + \sum_{k=1}^{N-1} \frac{2X_k}{N} \cos\left(\frac{\pi k (n+0.5)}{N}\right)$$

$$\boxed{x_n = \frac{1}{N}\left[X_0 + 2\sum_{k=1}^{N-1} X_k \cos\left(\frac{\pi k (n + 0.5)}{N}\right)\right]}$$

That's the inverse DCT. The factor of 2 and the special treatment of $X_0$ come purely from normalizing by basis vector lengths.

## Visualizing DCT Reconstruction

The animation below shows how DCT reconstruction works in practice. We start with a signal (shown in green) and progressively add DCT basis vectors weighted by their coefficients. Watch how the blue reconstruction curve converges to the original signal as more terms are added:

<video autoplay loop muted playsinline width="100%">
  <source src="/images/dct_reconstruction.mp4" type="video/mp4">
</video>

The bottom panel shows the DCT coefficients — notice how the low-frequency terms (small k) typically have larger coefficients, which is why DCT is so effective for compression: you can often discard the high-frequency terms with minimal visual impact.