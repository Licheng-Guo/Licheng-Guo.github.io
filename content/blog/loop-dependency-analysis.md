---
title: "Intuitive Loop Dependency Analysis: Think in 1D"
date: 2026-01-24
tags: ["Compilers"]
math: true
---

Dependency analysis is one of the most important and fundamental topics in compiler optimization. It answers whether and how we can parallelize a loop to run programs faster—whether on CPU, GPU, TPU, or any other architecture. For first-time learners, it's also one of the most opaque and abstract topics. Many end up memorizing the rules—which is what I did—then gradually, if ever, come to understand them through practice.

This article aims to help readers gain an intuitive understanding of the basics through concrete examples. We'll first walk through four typical cases, unroll them, and examine the dependency patterns with our own eyes. One especially powerful technique I've found is to mentally view a multi-dimensional array as a 1D array, just as it's physically stored in memory. Then we'll introduce the concept of the distance vector and explain how to systematically determine whether a loop can be parallelized.

To avoid confusion, **I'll deliberately avoid using the word "parallel" throughout**, because it has many nuanced meanings across different contexts and architectures. Instead, we'll assume a loop nest executes strictly **sequentially** and ask whether a given loop can be **randomly shuffled** without affecting correctness.

A loop is *shuffleable* if we can replace `for i in 0..N` with `for i in shuffle(0..N)` and still get the same result. For nested loops, shuffleability applies independently at each level:

```python
# Original (both in order)
for i in 0..N:
    for j in 0..M:
        A[i][j] = ...

# Outer shuffleable, inner in order
for i in shuffle(0..N):      # i iterations can run in any order
    for j in 0..M:           # but within each i, j runs 0,1,2,...
        A[i][j] = ...

# Outer in order, inner shuffleable
for i in 0..N:               # i runs 0,1,2,...
    for j in shuffle(0..M):  # but within each i, j can run in any order
        A[i][j] = ...
```


The goal: determine which loops can be shuffled without changing the program's output.


---

## Case Studies

We'll analyze four canonical dependency patterns, building intuition before formalizing a rule. For each case, we show the loop code, unroll it to see the actual memory operations, and determine by inspection which loops can be shuffled.

**Setup:** A 4×4 grid where $i$ (rows) is the outer loop and $j$ (columns) is the inner loop. Boundary values—cells read but not written by the loop—are assumed pre-initialized.

### Case 1: Vertical Dependencies

```c
for (int i = 1; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
        A[i][j] = A[i-1][j] + 1;
    }
}
```

Each cell reads from the cell directly above (previous row, same column).

To reason about dependencies concretely, it helps to think of the 2D array as what it really is in memory—a 1D contiguous block. For a 4×4 array, `A[i][j]` maps to `A_flat[i * 4 + j]`. Unrolling and labeling each operation with its 1D index:

```c
// Row i=1
A[4]  = A[0] + 1;    // (1,0) ← (0,0)  |
A[5]  = A[1] + 1;    // (1,1) ← (0,1)  |--------
A[6]  = A[2] + 1;    // (1,2) ← (0,2)  |       |
A[7]  = A[3] + 1;    // (1,3) ← (0,3)  |       |
                     //                        |-- cannot swap these two blocks
// Row i=2           //                        | 
A[8]  = A[4] + 1;    // (2,0) ← (1,0)  |       |
A[9]  = A[5] + 1;    // (2,1) ← (1,1)  |--------
A[10] = A[6] + 1;    // (2,2) ← (1,2)  |
A[11] = A[7] + 1;    // (2,3) ← (1,3)  |
                     //
// Row i=3           //
A[12] = A[8] + 1;    // (3,0) ← (2,0)  |
A[13] = A[9] + 1;    // (3,1) ← (2,1)  |--- we can freely swap inside this block
A[14] = A[10] + 1;   // (3,2) ← (2,2)  |
A[15] = A[11] + 1;   // (3,3) ← (2,3)  |
```

- **Can we shuffle $j$ (within a row)?** Within row 1, indices `A[4]`..`A[7]` read from `A[0]`..`A[3]` respectively—distinct, pre-initialized locations with no conflicts. **Yes.**
- **Can we shuffle $i$ (across rows)?** Row 2 reads from `A[4]`..`A[7]`, which were written by row 1. If we execute row 2 before row 1, we read uninitialized data. **No.**


### Case 2: Horizontal Dependencies

```c
for (int i = 0; i < 4; i++) {
    for (int j = 1; j < 4; j++) {
        A[i][j] = A[i][j-1] + 1;
    }
}
```

Each cell reads from the cell directly to its left (same row, previous column).

```c
// Row i=0
A[1] = A[0] + 1;    // (0,1) ← (0,0) |
A[2] = A[1] + 1;    // (0,2) ← (0,1) |-----
A[3] = A[2] + 1;    // (0,3) ← (0,2) |    |
                    //                    |
// Row i=1          //                    |-- can swap these blocks                  
A[5] = A[4] + 1;    // (1,1) ← (1,0) |    |
A[6] = A[5] + 1;    // (1,2) ← (1,1) |-----
A[7] = A[6] + 1;    // (1,3) ← (1,2) |

// Row i=2
A[9]  = A[8] + 1;   // (2,1) ← (2,0) |
A[10] = A[9] + 1;   // (2,2) ← (2,1) |--- must preserve order within
A[11] = A[10] + 1;  // (2,3) ← (2,2) |

// Row i=3
A[13] = A[12] + 1;  // (3,1) ← (3,0)
A[14] = A[13] + 1;  // (3,2) ← (3,1)
A[15] = A[14] + 1;  // (3,3) ← (3,2)
```

- **Can we shuffle $j$?** Within row 0, there's a chain: `A[1]` reads `A[0]`, `A[2]` reads `A[1]`, `A[3]` reads `A[2]`. Each step depends on the previous. **No.**
- **Can we shuffle $i$?** Row 1's operations read from `A[4]`, `A[5]`, `A[6]`—all locations within row 1. (`A[4]` is column 0, a boundary value; `A[5]` and `A[6]` were written earlier in the same row.) Critically, row 1 never reads from row 0. Rows are independent islands. **Yes.**


### Case 3: Diagonal Dependencies

Now things get interesting. Each cell reads from the cell diagonally above-left (previous row, previous column). Having seen Cases 1 and 2, many people's first instinct is that neither loop can be shuffled—after all, both indices change. Is that your guess?

```c
for (int i = 1; i < 4; i++) {
    for (int j = 1; j < 4; j++) {
        A[i][j] = A[i-1][j-1] + 1;
    }
}
```

Let's unroll and see:

```c
// Row i=1
A[5]  = A[0] + 1;   // (1,1) ← (0,0)
A[6]  = A[1] + 1;   // (1,2) ← (0,1)
A[7]  = A[2] + 1;   // (1,3) ← (0,2)

// Row i=2
A[9]  = A[4] + 1;   // (2,1) ← (1,0)
A[10] = A[5] + 1;   // (2,2) ← (1,1)
A[11] = A[6] + 1;   // (2,3) ← (1,2)

// Row i=3
A[13] = A[8] + 1;   // (3,1) ← (2,0)
A[14] = A[9] + 1;   // (3,2) ← (2,1)
A[15] = A[10] + 1;  // (3,3) ← (2,2)
```

- **Can we shuffle $i$?** Row 2 reads `A[5]` and `A[6]`, which were written by row 1. **No.**
- **Can we shuffle $j$?** Within row 1, `A[5]` reads `A[0]`, `A[6]` reads `A[1]`, `A[7]` reads `A[2]`—all from row 0, which is pre-initialized. No dependencies within the row. **Yes!**

Wait, what just happened? Why can we shuffle the inner loop when it reads from $j-1$ and writes to $j$?

This is where the 1D mental model really pays off. The absolute indices don't lie: we read from 0, 1, 2 and write to 5, 6, 7. There are no dependencies among those operations whatsoever. So what happened to the $j-1 \to j$ pattern?

The key: $j-1$ refers to a location **in the row above**, not the current row. When row 2 begins, all of row 1 is already complete. The column offset doesn't matter—the entire previous row is available.


### Case 4: Anti-Diagonal Dependencies

If you followed Case 3, this one should be straightforward.

```c
for (int i = 1; i < 4; i++) {
    for (int j = 0; j < 3; j++) {
        A[i][j] = A[i-1][j+1] + 1;
    }
}
```

Each cell reads from the cell diagonally above-right (previous row, *next* column).

```c
// Row i=1
A[4] = A[1] + 1;    // (1,0) ← (0,1)
A[5] = A[2] + 1;    // (1,1) ← (0,2)
A[6] = A[3] + 1;    // (1,2) ← (0,3)

// Row i=2
A[8]  = A[5] + 1;   // (2,0) ← (1,1)
A[9]  = A[6] + 1;   // (2,1) ← (1,2)
A[10] = A[7] + 1;   // (2,2) ← (1,3)  ← A[7] is boundary

// Row i=3
A[12] = A[9] + 1;   // (3,0) ← (2,1)
A[13] = A[10] + 1;  // (3,1) ← (2,2)
A[14] = A[11] + 1;  // (3,2) ← (2,3)  ← A[11] is boundary
```

- **Can we shuffle $j$?** Within row 1, all sources (`A[1]`, `A[2]`, `A[3]`) are in row 0. **Yes.**
- **Can we shuffle $i$?** Row 2 reads `A[5]` and `A[6]`, written by row 1. **No.**

Whether it's $j-1$ or $j+1$, both Case 3 and Case 4 read from a location in the previous row, which is entirely complete by the time we need it. The direction of the column offset is irrelevant; what matters is the row.

### Summary So Far

| Case | Pattern | Loop $i$ (outer) | Loop $j$ (inner) |
|------|---------|----------------|----------------|
| 1 | Vertical (read from $i-1$) | In order | Shuffle ✓ |
| 2 | Horizontal (read from $j-1$) | Shuffle ✓ | In order |
| 3 | Diagonal (read from $i-1, j-1$) | In order | Shuffle ✓ |
| 4 | Anti-Diagonal (read from $i-1, j+1$) | In order | Shuffle ✓ |

A pattern is emerging: the constraint seems tied to which index changes when we look at the source. In Cases 1, 3, and 4, the source is in a previous row ($i-1$), so the outer loop must stay in order. In Case 2, the source is in the same row but a previous column, so only the inner loop must stay in order. Let's formalize this.


---

## The Distance Vector

So far we've been doing eyeball analysis on unrolled code. It works, but doesn't scale. We need a systematic method.

The key insight from our case studies: what matters is not the absolute indices, but *how far back* we're reading in each dimension. We capture this with a **distance vector** which points from the source iteration to the sink iteration. For a dependency where iteration $(i, j)$ reads from $(i', j')$:

$$\vec{d} = (\text{sink} - \text{source}) = (i - i',\; j - j')$$

Each component measures "how many iterations back" in that dimension. A positive component means we're reading from earlier iterations; zero means same iteration; negative means we're reading from a "later" iteration in that dimension.

**The First-Positive Rule:** Scan the distance vector left-to-right (outermost to innermost):
1. Before we encounter the first positive component, those loops can be shuffled.
2. The first positive component $d_k > 0$ identifies the loop that *carries* the dependency—that loop must stay in order.
3. All loops *inside* the carrying loop can be shuffled.

For example, consider a 3-level loop nest with $\vec{d} = (0, 2, -1)$. Scanning left-to-right: $d_1 = 0$ (loop 1 can shuffle), $d_2 = 2$ (positive—loop 2 carries the dependency, must stay in order), $d_3 = -1$ (inside the carrying loop, can shuffle). The -1 looks scary but it's harmless—it's dominated by the +2 in a more significant position.

Now let's verify this rule against our case studies and build intuition for *why* it works. We'll go through each case, compute its distance vector, apply the rule, and confirm it matches what we saw earlier.

### Case 1 Revisited

The dependency $A[i][j] = A[i-1][j] + 1$ has sink $(i, j)$ and source $(i-1, j)$, giving $\vec{d} = (1, 0)$.

Applying the rule: $d_i = 1$ (positive—loop $i$ carries the dependency). Stop scanning. Result: $i$ must stay in order, $j$ can shuffle. This matches our visual inspection.

But *why* does the inner loop become free once the outer runs in order? Look back at the unrolled code. Row 2 needs `A[4]`..`A[7]`, all written by row 1. If we run rows in order (i=1, then i=2, then i=3), by the time we start row 2, the entire row 1 is done—all four cells are ready. So within row 2, it doesn't matter which column we compute first. We can do `A[8]`, `A[9]`, `A[10]`, `A[11]` in any order because they all read from row 1, which is complete.

### Case 2 Revisited

The dependency $A[i][j] = A[i][j-1] + 1$ has sink $(i, j)$ and source $(i, j-1)$, giving $\vec{d} = (0, 1)$.

Applying the rule: $d_i = 0$ (no dependency across rows), $d_j = 1$ (positive—loop $j$ carries the dependency). Result: $i$ can shuffle, $j$ must stay in order. ✓

This is the flip side of Case 1. There, we asked "why can inner shuffle?" Here, we ask "why can outer shuffle?" The answer: $d_i = 0$ means no cross-row dependency. Each row is a self-contained chain that doesn't touch other rows. We saw this in the unrolled code—row 1 only reads from within row 1.

### Case 3 Revisited

The dependency $A[i][j] = A[i-1][j-1] + 1$ has sink $(i, j)$ and source $(i-1, j-1)$, giving $\vec{d} = (1, 1)$.

Applying the rule: $d_i = 1$ (positive—loop $i$ carries the dependency). Stop. Result: $i$ must stay in order, $j$ can shuffle because it's inside the dependency-carry loop.

This is the case that trips people up the most. Both components are positive! Shouldn't both loops be constrained?

**No.** The key insight is that the two 1s have different *weight*. Let's see why by going back to the 1D view.

In memory, `A[i][j]` lives at address $i \times M + j$ (where $M = 4$ is the row width). So the 1D distance between source and sink is:

$$\text{Distance}_{1D} = d_i \times M + d_j = 1 \times 4 + 1 = 5$$

We're reading from 5 cells back—safely in the past. But here's the crucial point: $d_i$ is multiplied by $M$ (= 4), while $d_j$ is multiplied by 1. The outer component dominates because it controls which *row* we're in, and rows are far apart in memory.

This means: as long as $d_i > 0$, the 1D distance is at least $+M$, no matter what $d_j$ is. Even if $d_j$ were $-3$, we'd still have $1 \times 4 + (-3) = 1 > 0$. The outer dimension's positive contribution overwhelms anything the inner dimension can do.

Here's an analogy. Think of $i$ as the day and $j$ as the hour. The dependency $A[i][j] = A[i-1][j-1]$ says: "To do Tuesday's 10AM task, I need Monday's 9AM task done first." Days must stay in order—Tuesday comes after Monday. But within Tuesday, once Monday is fully complete, we can shuffle Tuesday's hourly tasks however we like. The 9AM dependency was on *Monday's* 9AM, not Tuesday's.

So once we find the first positive component ($d_i = 1$), we know the dependency is satisfied—no need to examine what comes after.

### Case 4 Revisited

The dependency $A[i][j] = A[i-1][j+1] + 1$ has sink $(i, j)$ and source $(i-1, j+1)$, giving $\vec{d} = (1, -1)$.

Applying the rule: $d_i = 1$ (positive—loop $i$ carries the dependency). Stop. Result: $i$ must stay in order, $j$ can shuffle. ✓

Wait—there's a *negative* component! Doesn't $d_j = -1$ mean we're reading from column $j+1$, a "future" column? Isn't that illegal?

No. The key is that the -1 is in a *non-leading* position. Let's check the 1D distance:

$$\text{Distance}_{1D} = d_i \times M + d_j = 1 \times 4 + (-1) = 3$$

Still positive! We're reading from 3 cells back in memory—safely in the past.

Back to our day/hour analogy: $A[i][j] = A[i-1][j+1]$ says "Tuesday's 10AM task needs Monday's 11AM task done first." Wait—11AM is *later* than 10AM! But that's fine, because it's Monday's 11AM, and Monday is completely done before Tuesday starts. The "future hour" is on a past day, so it's already available.

**When would negative be fatal?** Only if it appears in the *leading* position. Suppose we had $\vec{d} = (-1, 1)$—reading from $A[i+1][j-1]$. That would mean "Monday's 10AM task needs Wednesday's 9AM task done first." We'd be waiting on work from the future. Impossible.

In 1D terms: $-1 \times 4 + 1 = -3$. Negative distance means reading from a memory location we haven't written yet.

### Putting It Together

Let's tabulate what we've verified:

| Case | $\vec{d}$ | First Positive | Outer ($i$)  | Inner ($j$)  |
| ---- | --------- | -------------- | ---------- | ---------- |
| 1    | $(1, 0)$  | $d_i$          | In order | Shuffle ✓  |
| 2    | $(0, 1)$  | $d_j$          | Shuffle ✓  | In order |
| 3    | $(1, 1)$  | $d_i$          | In order | Shuffle ✓  |
| 4    | $(1, -1)$ | $d_i$          | In order | Shuffle ✓  |

Every case matches what we discovered through the laborious unrolling exercise. But now we have a mechanical rule: compute the distance vector, find the first positive component, done. No need to unroll anything.


---

## Multiple Dependencies

So far each example had exactly one dependency. Real code is messier—a single statement might read from multiple locations, or a loop body might have multiple statements with different access patterns.

The rule is simple: **analyze each dependency separately, then intersect the results.** A loop can only be shuffled if it's shuffleable for *every single* dependency. One bad apple spoils the bunch.

### Example: Total Lockdown

Here's a pattern that appears constantly in dynamic programming—edit distance, longest common subsequence, path counting, you name it:

```c
for (int i = 1; i < N; i++) {
    for (int j = 1; j < M; j++) {
        A[i][j] = A[i-1][j] + A[i][j-1];
    }
}
```

Each cell depends on the cell above *and* the cell to the left. Two dependencies, two distance vectors:

| Source | $\vec{d}$ | Carries | Outer ($i$) | Inner ($j$) |
|--------|-----------|---------|-----------|-----------|
| `A[i-1][j]` | $(1, 0)$ | $d_i$ | In order | Shuffle ✓ |
| `A[i][j-1]` | $(0, 1)$ | $d_j$ | Shuffle ✓ | In order |

Now intersect. For loop $i$ to be shuffleable, it must be shuffleable for *both* dependencies. The first dependency says no. For loop $j$ to be shuffleable, the second dependency says no.

| Loop | From $(1,0)$ | From $(0,1)$ | Final |
|------|--------------|--------------|-------|
| $i$ | In order | Shuffle ✓ | **In order** |
| $j$ | Shuffle ✓ | In order | **In order** |

Neither loop can be shuffled. This is a "total lockdown"—the iteration order is completely fixed. If you've ever wondered why DP algorithms seem inherently sequential, this is why. (There are ways around this—wavefront techniques can expose diagonal shuffleability—but that's a topic for another day.)


---

## Wrapping Up

Let's step back and see what we've learned.

The core technique is to **think in 1D**. A multi-dimensional array is just a contiguous block of memory with fancy indexing. When you flatten it mentally, dependencies become arithmetic: is the source index less than the sink index? If yes, we're reading from the past (safe). If no, we're reading from the future (impossible).

The **distance vector** $\vec{d} = \vec{I}_{sink} - \vec{I}_{source}$ encodes this arithmetic compactly. The **first-positive rule** tells us which loop carries the dependency: scan left-to-right, find the first positive component, that loop must stay in order, everything inside it can shuffle.

Why does this work? Because outer loop indices have heavier weight when flattening. In a 2D array with $M$ columns, the 1D distance is $d_i \times M + d_j$. A positive $d_i$ contributes at least $+M$ to the distance, which overwhelms any negative $d_j$. This is why $(1, -1)$ is fine but $(-1, 1)$ is fatal.

When there are **multiple dependencies**, analyze each separately and intersect. A loop can only shuffle if *every* dependency allows it.


---


This covers the fundamentals. There's much more to explore—direction vectors (when distances aren't constant), loop transformations like interchange and skewing, the polyhedral model for more complex analyses—but the core intuition stays the same: think in 1D, find the first positive, and you're most of the way there.

**Further Reading:**
- Allen & Kennedy, *Optimizing Compilers for Modern Architectures*
- Wolfe, *High Performance Compilers for Parallel Computing*
