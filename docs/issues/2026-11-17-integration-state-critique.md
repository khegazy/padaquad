# Critique of "Integration state and a batch size that grows: design"

Adversarial review of `docs/issues/2026-11-17-integration-state-design.md` (the
"design"), written 2026-11-17 and 2026-11-18 on branch `fix_warm_start` (commit
5af0d76). The design changed three times while I worked; this review is of the
last version: `integrate` never runs the memory benchmark, it calls `f` on 1, 10,
100 and 1,000 copies of one point only to see whether `f` runs, a test call that
runs out of memory is retried at 0.75 times its size, and the benchmark stays
only inside `solver.get_max_batch(f=...)`.

Every claim below was checked against the code, a measurement I made, or a page I
opened. My scripts and raw outputs are in
`/pscratch/sd/k/khegazy/padaquad_membench/state_critique/` (scratch is purged;
copy it if it should be kept). GPU measurements ran on one node I allocated
myself (job 61518861, node nid008473, four NVIDIA A100 80 GB, torch 2.8.0); the
node was released after 7 minutes.

Terms used below.

- A **batch** is the set of points given to the integrand `f` in one call; the
  **batch size** is the largest number of points allowed in one call.
- A **pass** is one turn of the main loop of `integrate`.
- A **cold call** starts from a fresh starting mesh; a **warm call** starts from
  the mesh an earlier call returned.
- An **unlimited run** is a call whose batch size is larger than every pass, so
  every pass is one batch. Today's benchmark gives this on a large GPU.
- **F** is the peak GPU memory during one call of `f`, minus the memory in use
  before the call (the design's "forward memory").
- **R** is the memory still held after that call (outputs and the stored
  intermediate values that backpropagation needs).
- **X** is the extra memory that `loss.backward()` needs on top of R: the peak
  during the backward pass minus the memory held right before it.
- **U** is the design's "usable memory": free memory minus the share that
  `total_mem_usage` keeps free.
- **The backward check** is the rule of the design's section 6.3: run the
  backward pass only if U >= F.
- A call **fails** when it raises an out-of-memory error.

## 1. Summary

**Verdict: build after fixes.** The state and growth by trying survive. Five
things must change or be decided first.

1. **`conserve_memory=True` silently returns wrong integrals when the batch size
   changes inside a pass (F1).** Today's code already does: after one
   out-of-memory retry, 7 of 8 runs were off by 0.05 to 0.29 on an integral of
   0.54, with `converged=True`. The design makes such changes routine.
2. **The backward check holds for ordinary networks and fails for activation
   checkpointing (F2).** Ordinary models: 0 failures in the 31 batches it
   allowed. One deep checkpointed segment: 7 failures in 7 allowed batches.
3. **"Within the tolerance" is wrong (F3).** With `bosh3` and `dopri5`, runs with
   a limited batch were 4 to 166 tolerances off where the unlimited run was
   inside it. With `gk21` only the `converged` flag changed.
4. **The first out-of-memory error costs 4.9 to 6.3 s, not "under a second"
   (F4).**
5. **Two promises are dropped unsaid (F5, F6):** `total_mem_usage` keeps no
   memory free outside the backward check, and a batch size that shrank never
   grows back.

## 2. Findings, most severe first

### F1. Wrong: `conserve_memory=True` does not cope with a changing batch size, and today's code already returns wrong integrals there

- **Whose decision:** the author's ("`conserve_memory=True` grows like the other
  paths", section 11).
- **Claim** (section 5.3): "Its batch size already changes between passes today
  (it is recomputed from live memory) and within a pass (the out-of-memory
  retry), so its residual handling already copes with a changing size."
- **Verdict:** wrong.
- **Evidence.**
  - Code. `_evaluate_f_on_split_residual_nodes` lays a pass out with arithmetic
    on the batch size (`base.py:2262`, `2272`, `2278`, `2307-2310`) and then
    hands the nodes to `_evaluate_f_in_batches`, which may change the batch size
    during the pass (`base.py:2061-2070`). The points evaluated beyond the
    planned number become the "residual" carried to the next pass
    (`base.py:2319-2339`). When a changed batch size makes the residual a whole
    subinterval, `nodes_flat[-self.C : -self.C + num_residual_nodes]`
    (`base.py:2326`) is `nodes_flat[-C:0]`, an empty slice, while the values of
    those C points are kept. Nodes and values are then paired wrongly.
  - Measurement (`exp_conserve.py`, login node, CPU, explicit `max_batch`, the
    unmodified library). Integrand sin(40 t) + t on [0, 1], exact integral
    0.5416734515, atol = rtol = 1e-9, `take_gradient=False`,
    `conserve_memory=True`. One fake out-of-memory error is raised in the first
    batch, so today's retry shrinks the batch by 0.75 inside the pass.

    | Method | `max_batch` | Integral without the error | Integral after one retry | Error after one retry | `converged` |
    |---|---|---|---|---|---|
    | gk21 | 60 | 0.5416734515 | 0.6297257813 | 8.8e-02 | True |
    | gk21 | 100 | 0.5416734515 | 0.3804359855 | 1.6e-01 | True |
    | gk21 | 150 | 0.5416734515 | 0.2544132472 | 2.9e-01 | True |
    | gk21 | 200 | 0.5416734515 | 0.4654493653 | 7.6e-02 | True |
    | gk15 | 60 | 0.5416734515 | 0.4941298439 | 4.8e-02 | True |
    | gk15 | 100 | 0.5416734515 | 0.4412188978 | 1.0e-01 | True |
    | gk15 | 150 | 0.5416734515 | 0.3706556816 | 1.7e-01 | True |
    | gk15 | 200 | 0.5416734515 | 0.5416734515 | 1.1e-16 | True |

    Seven of eight are wrong. The reported error estimate stayed between 1.8e-13
    and 1.9e-10 and `converged` is True, so nothing in the result shows the
    problem (my script silenced padaquad's log; in normal use the only message
    would be today's "Caught OOM" warning about the retry). The same test with
    `conserve_memory=False` is correct.
  - The design's growth, emulated: the pass is laid out for the batch size B and
    the first call of `f` takes 10 B points. Same eight cases: seven wrong, with
    errors of 0.064 to 0.29 and `converged=True`.
  - Why the test suite did not see it:
    `tests/test_oom_recovery.py::test_integrate_recovers_from_oom` covers
    `conserve_memory=True` but integrates f = 1, and a constant integrand gives
    the right answer however nodes and values are paired.
- **Consequence.** Today, a user of `conserve_memory=True` whose GPU runs out of
  memory once gets a wrong number that looks converged. Under the design, the
  batch size changes inside a pass at every growth step, so the wrong number
  would appear on ordinary runs.
- **What to change.**
  1. Fix today's bug first, as its own change, with a regression test that uses
     a non-constant integrand.
  2. In the design, choose the batch size for `conserve_memory=True` once per
     pass, before the layout is computed, never inside
     `_evaluate_f_in_batches`. That contradicts "each batch, not each pass"
     (section 5.3) for this path; say so.
  3. Section 7.4 says of `take_gradient=False` that "the batch size only cuts a
     pass into calls of `f`, so results do not depend on it". That is true only
     with `conserve_memory=False`. With `conserve_memory=True` the batch size
     decides which subintervals a pass holds
     (`tests/test_error_integral_reference.py:29-34` relies on this).
- **Not determined.** A first, larger version of this check (four methods, also
  `max_batch=30`) did not finish in 170 s on the login node and I killed it. I do
  not know whether that was the slow low-order methods or a loop that does not
  end after a retry.

### F2. Partly confirmed, partly wrong: the check before the backward pass (owner's rule)

- **Whose decision:** the owner's (section 6.3: "the backward pass costs roughly
  as much memory as the forward pass"); running it on a size the user fixed is
  the author's.
- **Claim** (section 6.3): "If U >= F: go on to the backward pass." Section 10
  calls the assumption "backward memory about equals forward memory".
- **Verdict:** confirmed for ordinary networks, with a margin of two or more;
  wrong for activation checkpointing; incomplete in three smaller ways.
- **Evidence 1: X against F** (`exp_gpu_mem.py`, `exp_extra.py`, A100 80 GB,
  float64). Each row emulates one `take_gradient=True` batch: one call of `f` on
  N sorted points, the quadrature sum with 23 nodes per subinterval, the default
  loss, `loss.backward()`. Parameters start with `.grad = None`. Memory in MiB.
  Three batch sizes are shown for each model (253, 9,982 and the
  largest that ran); `gpu_mem.out` and `extra.out` hold all 52 rows.
  The check is unsafe where X/F is above 1. (R + X)/F is the number to compare
  with today's factor 2.1 (`_GRADIENT_MEMORY_SAFETY_FACTOR`, `memory.py:48`).

  | Model | N | F | R | X | X/F | (R + X)/F |
  |---|---|---|---|---|---|---|
  | 3 layers of 64, tanh | 253 | 1.37 | 0.37 | 0.25 | 0.18 | 0.45 |
  | | 9,982 | 20.00 | 15.21 | 9.75 | 0.49 | 1.25 |
  | | 999,994 | 1,953.11 | 1,473.13 | 976.56 | 0.50 | 1.25 |
  | 3 layers of 512, tanh | 253 | 3.96 | 2.97 | 4.01 | 1.01 | 1.76 |
  | | 9,982 | 155.97 | 117.06 | 119.00 | 0.76 | 1.51 |
  | | 999,994 | 15,624.91 | 11,726.97 | 7,812.46 | 0.50 | 1.25 |
  | 3 layers of 512, ReLU | 253 | 3.96 | 2.97 | 4.75 | 1.20 | 1.95 |
  | | 9,982 | 155.97 | 117.06 | 119.74 | 0.77 | 1.52 |
  | | 999,994 | 15,624.91 | 11,726.97 | 7,812.46 | 0.50 | 1.25 |
  | 6 layers of 2048, tanh | 253 | 27.67 | 23.72 | 148.24 | 5.36 | 6.21 |
  | | 9,982 | 1,092.00 | 936.08 | 404.03 | 0.37 | 1.23 |
  | | 99,981 | 10,935.42 | 9,374.05 | 3,124.42 | 0.29 | 1.14 |
  | 3 layers of 512, SiLU | 253 | 5.94 | 5.93 | 2.04 | 0.34 | 1.34 |
  | | 9,982 | 234.03 | 234.04 | 80.01 | 0.34 | 1.34 |
  | | 999,994 | 23,444.99 | 23,445.65 | 3,913.86 | 0.17 | 1.17 |
  | Force: gradient of an energy network inside `f` (3 layers of 512, SiLU) | 253 | 24.71 | 22.73 | 1.98 | 0.08 | 1.00 |
  | | 9,982 | 974.81 | 896.90 | 77.99 | 0.08 | 1.00 |
  | | 99,981 | 9,763.77 | 8,983.50 | 781.11 | 0.08 | 1.00 |
  | 6 checkpointed blocks of 2 layers of 512 | 253 | 8.91 | 6.92 | 20.10 | 2.26 | 3.03 |
  | | 9,982 | 312.94 | 273.03 | 157.99 | 0.50 | 1.38 |
  | | 999,994 | 31,250.81 | 27,351.88 | 11,718.68 | 0.37 | 1.25 |
  | The same 6 blocks, no checkpointing | 253 | 13.85 | 12.85 | 14.17 | 1.02 | 1.95 |
  | | 9,982 | 545.89 | 506.98 | 119.00 | 0.22 | 1.15 |
  | | 999,994 | 54,687.17 | 50,789.24 | 7,812.46 | 0.14 | 1.07 |
  | Pairwise term inside `f` (memory grows like N²) | 253 | 2.44 | 1.96 | 0.98 | 0.40 | 1.20 |
  | | 9,982 | 3,800.98 | 3,040.87 | 1,520.39 | 0.40 | 1.20 |
  | | 29,992 | 34,313.97 | 27,451.42 | 13,725.59 | 0.40 | 1.20 |
  | **12 layers of 512 inside ONE checkpoint segment** | 253 | 3.96 | 1.98 | 25.04 | 6.32 | 6.81 |
  | | 9,982 | 117.98 | 78.07 | 547.92 | 4.64 | 5.31 |
  | | 999,994 | 11,719.68 | 7,820.99 | 50,780.95 | 4.33 | 5.00 |
  | The same 12 layers, no checkpointing | 253 | 13.85 | 12.85 | 14.17 | 1.02 | 1.95 |
  | | 9,982 | 545.89 | 506.98 | 119.00 | 0.22 | 1.15 |
  | | 999,994 | 54,687.17 | 50,789.48 | 7,812.46 | 0.14 | 1.07 |

  What the table says:
  - **Per point, the backward pass needs less than the forward pass.** For the
    ten models other than the one-segment network, X/F was 0.08 to 0.50 for N
    above 20,000 (17 rows) and 0.08 to 0.77 for N between 1,000 and 20,000 (10
    rows). The rule "X is about F" is cautious there by a factor 1.3 to 12.
  - **A fixed cost breaks the rule at small N.** At N = 253, X/F was 1.0 to 5.4
    in six of these ten models. The cause is the gradients of the
    parameters, which do not depend on N: the 6-layer network of width 2048 has
    about 160 MiB of parameters, and its X at N = 253 was 148 MiB against an F
    of 28 MiB. (That the fixed part is the parameter gradients is my reading of
    the numbers; I did not measure it separately.)
  - **One checkpoint segment around a deep network breaks the rule at every
    N.** Activation checkpointing runs a part of the network without storing its
    intermediate values and computes them again during the backward pass: "the
    unsaved tensors are recomputed by re-invoking function in the backward pass"
    (PyTorch 2.15 documentation, `torch.utils.checkpoint`,
    https://docs.pytorch.org/docs/2.15/checkpoint.html). The forward pass is
    then small and the backward pass large: X/F = 4.1 to 6.3 over the five batch
    sizes. Today's factor 2.1 does not cover this either ((R + X)/F = 4.7 to
    6.8).
- **Evidence 2: the check itself** (`exp_gpu_check.py`, A100 80 GB). For each
  model the batch size was raised in steps of 1.12 from about a quarter of the
  GPU's memory until the forward call failed. At each size I computed F and U as
  the design does (U with `total_mem_usage` 0.9, the default, and 1.0), and then
  ran `loss.backward()` whatever the check said.

  | Model | Trials | Check passes, backward fits | Check passes, backward fails | Check says shrink, backward would have fit | Check says shrink, backward fails | Largest N the check allows | Largest N whose backward fit |
  |---|---|---|---|---|---|---|---|
  | 3 layers of 512, tanh | 13 | 7 | 0 | 4 | 2 | 2,562,862 | 4,032,711 |
  | 6 checkpointed blocks | 13 | 6 | 0 | 5 | 2 | 1,140,476 | 2,009,906 |
  | 6 layers of 2048, tanh | 13 | 6 | 0 | 5 | 2 | 326,827 | 575,979 |
  | Force (gradient inside `f`) | 13 | 6 | 0 | 7 | 0 | 366,120 | 809,371 |
  | 3 layers of 512, SiLU | 13 | 6 | 0 | 5 | 2 | 1,525,016 | 2,687,596 |
  | 12 layers in one checkpoint segment | 13 | 0 | 7 | 0 | 6 | 3,388,184 | none of the 13 |

  Counts are for `total_mem_usage` = 0.9. With 1.0 the five ordinary models gave
  36 passes, all of which fit, and the one-segment model gave 8 passes, all of
  which failed.
  - For the five ordinary models the check never let a failing backward pass
    through (0 of 31), and it stopped all 8 that would have failed. It also
    shrank 26 batches whose backward pass would have fit: the largest batch it
    allows is 1.6 to 2.2 times smaller than the largest that works.
  - For the one-segment model every batch the check allowed failed. The smallest
    batch I tried there (F = 19.7 GiB) already needed more than the GPU holds,
    so this run shows that the check does not stop such batches, not where the
    true limit lies. From the table above (R + X = 5.0 F), the backward pass
    fits up to F of about 16 GiB, and the check allows F up to 39 GiB.
- **Evidence 3: where the check sits in the code.**
  - The backward pass runs only if at least one subinterval of the batch was
    accepted (`base.py:1126`) and the loss requires a gradient (`base.py:1156`).
    `take_gradient=True` is the default, so every plain integral without
    learnable parameters takes this path. The design runs the check "on every
    gradient batch on a GPU". As written, a user who integrates sin(t) on a GPU
    has batches thrown away and evaluated again whenever U < F, although no
    backward pass will ever run.
  - "Drop this batch's outputs and evaluate again" is safe only before
    `_adaptively_increase_mesh` (`base.py:1080`), which rewrites the mesh. The
    simplest safe place is inside `_evaluate_f_on_full_nodes`, right after the
    call of `f` (`base.py:1971`). The design does not say.
  - The shrink rule contradicts itself: the batch "must shrink by at least the
    factor 0.75", and "a proportional estimate such as N (U + F) / (2 F) is
    acceptable". For U just below F the estimate is just below N, not below
    0.75 N.
  - Reading F means resetting the GPU's peak-memory counter before every
    gradient batch, also when the user gave `max_batch`. A user who reads
    `torch.cuda.max_memory_allocated()` to profile a training step will get the
    peak since padaquad's last batch, not since their own reset. Today only the
    benchmark resets the counter, and never when `max_batch` is given.
- **Consequence.**
  - Ordinary networks: the owner's position is supported. I found no case where
    a default training call would raise the new error with the check in place.
    The price is batches about half the size the GPU could hold, which is close
    to today's caution (today's rule allowed N of about 1.7 million for the
    3-layer tanh network on this GPU, the check 2.6 million, the hardware 4.0
    million; the first number is my calculation from today's constants, not a
    run).
  - Checkpointed models: the check passes and the backward pass fails, so the
    call raises `BackwardOutOfMemoryError`, or, with `parameters=`, restarts the
    whole integral once per shrink by 0.75 (about 4 restarts from F = 39 GiB to
    16 GiB; my arithmetic). Today such a model also fails at large batches.
  - A nearly full GPU: with `total_mem_usage` = 1.0, or on a small GPU, the fixed
    cost can exceed U while U >= F. With the default 0.9 on an 80 GB GPU the 8 GB
    kept free hides this unless the parameter gradients exceed about 8 GB.
- **What margin the check needs.** No constant fits all models: 1.0 is enough,
  with room to spare, for the per-point part of every ordinary model; the fixed
  part needs the size of the parameter gradients added; checkpointing needs a
  factor that grows with the depth of the segment (4.1 to 6.3 here).
- **What to change or decide.**
  1. Skip the check when the output of `f` does not require a gradient.
  2. Owner to decide: keep the constant rule U >= F and document checkpointing
     and the fixed cost as limitations, or measure instead of assuming. Measuring
     is one more counter read: after each backward pass read the peak again,
     which gives X of that batch, and require U >= max(F, X_prev N / N_prev) for
     the next one. The first batch of a cold call is small (253 points with the
     default settings), so the ratio is learned where a failure is least likely.
     This is my proposal; it is not tested.
  3. State the shrink rule once, for example min(0.75 N, N (U + R) / (R + F)).
     (U already counts the forward pass's freed temporary memory, so what
     dropping the batch frees is R, not F. My derivation, not tested.)
  4. Say where the check runs in the loop.
- **Not measured.** float32, attention or convolution layers, a custom
  `loss_fxn`, custom autograd functions, `torch.func` Jacobians, a second GPU
  model, and `checkpoint(..., use_reentrant=True)`.

### F3. Wrong: "slightly different results (within the tolerance)"

- **Whose decision:** the author's (section 7.4; growth per call of `f`; the
  starting size 1,000).
- **Claim** (section 7.4): with `take_gradient=True`, "the first calls of a state
  can give slightly different results (within the tolerance) from a run with an
  unlimited batch [...]. Once the batch size covers the passes, results match
  that run."
- **Verdict:** wrong as a general statement; one part confirmed.
- **Why the batch size matters here.** With `take_gradient=True` one batch is one
  turn of the loop. Each subinterval is accepted or split by comparing its error
  with max(atol, rtol |running integral|) (`base.py:992-1007`), and the running
  integral covers only the batches evaluated so far. The step that later
  re-checks accepted subintervals is off with `take_gradient=True`
  (`base.py:1032-1034`).
- **Evidence** (`exp_growth.py`, CPU on the allocated node). The script replaces
  the batch size that the loop passes to `_evaluate_f_on_mesh` by the design's
  rule (10 times the batch size when more points are waiting and the state is
  not converged; afterwards the larger of the old size and what ran). On this
  path one call of that method is one call of `f`, so the emulation is exact.
  Integrands are sums of narrow Lorentzian peaks times a learnable factor, with
  exact integrals; atol = 1e-12, rtol = 1e-6. "A" has three positive peaks on the
  left and three negative ones on the right, so the running integral is about
  1,000 times the final one halfway through. "C" has 40 positive peaks.
  1. **Cold calls: confirmed, no difference.** In all nine cold comparisons
     (three integrands, methods `gk21`, `bosh3`, `dopri5`) growth from 1,000 gave
     the same batches and an integral that differs by exactly 0 from the
     unlimited run. The reason (my reasoning, consistent with the runs): a pass
     can at most double the number of waiting points, the batch size is at least
     the previous pass, and the growth factor is 10, so a cold call whose first
     pass is at most 10,000 points never cuts a pass. The default first pass has
     at most 11 x 65 = 715 points.
  2. **A starting mesh larger than 10 times the batch size: results change.**
     Cold call from a mesh the user passes (evenly spaced), integrand A:

     | Method (points in the first pass) | Run | Batches (first three) | Error ÷ tolerance | `converged` |
     |---|---|---|---|---|
     | gk21 (46,000) | unlimited | 46,000; 552; 552 | 0.000005 | True |
     | | growth from 1,000 | 9,982; 36,018; 368 | 0.004 | False |
     | | growth from 253 | 2,530; 25,300; 18,170 | 0.006 | False |
     | bosh3 (120,000) | unlimited | 120,000; 7,488; 8,928 | 0.0000001 | True |
     | | growth from 1,000 | 10,000; 100,000; 16,816 | 0.0000001 | False |
     | | growth from 253 | 2,528; 25,280; 92,672 | 4.0 | False |
     | dopri5 (140,000) | unlimited | 140,000; 1,176; 756 | 0.000003 | True |
     | | growth from 1,000 | 9,996; 99,960; 30,044 | 165.6 | False |
     | | growth from 253 | 2,527; 25,270; 112,203 | 82.8 | False |

  3. **A state whose batch size stopped growing below the pass size** (what an
     out-of-memory error or the backward check leaves behind), second call from
     the first call's mesh with the peaks shifted by 0.3 of their width:

     | Case | Unlimited: error ÷ tolerance, `converged` | Batch size frozen at the first call's value: error ÷ tolerance, `converged` |
     |---|---|---|
     | A, gk21 (frozen at 1,000) | 0.000007, True | 0.000007, True |
     | A, bosh3 (frozen at 12,664) | 0.2, True | 79.7, False |
     | A, dopri5 (frozen at 1,000) | 2.3, False | 7,718, False |
     | C, gk21 (frozen at 2,162) | 0.0005, True | 0.0004, True |
     | C, bosh3 (frozen at 3,112) | 49,497, True | 4.2, True |

     The last row shows the dependence in the other direction: there the
     unlimited run is wrong by 6.22 on an integral of 125.67 (about the area of
     two peaks, each of area pi)
     and the limited run finds them, because its smaller running integral makes
     the tolerance tighter. Both report `converged=True`.
  4. The same state with growth allowed gave results identical to the unlimited
     run in every case where the mesh was at most 10 times the batch size.
- **Consequence.**
  - With `gk21` (the default) the true error stayed far inside the tolerance in
    all my cases; what changes is the `converged` flag and its warning, and the
    number of evaluations.
  - With the low-order methods the answer itself changes by many tolerances,
    though `converged=False` then says so in most rows.
  - The results of every later call depend on the history of the state: on the
    constant 1,000 when a large mesh is passed in, and on whether an
    out-of-memory error or the backward check (which is not an error) ever froze
    the batch size. The anti-goal of section 2 allows this by its wording
    ("beyond what a different batch size already changes today"); the owner
    should know that the allowed part is this large. Today the same happens
    whenever `max_batch` is small with `take_gradient=True`; what is new is that
    the batch size now changes by itself.
- **What to change or decide.**
  1. Replace the sentence in 7.4 by what was measured.
  2. Decide whether a state may start a warm call with a batch smaller than the
     mesh it carries. A rule "the first batch of a call is the whole first pass,
     unless the state is converged" would make results independent of the
     constant; its cost is more retries when that pass does not fit (F4).
- **Side observation, outside this review.** In case C with `bosh3`, the
  unlimited warm call returned an integral that is 5 % off with
  `converged=True`. That is today's code.

### F4. Wrong: the cost of the first out-of-memory error

- **Whose decision:** the estimate is the author's; growth until failure and the
  factor 0.75 are the owner's.
- **Claim** (section 5.4): each retry "costs a failed call of `f` and today's
  clean-up [...], roughly 0.1 s, so under a second in the worst case (my
  estimate, not measured)". Section 2: "a few milliseconds, plus, once per
  state, the retries".
- **Verdict:** wrong by a factor of 5 or more; the count of retries is confirmed.
- **Evidence** (`exp_gpu_oom.py`, A100 80 GB). The design's rule was run
  literally: batches of 1,000 times 10 until a call of `f` fails, then times 0.75
  with padaquad's clean-up (`_release_memory`, `memory.py:504-521`: garbage
  collection, emptying PyTorch's cache of GPU memory, a 0.05 s sleep) until a
  call fits.

  | Model | Failed calls | Time of each failed call (s) | Time lost in total (s) | Of which failed calls / garbage collection / emptying the cache / sleep (s) | First call that fit afterwards |
  |---|---|---|---|---|---|
  | 3 layers of 512, tanh, graph kept | 3 | 0.577, 2.110, 1.501 | 5.206 | 4.187 / 0.099 / 0.769 / 0.150 | 4,218,750 points, 1.897 s |
  | The same under `torch.no_grad()` | 8 | 1.094, 1.018, 0.011, 0.005, 0.005, 0.005, 0.012, 1.910 | 4.897 | 4.062 / 0.195 / 0.240 / 0.401 | 10,011,292 points, 2.239 s |
  | 6 layers of 2048, tanh, graph kept | 2 | 2.647, 2.067 | 6.344 | 4.714 / 0.071 / 1.459 / 0.100 | 562,500 points, 2.085 s |

  - A call that is far too large can fail in 5 ms (one allocation is refused at
    once). A call that is a little too large runs almost to its end first: 1.5
    to 2.6 s here, about the time of a call that fits.
  - Emptying the cache took 0.000 to 0.950 s after a real failure. Alone, on 12
    to 60 GiB of cached memory, it took 0.005 to 0.033 s for blocks of 1 GiB or
    more and 0.575 s for 31 GiB in blocks of 64 MiB.
  - The bound of "about 8 retries" is tight: the second row has 8 failed calls,
    and the call that finally fit was 1.001 times the size that had run before.
  - In a cold call the batch size follows the passes, which at most double
    (F3), so the failed batch is at most twice one that ran and I expect at most
    3 failed calls there (my reasoning, log 2 / log (4/3) = 2.4; not measured).
    The bound of 8 belongs to warm calls and large starting meshes.
- **The bound does not hold across settings.** It assumes the failed batch is at
  most 10 times one that ran under the same conditions. A state carries its
  batch size to calls where that is false: a size learned under
  `torch.no_grad()` or with `take_gradient=False` used for training (in the
  table above the same network settled at 10.0 million points without a graph
  and at 4.2 million with one), float32 to float64, a larger GPU to a smaller one, another
  integrand, or a `max_batch` the user typed. The number of retries is then
  log(ratio) / log(4/3) with no upper limit; a `max_batch` 100 times too large
  costs 16 failed calls.
- **Consequence.** Reaching the memory limit costs about 5 to 6 s once, where
  today's benchmark costs about 0.1 s per call on this GPU (brainstorm section
  2.2). For a training run of thousands of calls that is a good trade. For a
  single large integral on a GPU it is slower than today. In the first row the call
  that fit after the failures took 1.9 s for 4.2 million points, 4.9 times
  longer per point than the call on 1 million points before them; I did not
  establish why (the clean-up empties the cache, so that call must get all its
  memory from the driver again, which is one candidate).
- **What to change or decide.**
  1. Correct section 5.4 and the limitations text of section 10 with measured
     numbers.
  2. Owner to decide: keep 0.75, or shrink faster after a failure (for example
     halve, then stop). Halving bounds a factor-10 overshoot at 4 failed calls
     and leaves the batch between 50 % and 100 % of what fits, against 75 % to
     100 %. Not tested.

### F5. Incomplete: the promise of `total_mem_usage` is dropped without being said

- **Whose decision:** growth until failure is the owner's; the silence is the
  author's.
- **Claim** (Terms): usable memory is "`free - (1 - total_mem_usage) * total`
  (`_get_usable_memory`, unchanged)". Section 7: `total_mem_usage` is "used in
  this call".
- **Verdict:** incomplete.
- **Evidence.**
  - Today the default is 0.9 (`base.py:206`), and every batch without
    `max_batch` is sized to three quarters of the usable memory at that moment
    (`base.py:1860-1864`, `memory.py:1292-1293`, `_MEMORY_BUDGET = 0.75`). The
    user documentation promises it: "batches fill the device memory until a
    fraction `1 - total_mem_usage` of the total remains free" (`README.md:318-321`),
    and `docs/user_guide/devices_memory.md:39-42`. The docstring says "Used only
    when no `max_batch` is set" (`base.py:724-726`).
  - Under the design the only reader of `total_mem_usage` in `integrate` is the
    backward check. Forward calls on every path grow until the GPU refuses an
    allocation, and the size they settle at is 75 % to 100 % of whatever was
    free at that moment, with nothing kept back.
  - The meaning also flips: the check applies `total_mem_usage` even when the
    user gave `max_batch`, which today switches it off.
- **Consequence.**
  - Other work on the same GPU (a second process, a notebook, another rank of
    the user's own job) sees the GPU filled to the last byte during growth and
    can fail with an out-of-memory error that nothing retries.
  - After convergence a `take_gradient=False` run keeps no reserve, so the
    user's own next allocation outside padaquad can fail.
  - A user who lowered `total_mem_usage` to share a GPU gets no effect on
    `take_gradient=False` calls and no message. The brainstorm's own "LIMITED"
    test cases used exactly this knob.
- **What to change or decide.** Owner to decide between: (a) state it plainly in
  the documentation and change the docstring and the two documentation
  sentences; or (b) keep the promise cheaply by not growing a batch when the
  usable memory before the call is below the peak of the last batch times the
  growth factor (two counter reads per growth step, no model). Option (b) is my
  idea and is not tested.

### F6. Incomplete: a batch size that shrank never grows back

- **Whose decision:** the owner's ("Converged by an out-of-memory error").
- **Claim** (section 5.3): "Once converged, the batch size never grows again in
  this or any later call that uses the state."
- **Verdict:** incomplete; the consequence is not stated.
- **Evidence.**
  - Today, without `max_batch`, every batch is sized from the memory free at
    that moment, so the size follows memory up and down within a call and
    between calls. Today's reuse path (`max_batch=result.max_batch`) has a fixed
    size, but nothing lowers it permanently: the cap after a failure lasts one
    call (`base.py:865-867`).
  - Under the design every failure, and every time the backward check says
    "shrink" (which is not a failure, and in F2 happened in 26 of 65 trials near
    the limit where nothing was wrong), lowers the size in the state for good.
  - A concrete way to get there without any real shortage (my reasoning from the
    code, not run): with `take_gradient=False` and learnable parameters, the
    stored intermediate values of every batch are kept until the user calls
    `backward()` (`base.py:720-723`). Late in a call memory is nearly full of
    them, a batch fails, and the state leaves that call converged at a small
    size, although the next call starts with the memory free again.
- **Consequence.** Over a long training run the batch size can only go down. One
  short shortage (an evaluation loop that held memory, another job) lowers it
  for every later call. With `take_gradient=True` a smaller batch also changes
  the results (F3) and adds turns of the loop. The user is not told, because the
  design logs no message on growth and the shrink warning appears once.
- **What to change or decide.** Owner to decide: (a) accept and document, with
  the one-line reset `dataclasses.replace(state, max_batch_converged=False)`;
  (b) let a converged state try one growth step again after some number of
  calls; or (c) record in the state why it converged (given by the user, or
  found by a failure), so that only the second kind is retried. Also decide
  whether a shrink by the backward check should set "converged" at all.

### F7. Incomplete: the restart after a failed backward pass, and the default user without `parameters=`

- **Whose decision:** the restart from scratch and the optional `parameters=`
  are the owner's; the completeness check and the hook rule are the author's.
- **Claims** (section 6.4): "padaquad cannot undo it by itself: `f` can be any
  function, so padaquad does not reliably know the model's parameters"; "walk
  the loss's autograd graph [...] and raise `ValueError` if it reaches a tensor
  that requires a gradient and is [not in] the list"; "Parameters hidden inside
  activation checkpointing [...] are not seen by this check".
- **Verdict:** the mechanism is confirmed; the check is incomplete; the first
  claim contradicts the second.
- **Evidence** (`exp_gpu_oom2.py` on the A100, `exp_login.py` on the login node).
  - **Confirmed: a failed backward pass leaves part of its gradient.** With the
    memory for the backward pass capped at 0.25 to 0.95 of X, the pass failed
    after 0.007 to 1.9 s, and each time the last layer's weight and bias already
    held this batch's full gradient while the others did not: 2 of 8 parameters
    in 8 trials (3 layers of 512), 2 of 18 in 4 trials (8 layers of 256).
    Parameters that held a gradient before the batch kept it unchanged where the
    pass had not reached them.
  - **Confirmed: the GPU is usable afterwards.** After each of these 12 failures
    and 4 more in `exp_gpu_oom.py`, garbage collection took 0.021 to 0.056 s and
    emptying the cache 0.009 to 0.098 s, `torch.cuda.synchronize()` returned at
    once, and the same batch run again after resetting the gradients gave the
    reference gradients bit for bit (16 of 16). Over all my GPU scripts more than
    50 out-of-memory errors were caught and every script went on to correct
    results. The comment at `memory.py:517-520` (a stream left in an error
    state) did not occur here; I cannot say it never does.
  - **The walk finds the parameters by itself.** On a small network it returned
    exactly the 4 parameters. So padaquad can know which tensors a backward pass
    will write to, without being told, in the same cases in which the
    completeness check works at all.
  - **A check before the first backward pass only is not enough.** For an `f`
    with a branch that depends on the data, the first batch reached 2 tensors
    and a later one 4. The two unlisted ones pass the check, get gradients later,
    and are not restored by a restart: gradients counted twice, silently, which
    is an anti-goal.
  - **Legitimate code would raise.** A learnable scale that requires a gradient
    but is not in the optimizer is "a tensor that requires a gradient and is not
    in the list".
  - **The checkpointing sentence is wrong for the recommended mode.** With
    `checkpoint(..., use_reentrant=False)` the walk found all 4 parameters
    (torch 2.8.0). `use_reentrant=True` was not tested.
  - **Cost.** The walk covered 0.57 million graph nodes per second: 0.18 s for a
    graph of 100,003 nodes (an integrand with a 20,000-step loop inside). A
    vectorized network has tens of nodes whatever the batch size, so the cost is
    usually negligible, but it is paid on every call.
  - **Hooks.** `_post_accumulate_grad_hooks` is `None` without such a hook and a
    dictionary with one, as the design says. Hooks added with the ordinary
    `tensor.register_hook` are not in it; a restart runs them again.
  - **What a restart repeats besides gradients** (reading of the design, not
    run): every side effect of `f` and of `loss_fxn` (counters, logs,
    running statistics of normalization layers, random draws; the fresh mesh is
    kept, but the random number generator has moved on). The same holds, in
    small, for every forward retry and for every batch the backward check drops
    and evaluates again.
  - **Multi-GPU detection.** padaquad itself starts a `torch.distributed` process
    group whenever the environment variable `RANK` is set
    (`distributed.py:201-204`, `316-320`). "A process group with more than one
    process is running" is therefore true for every solver in a job started
    with `torchrun`, whether or not the model's gradients are averaged.
- **Consequence.** `take_gradient=True` is the default and `parameters=` is
  optional, so most users get the exception, not the restart. With the backward
  check in place that should be rare for ordinary networks (F2), and it is what
  happens today too (a plain out-of-memory error, with the same partial
  gradients and no hint). So this is not a regression; it is a protection that
  most users will not have.
- **What to change or decide.**
  1. Owner to decide whether `parameters=` is needed at all. Alternative (my
     idea, not tested): before every backward pass, walk that batch's graph and
     save the `.grad` of each tensor seen for the first time in this call. That
     needs no argument, covers branches that appear later, and fails only for
     custom autograd functions that hide tensors, where it can raise as now.
     Cost: one walk per batch.
  2. If `parameters=` stays, run the completeness check on every batch, and
     decide what an unlisted tensor does (raise, or be saved as well).
  3. Correct the checkpointing sentence.
  4. Say in the user documentation what a restart repeats.
  5. Decide the multi-GPU rule from what the user's model does, not from the
     process group padaquad creates.

### F8. Incomplete: the new error is caught by padaquad's own out-of-memory handlers, and cannot be pickled

- **Whose decision:** the author's.
- **Claim** (section 6.4): "`padaquad.BackwardOutOfMemoryError`, a subclass of
  `torch.OutOfMemoryError` (so existing `except torch.OutOfMemoryError` clauses
  still catch it). Its attribute `state` is the state to call again with".
- **Verdict:** confirmed that it can be built; incomplete.
- **Evidence** (`exp_login.py`, login node, torch 2.8.0).
  - A Python subclass of `torch.OutOfMemoryError` with an extra attribute can be
    raised and caught; its base classes are `OutOfMemoryError`, `RuntimeError`,
    `Exception`.
  - `padaquad.quadrature.memory._is_out_of_memory` returns True for it
    (`memory.py:242`), and `except (RuntimeError, MemoryError)` catches it. Those
    are the clauses of `_measure_call` (`memory.py:1215`) and
    `_evaluate_f_in_batches` (`base.py:2043`), and the retry of section 6.2 will
    be a third.
  - With `state` as a required argument of `__init__`, `pickle.loads(pickle.dumps(e))`
    fails with `TypeError: __init__() missing 1 required positional argument`.
- **Consequence.**
  - If the new error travels through a call of `f` (an integrand that itself
    calls a padaquad solver with `take_gradient=True`, as in a double integral),
    the outer solver treats it as an ordinary out-of-memory error, shrinks its
    batch and calls `f` again, while the inner call's partial gradients stay.
    That is the double counting the error exists to prevent.
  - An error that cannot be pickled is lost or replaced by a `TypeError` when it
    crosses a process boundary (`multiprocessing`, `concurrent.futures`).
- **What to change.** Make `_is_out_of_memory` return False for the new class;
  give the class a `__reduce__`, or set `state` after construction; add both to
  the tests.

### F9. Unsupported: what the three test calls are for (owner's decision)

- **Whose decision:** the owner's ("it needs to run the test calls"); the
  reading that they "no longer measure anything" is the author's.
- **Claims** (section 1): "four small test calls [...] that only check whether
  `f` runs at those sizes". Section 2: the time spent on memory "is a few
  milliseconds".
- **Verdict:** unsupported (the cost), and the benefit is not stated.
- **What they protect against.** On a GPU, nothing that the first real batch
  does not also find: the design adds a forward retry to both paths (sections
  6.1 and 6.2), so an `f` that cannot take 253 points is caught and retried
  there. They do give steps of 10 from one point, which matters only on a CPU,
  where the design accepts the risk anyway.
- **What they cost** (`exp_extra.py`; median of 7; the integration is a cold call
  with a fixed batch size, atol = rtol = 1e-6, `gk21`):

  | Device | Integrand | Points of the integral | Integration (s) | Calls on 10 + 100 + 1,000 copies (s) | Ratio |
  |---|---|---|---|---|---|
  | CPU, 64 threads | sin(1e3 t) | 2,967 | 0.0086 | 0.0000 | 0.01 |
  | | 3 layers of 512 | 253 | 0.0057 | 0.0050 | 0.88 |
  | | 6 layers of 2048 | 253 | 0.0475 | 0.1274 | 2.68 |
  | | Force (gradient inside `f`) | 253 | 0.0096 | 0.0324 | 3.37 |
  | A100 | sin(1e3 t) | 2,507 | 0.0225 | 0.0001 | 0.01 |
  | | 3 layers of 512 | 299 | 0.0113 | 0.0008 | 0.07 |
  | | 6 layers of 2048 | 299 | 0.0112 | 0.0038 | 0.34 |
  | | Force (gradient inside `f`) | 253 | 0.0078 | 0.0024 | 0.30 |

  Rows are `take_gradient=False`; with `take_gradient=True` the ratios were 0.55,
  0.83 and 1.68 on the CPU and 0.06, 0.25 and 0.26 on the GPU. So "a few
  milliseconds" holds on the GPU and not on a CPU for an expensive integrand
  (0.13 s, 2.7 times the integral). Both are far below today's 0.7 to 1.6 s.
- **What they miss.** The backward pass; every value of t but one (an integrand
  with branches or tables may behave differently on 1,000 copies of the lower
  bound); the stored graph of a real batch when the test point needs none.
- **What they can do wrong.** A test call of 1,000 points is larger than the
  whole first pass (253 points by default). An integrand that holds 300 points
  but not 1,000 leaves the test calls with a state that is "converged" at 750
  or 562 points before the integral has started, by failed calls that the
  integral did not need (cost as in F4).
- **Smaller points.**
  - The "Terms" section still defines a test call as "made only to read how much
    memory it uses"; section 5.2 says no memory is read.
  - Section 5.2 says "nothing is done after the calls (no garbage collection, no
    emptying of the GPU cache, no sleep)" and also that a failed test call
    follows "the same rule as every other out-of-memory retry", which includes
    that clean-up.
- **What to decide.** Owner: keep the test calls for what they do on a CPU and
  as a habit of caution, or drop them and let the first pass be the first test.
  If they stay, cap them at the size of the first pass.

### F10. Vague: which parts of today's per-call machinery survive

- **Whose decision:** the author's.
- **Claim** (section 5.2): "`_setup_memory_checks` stays, unchanged in what it
  measures, as the engine of `solver.get_max_batch(f=..., t=...)`, its only
  caller."
- **Verdict:** vague. The design names what `integrate` stops doing, not what
  happens to the pieces. From the code:

  | Piece | Used today by | Under the design |
  |---|---|---|
  | `_setup_memory_checks`, the reader, the fit (`memory.py:523-1240`) | `integrate`, `get_max_batch(f=...)` | `get_max_batch(f=...)` only |
  | `f_mem_coeffs`, `_get_max_f_evals` | every batch without `max_batch` (`base.py:1860`), `get_max_batch()` | `get_max_batch()` only |
  | `_oom_max_batch` on the solver (`base.py:356`, `867`, `2061`) | the retry, `_cap_max_batch` | not said. The state now holds "the batch size after a failure"; keeping the attribute too is the same value in two places. |
  | `_cap_max_batch` | every batch, `result.max_batch`, `get_max_batch` | not said |
  | `limit_benchmark_time` (constructor) | the benchmark | `get_max_batch(f=...)` only (section 10) |
  | `total_mem_usage` (constructor and call) | every batch without `max_batch` | the backward check and `get_max_batch` (F5) |
  | the warning "max_batch is 0, increasing to 1" (`base.py:1872-1881`) | a collapsed memory estimate | its cause no longer exists; its text is wrong for a state |
  | the assert "Not enough free memory to run 1 integration steps" (`base.py:1960-1962`) | `take_gradient=True` with a batch below one subinterval | reachable from a state (F11) with a misleading text, as an `AssertionError` |

- **Consequence.** The package keeps two ways of finding a batch size that do
  not agree: the benchmark's fit with the factor 2.1 (about 700 lines, now
  serving one optional helper), and growth by trying. A number from the first,
  passed as `max_batch`, is then policed by the second. The owner's rule is one
  source of truth; he should decide knowingly that both stay.
- **What to change.** Add this table to the design with the last column filled
  in, and remove `_oom_max_batch` from the solver if the state replaces it.

### F11. Vague: edge cases of the state

- **Whose decision:** the author's.
- **Verdict:** incomplete. Each item is small; together they are the difference
  between a design and a sketch.
- **Checked on the login node** (`exp_login.py`, torch 2.8.0):
  - Confirmed: `dataclasses.asdict(state)` gives a tensor, an `int` and a `bool`,
    and `torch.save` followed by `torch.load` with its default
    `weights_only=True` restores them. `asdict` deep-copies the mesh.
  - `state == other_state` raises `RuntimeError` ("Boolean value of Tensor with
    more than one value is ambiguous") when the meshes are different objects.
    Declare the dataclass with `eq=False`.
  - If `max_batch` is ever a NumPy integer, the default `torch.load` refuses the
    file. The code must store a Python `int`.
  - `asdict` raises on a mesh that is not a leaf tensor and requires a gradient
    (possible after `dataclasses.replace(state, mesh=m)`).
  - "Frozen" does not protect the mesh: `state.mesh[0] = -5.0` goes through.
  - A plain dataclass validates nothing: `IntegrationState("not a mesh", -3,
    "yes")` is accepted. The design has no rule for checking a state that a user
    edited or loaded (the mesh is checked by `_setup_integral_bounds`; the two
    other fields by nothing).
- **Read from the design and the code, not run:**
  - **Three meanings of `max_batch` in one field.** Section 4: "the batch size
    the next call starts with". Section 5.3: "the largest batch known to fit".
    After test calls it is 1,000 forward-only copies of one point; after
    `max_batch=n` from the user it is a number nobody ran.
  - **A state reused under other settings** (`take_gradient`, `torch.no_grad()`,
    dtype, device, integrand). The design says nothing is stored about them.
    The retries then have no bound (F4).
  - **A batch size below the nodes of one subinterval.** On the
    `take_gradient=True` path the number of subintervals is `max_batch // C`
    (`base.py:1889`), and 0 trips the assert at `base.py:1960`. A state can
    arrive there from a failed test call at 10 points (retried at 7) or from a
    `take_gradient=False` call. Section 6.4 covers this for restarts only.
  - **A call that raises returns no state**, except through the new error. The
    caller then retries from the old state and repeats the same failures.
  - **`solver.get_max_batch()` without `f`** after the design: the constructor's
    value, else the last explicit benchmark's fit, else `None`. It no longer
    reflects any `integrate` call; its docstring (`memory.py:345-351`) says
    otherwise.
- **What to change.** Add a short section "Rules of the state" that settles each
  item, and a test for each.

### F12. Incomplete: the amount of test and documentation work

- **Whose decision:** the author's.
- **Claim** (section 10): "Everything that passes them on (README,
  `repeated_calls.md`, `troubleshooting.md`, `devices_memory.md`,
  `examples/pode/trainer.py`, `examples/quadrature_basics.py`, about 30 tests)
  moves to `state=`."
- **Verdict:** the number 30 is confirmed for what it counts; the list is
  incomplete.
- **Evidence** (a count over the source of the 645 test functions in `tests/`):
  - 33 test functions use `mesh_optimal` or a result's `max_batch` (24 and 15,
    with overlap). This matches "about 30".
  - 39 more, in 2 files, exercise the benchmark through names such as
    `_setup_memory_checks`, `f_mem_coeffs` or the warning texts; those that
    reach it through `integrate` must be rewritten or dropped.
  - 13 use `_oom_max_batch` or `_cap_max_batch`; 29 call the evaluation
    functions whose signatures change (`tests/test_evaluate_f_on_mesh.py`).
  - 107 distinct test functions match at least one of these.
  - `tests/_helpers.py` gives nearly every solver a `max_batch`
    (`cached_max_batch`, used in 11 files). Under the design that makes every
    such state converged, so the existing tests keep running single-batch (good
    for the snapshots) and never exercise growth. All coverage of growth, the
    retries, the backward check and the restart must be new.
  - Files that use the removed attributes and are not in the design's list:
    `examples/verification/merge_prediction.py` (lines 118-119),
    `docs/user_guide/how_it_works.md`, `docs/reference/solvers.md`,
    `docs/glossary.md`, `docs/development/architecture.md`,
    `docs/development/docstrings.md`, `CLAUDE.md`, and the docstring examples in
    `padaquad/quadrature/base.py:783-797`, `padaquad/_frontend.py:73-87` and
    `padaquad/results.py`, which the test suite runs.
  - The limitations section that the benchmark's warning links to
    (`ug-memory-limitations`) describes the benchmark as part of `integrate`; it
    needs rewriting, not three added lines.
- **What to change.** Replace "about 30 tests" by this list.

### F13. Incomplete: "grow until it fails" needs a failure that can be caught

- **Whose decision:** the owner's (accepted for CPUs).
- **Claim** (section 1): "The batch size grows whenever a pass needs more, until
  the GPU runs out of memory; then it shrinks and stops growing."
- **Verdict:** confirmed on the machine I tested; incomplete elsewhere.
- **Evidence.**
  - On Linux with an A100, every out-of-memory error I caused (more than 50, in
    forward and backward calls) was an ordinary `torch.OutOfMemoryError`.
  - On Windows, NVIDIA drivers from version 536.40 let a program "use shared
    memory in cases that exhausted the GPU memory", so a too-large batch becomes
    very slow and does not fail; a setting to turn this off exists from driver
    546.01 (NVIDIA support article 5490, "System Memory Fallback for Stable
    Diffusion", https://nvidia.custhelp.com/app/answers/detail/a_id/5490). There
    the design never converges by failure; it runs in system memory.
  - Recognition is by error class or by words in the message (`memory.py:242-250`).
    An error of a GPU library with other words is raised to the user as it is.
  - PyTorch's own advice on catching these errors is the pattern the code
    already uses: act after the `except` block, because "the python exception
    object holds a reference to the stack frame where the error was raised"
    (PyTorch documentation, Frequently Asked Questions,
    https://docs.pytorch.org/docs/2.15/notes/faq.html). The new retry of section
    6.2 and the restart of 6.4 must follow it.
- **What to change.** List Windows with system-memory fallback as a limitation
  next to the CPU one, and say that `max_batch` is the way out there.

### F14. Confirmed claims

| Claim (section) | How I checked | Result |
|---|---|---|
| A cold call's results equal the unlimited run (implied by 7.4) | 9 emulated comparisons, 3 methods (F3) | Identical, difference exactly 0 |
| `take_gradient=False` without `conserve_memory`: the batch size only cuts a pass into calls (7.4) | `_evaluate_f_on_split_nodes` evaluates every waiting subinterval, `base.py:2139-2146`; one injected retry left the integral exact | Confirmed for this path only (F1) |
| "Today `_evaluate_f_on_full_nodes` has no retry" (6.2); a forward retry writes no gradient | `base.py:1935`, `1971`; the mesh is not changed before `base.py:1080` | Confirmed |
| A failed backward pass leaves part of its gradient (6.4) | F7 | Confirmed, 12 of 12 |
| The GPU can be used after such a failure (assumed by 6.4) | F7 | Confirmed, 16 of 16 bit-identical reruns |
| "Gradients of the bounds are not affected" by a restart (6.4) | each batch is evaluated on a detached copy of the mesh and the bounds get their gradient once, after the loop (`base.py:927-939`, `1171-1181`) | Confirmed by reading |
| A subclass of `torch.OutOfMemoryError` with a `state` attribute (6.4) | F8 | Works, with the two gaps of F8 |
| `asdict`, then `torch.load` with the default `weights_only=True` (4) | F11 | Works |
| `node.variable` and `_post_accumulate_grad_hooks` (6.4) | F7 | As described, torch 2.8.0 |
| At most about 8 retries after a growth step (5.4) | F4 | 8 observed; holds within one setting |
| `mesh_optimal` was released in 1.4.1, `max_batch` was not, 2.0.0 is not tagged (10) | `git show v1.4.1:padaquad/results.py`, `git tag` | Confirmed |
| The new start warning and end clause (8) | compared with `memory.py:684-706`, `1006-1014` | Consistent with the rest of the design; `tests/test_memory_benchmark.py` pins the old texts in 3 places |

## 3. Claims that borrow confidence from measurements of another design

No part of this design has been run. Its only numbers come from the emulation of
"option E" (brainstorm section 2.6), a design in which test calls and a memory
model made the batch size cover every pass before it ran, and in which nothing
ever ran out of memory.

- **"A few milliseconds"** (section 2) is option E's 2 to 8 ms, for test calls
  that stopped at a tenth of the first pass. The present test calls always go to
  1,000 points; measured here, under 0.1 ms to 0.15 s (F9).
- **Unchanged results.** Option E's 63 of 63 identical results came from a batch
  size that always covered the pass. Here that holds for cold calls for a
  different reason (F3) and fails for warm calls, large meshes and frozen
  states.
- **No out-of-memory error.** Option E never failed a call. This design fails
  calls on purpose, and no failure cost had been measured before this review
  (F4).
- **"No clean-up after test calls"** rests on option E's measurement and
  carries over.
- **The motivation** (0.7 to 1.6 s on a CPU node, about 0.1 s on an A100) is a
  measurement of today's code and stands. Note what it implies: on a GPU, where
  the design says padaquad belongs, today's cost is 0.1 s per call, of which
  0.05 s is a sleep added by accident (brainstorm section 2.3). The design was
  not compared with the two-line fix for that.

## 4. The out-of-date plan

`2026-11-17-memory-benchmark-on-demand-plan.md` implements option E. Tasks 1 to 3
(a benchmark search that can be resumed, the `n_needed` argument, `integrate`
calling it before every pass) have no place in the new design. Still usable:

- **Global Constraints:** the branch and commit rules, the Python environment,
  "never run the suite on the login node", the docstring, mypy, ruff and
  vocabulary rules, and judging test runs by the set of failing tests against a
  recorded baseline.
- **Review Focus, items 1 to 3** as test situations: a warm call from a large
  mesh, `conserve_memory=True`, float32 with another method, tracked variables.
  This review shows the first two are where the new design breaks.
- **Task 4 as an idea:** measure the implementation before writing numbers into
  the documentation.
- **Design decision 3** (no clean-up after test calls), which the design keeps.

## 5. Experiments

Every number was checked against the raw output file named in the row.

| # | What | Where | Result | Script, output |
|---|---|---|---|---|
| 1 | X against F for 11 models at up to 5 batch sizes | A100 80 GB | X/F 0.08 to 0.77 above N = 1,000 for ordinary models; up to 5.4 at N = 253; 4.1 to 6.3 for one checkpoint segment (F2) | `exp_gpu_mem.py`, `gpu_mem.out`; `exp_extra.py`, `extra.out` |
| 2 | The backward check against a real backward pass, 6 models, 13 batch sizes each | A100 80 GB | Ordinary models: 31 passes, 0 failures, 26 needless shrinks. One checkpoint segment: 7 passes, 7 failures (F2) | `exp_gpu_check.py`, `gpu_check_a.out`, `_b`, `_c` |
| 3 | Growth by 10, then 0.75, with the real clean-up, 3 models | A100 80 GB | 2 to 8 failed calls, 4.9 to 6.3 s lost (F4) | `exp_gpu_oom.py` part A, `gpu_oom.out` |
| 4 | Failure inside `loss.backward()` by a memory cap, 12 trials | A100 80 GB | 2 of 8 and 2 of 18 gradients already written; GPU usable; rerun bit-identical 12 of 12 (F7) | `exp_gpu_oom2.py`, `gpu_oom2.out` |
| 5 | Failure inside `loss.backward()` by filling the GPU with other tensors, 4 trials | A100 80 GB | Failed at the first allocation, before any gradient was written; GPU usable; rerun bit-identical 4 of 4. My filler split the free memory into pieces, so this says nothing about fragmentation in real use | `exp_gpu_oom.py` part B, `gpu_oom.out` |
| 6 | Time of emptying PyTorch's GPU cache | A100 80 GB | 0.005 to 0.033 s for blocks of 1 GiB or more (12 to 60 GiB cached); 0.019 s and 0.575 s for 16 and 31 GiB in 64 MiB blocks | `exp_gpu_oom.py` part C, `gpu_oom.out` |
| 7 | Growth rule emulated with `take_gradient=True`, 3 integrands, 3 methods, cold, warm and large-mesh calls | CPU of the allocated node | F3 | `exp_growth.py`, `growth.out`, `growth_bosh3.out`, `growth_dopri5.out` |
| 8 | Time of the three test calls against a small integral, 4 integrands | CPU (64 threads) and A100 | Ratio 0.01 to 3.37 on the CPU, 0.01 to 0.34 on the GPU (F9) | `exp_extra.py`, `extra.out` |
| 9 | `conserve_memory=True` with one injected retry, and with growth per call | Login node, CPU, explicit `max_batch` | 7 of 8 wrong in each (F1) | `exp_conserve.py`, `conserve.out` |
| 10 | The error subclass, the dataclass, the graph walk | Login node, CPU | F7, F8, F11 | `exp_login.py`, `exp_login.out` |
| 11 | Count of affected tests and files | Login node (text search) | F12 | commands only |

Notes on the runs.

- The first version of experiment 9 ran for about 3 minutes on the login node
  before I killed it; that was longer than the "seconds" allowed there. The
  version reported takes a few seconds and stops any run after 400 calls of `f`.
- `exp_gpu_mem.py` was extended with the one-segment model after experiment 1
  had run; the rows of that model come from `exp_extra.py`.
- GPU scripts ran on separate GPUs of the node at the same time as the CPU
  emulation, so timings can scatter by tens of percent. Memory numbers do not
  depend on that.

## 6. Confidence

Sure (measured here, or read in the code):

- F1: the wrong integrals of `conserve_memory=True` after a retry, on today's
  code, for `gk21` and `gk15`. I read the lines that cause it, but I did not
  prove that the empty slice is the only cause.
- The X/F table and the outcome of the backward check for the models, dtype
  (float64) and GPU listed.
- The timings of failed calls on one A100, as orders of magnitude.
- A failed backward pass leaves part of its gradient, and the GPU works
  afterwards, on this machine.
- The code references for `total_mem_usage`, live sizing and the gradient path.
- The counts of tests and files.

Less sure:

- F3 is an emulation that replaces the batch size inside today's loop. It is
  exact for the `take_gradient=True` path as the design describes it, but a real
  implementation can differ in detail. The integrands were built to make the
  running integral misleading; I do not know how often real training integrands
  behave like that. With the default method the true error never left the
  tolerance.
- That the fixed part of X is the memory of the parameter gradients.
- The comparison "today's rule allows 1.7 million points" is arithmetic from
  constants, not a run of today's benchmark.
- The reasoning that a cold call fails at most 3 times, the shrink formula in
  F2, and all proposals marked "my idea": none was run.
- F6's example with `take_gradient=False` and stored graphs was not run.

Not checked:

- Any real workload of the owner's (popcornn, the `examples/pode` trainer).
- CPU out-of-memory behavior, macOS, Windows; the NVIDIA fallback is from the
  vendor's page, not observed.
- float32, mixed precision, more than one GPU, `torch.distributed`, a GPU other
  than the A100 80 GB.
- `checkpoint(..., use_reentrant=True)`, custom autograd functions, a custom
  `loss_fxn`, tracked variables, bounds that require a gradient.
- Why the first version of experiment 9 did not finish.
- User documentation pages line by line; I read only the passages cited.

## 7. Connections

- Finding the batch size by failing is what PyTorch Lightning's batch-size
  finder and Hugging Face Accelerate's `find_executable_batch_size` do (sources
  in the speed brainstorm, section 7). Both rerun a whole training step that
  starts by clearing gradients, which is why they do not meet the partial
  gradients of F7; padaquad's per-batch backward pass inside one call is the
  difference.
- F3 is the same effect the earlier review reported as its finding F2 (results
  depend on the batch outside the default path); the state makes the batch size
  a quantity with a history.
