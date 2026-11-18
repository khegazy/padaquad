# Recovering from an out-of-memory error in the backward pass during multi-GPU training

Status: open, for the future. Recorded 2026-11-17 while designing the
integration state (that design, section 6.4, is in git history: commit
`6b3cf42`, `docs/issues/2026-11-17-integration-state-design.md`; the
restart it describes is implemented on branch `change_memory`).
Written so that another agent can pick it up without the original session.
Nothing here was run or measured.

## Summary

With `take_gradient=True`, padaquad calls `loss.backward()` after every batch.
When a backward pass runs out of GPU memory and the user passed `parameters=`
(the model's parameters, or its optimizer), the design restores the gradients
those parameters had at the start of the call and starts `integrate` again from
scratch with a smaller batch size.

In multi-GPU training this restart is not safe, so the design raises the error
instead (the same error a user gets without `parameters=`). This issue is about
making a restart work there.

## Why a restart on one GPU is not safe

The reasoning below is from how PyTorch's distributed training works; none of
it was tested with padaquad.

- `torch.nn.parallel.DistributedDataParallel` (DDP) averages gradients across
  GPUs during each backward pass: hooks on the parameters start a collective
  operation (all-reduce) in which every process must take part.
- So every process must run the same number of backward passes, in the same
  order. If one process restarts its integral after an out-of-memory error, it
  runs extra backward passes that the others do not match. The others then
  wait in a collective operation that never completes (until the process
  group's timeout), or the averaged gradients mix different batches.
- Restoring the gradients on one process does not undo the averaging that
  already happened on the others.

A question to answer first: padaquad refines each process's mesh adaptively,
so even without any error, different processes can make different numbers of
batches, and so different numbers of backward passes. Whether
`take_gradient=True` works at all with a DDP-wrapped model today is not
verified. `padaquad/distributed.py` sets up process groups and devices; no code
in `padaquad/quadrature/` performs a collective operation.

## What padaquad does now

- Detects a running `torch.distributed` process group with more than one
  process (`torch.distributed.is_initialized()` and `get_world_size() > 1`,
  `padaquad/quadrature/base.py:1279-1280`).
- In that case never restarts: it raises the backward out-of-memory error,
  which carries a ready state (starting mesh, smaller batch size, converged)
  and says that the gradients now hold a partial result and must be reset.

## Possible approaches (not decided)

1. **All processes agree on a restart.** After every backward pass, the
   processes exchange one flag (an all-reduce of "did I run out of memory").
   If any did, all restore their gradients and restart together, with the
   smallest batch size. Costs one small collective operation per batch, and
   needs all processes to make the same number of batches, which adaptive
   refinement does not guarantee.
2. **No averaging during the integral.** Run the per-batch backward passes
   under DDP's `no_sync()` context manager, so gradients are only added
   locally, then let one final step average them. A local restart is then
   safe, since nothing was shared yet. Needs a way to trigger the final
   averaging, and changes how padaquad interacts with the user's training
   loop.
3. **Leave it to the user.** Keep raising the error, and document how to
   restart all processes together in the user's training loop.

## Checks before choosing

- Run `take_gradient=True` with a DDP-wrapped model on 2 GPUs, with meshes that
  refine differently on each process, and see whether it completes today.
- Confirm in the PyTorch documentation how DDP behaves when processes make
  different numbers of backward passes (the `join()` context manager exists for
  uneven inputs).
