# Speed-up plan: the fixed cost of every `integrate` call (`call-setup-cost`)

**Status:** plan, reviewed and corrected 2026-11-25/26 against commit `d11df47`
(branch `call_setup_cost`); first written against `0590ccf` on branch
`fix_conserve_memory`. Line numbers below are of `d11df47`. Prototype and
measurements of the planning stage in
`$SCRATCH/padaquad_speedup_plans/call-setup-cost/` (Section 7); scripts of the
review and implementation in its subfolder `impl/`.

## Review of 2026-11-25/26

Checked every claim against the code at `d11df47` and by experiments
(`impl/` scripts; torch 2.8.0, CPU of a Perlmutter login node, AVX2). Found
wrong, and changed in this file:

1. **D4's claim was wrong.** The current expression
   `int(torch.sqrt(torch.tensor(N, dtype=torch.float)).to(torch.int))` first
   differs from `math.isqrt(N)` at N = 9,443,328 = 3073² − 1, not at
   16,785,407: PyTorch's float32 square root on this CPU is not correctly
   rounded there and returns 3073 (3,368 differing values up to 3·10⁷, all
   one more than `isqrt`; `impl/d4_check.py`, `impl/d4_scalar.py`, the scalar
   path exactly as in the code). Also, `math.isqrt` refuses a float
   (`N_init_steps=1e4` works today and would raise `TypeError`), a public
   change the user was not told of; the step now uses
   `math.isqrt(int(N_init_steps))`, which keeps floats working.
2. **The tolerances are not "the same bits" in every case.** As Python floats
   they give the same bits as the 0-dimensional tensors whenever the values
   they meet have the solver's dtype (2,700 of 2,700 cases on the CPU,
   including NaN, ±Inf, ±0, subnormals, 0-dimensional operands and
   `torch.clamp` for `torch.maximum`; `impl/tol_bits_cs.py`). They differ when
   a float32 solver meets float64 values, which happens when the integrand
   returns float64 to a float32 solver (the integral then has the wider dtype
   of the weights and the values): the old code used the tolerances rounded to
   float32 in float64 arithmetic, the new one uses the given values (a
   relative difference of about 1e-8 in the tolerance; 1,048 of 1,350 cases
   differ). The opposite case (float32 values, float64 solver) cannot occur:
   the integral has at least the solver's dtype. Section 3.2 now says so, and
   the grid harness has cases for it.
3. **`t.to(device).to(dtype)` returns the source tensor itself** when the
   device and dtype already match (a float64 solver on the CPU). The plan's
   "converted" object would then share its tensors with the registry, and an
   in-place write into a solver's table would reach the registry and every
   later solver. The conversion now copies (`.to(dtype, copy=True)`), so no
   converted method shares memory with a registry entry.
4. **Freezing `_Tableau` does not stop in-place writes** (`method.tableau.b[0] =
   1.0`, `b.mul_(2)`). PyTorch 2.8 has no read-only tensors; tensors made in
   inference mode refuse in-place writes but cannot be saved for backward
   ("Inference tensors cannot be saved for backward", checked with `b * y`,
   `y` requiring grad), so they cannot hold the weights. The hazard is bounded
   instead (Section 3.1): conversions never share memory with the registry;
   the package never writes into a method's tensors (a new test checks the
   tensors' version counters across integrations of both solvers, with and
   without `take_gradient`, with `result_device` on another device and with
   the merge prediction); the developer pages say that a method's tensors must
   not be modified in place.
5. **Cache keys.** `torch.device("cuda")` and `torch.device("cuda:0")` are
   different keys; an index-less CUDA device now resolves to the current
   device in `MethodClass.to`, as `Tensor.to("cuda")` does. The solver itself
   always passes an indexed device.
6. **The tolerance moves after the loop** (four host waits) are in the two
   `_compute_error_ratios` calls of `_get_optimal_mesh` (3277, 3298), whose
   inputs are on `result_device`; the final check (1711) runs on the
   integration device, where the moves do nothing.
7. **D6 (`_flatten_output` in one copy) is already on `main`** (pull request
   #74, `quadrature/base.py:4541`). Removed from this plan.
8. **The merge weight cache stays on the solver.** Moving it onto the shared
   method object belongs to `optimal-mesh-cost`, is not needed for
   `_integration_on_device` to swap method objects (the per-solver cache is
   already cleared when the device changes, 3114-3116), and a cache shared
   between solvers would make a call's result depend on earlier calls of
   other solvers unless the weight vectors are proven independent of how the
   width ratios are grouped for every method (not shown for
   `interpolatory3_variable`); it also depends on solver class constants
   (`_RATIO_QUANTUM`, `_MERGE_CACHE_SIZE`). Not done here.
9. **The bounds' combined read (old Section 3.3) is replaced** by a smaller
   change: with a mesh and no bound (`integrate(f, state=state)`, the
   README's training loop), there is nothing to cut, so the cut (two
   `torch.equal`, a boolean index and about ten kernels) is skipped and the
   mesh returned as given, which its docstring already promises ("A mesh
   alone ... leave it unchanged"). The combined read for given bounds would
   rewrite the validation of user input (its error messages, the rounding
   snap, the autograd path of a bound) to save about two host waits per call;
   kept only if the measurement shows the saving (Section 7), otherwise left
   out for readability.
10. **A dtype change re-reads the registry by name** (`_set_solver_dtype` →
    `_get_method`), today and after the change: a solver switched to another
    dtype by its input after a user replaced the registry entry gets the new
    entry. Unchanged; noted in Section 3.1.
11. **Distributed construction (D5) confirmed by experiment** (`impl/d5_check.py`,
    gloo, one process with `RANK=0`): a second solver constructed while the
    first exists raises `ValueError: trying to initialize the default process
    group twice!`, and deleting a solver destroys the process group. Out of
    scope (Section 9).
12. **Vague steps rewritten** (Section 5 names files, functions and tests), the
    first plan's measurement scripts filter frames by the path
    `/padaquad/padaquad/`, which matches only the user's checkout; the scripts
    in `impl/` find the package from `padaquad.__file__`.
13. **The not-converged warning** formats the tolerances with `%.3g`, so it
    prints the same text with floats; the old remark that it "prints the value
    given instead of its float32 rounding" was moot.

## 1. Summary

- **What is slow.** Every call pays costs that are not evaluations of `f`:
  building a solver (`padaquad.integrate` builds one per call), a deep copy of
  the method and three copies of its tables to the GPU, the tolerances and the
  default `y0` as device tensors made from host values, the tables moved to
  `result_device` and back around the optimal mesh, the checks of a given
  mesh, the starting mesh, and a few reads of the device after the loop. On
  the GPU each read of a device value, and each copy from ordinary host
  memory to the device, makes the host wait for the GPU (a *host wait*, about
  15-20 µs each).
- **What the plan changes.** (1) Converted methods are built once per process
  and shared, never changed in place (`MethodClass.to`). (2) `atol` and `rtol`
  become Python floats (D2, droppable). (3) A mesh given without bounds is not
  cut. (4) The starting mesh, the start of the loop and the statistics after
  the loop drop avoidable reads. (5) `math.isqrt` for the starting mesh (D4).
- **Expected gain** (one-shot `damped_sine` on the GPU, about 6 ms): about
  0.4-0.6 ms; a second call with `state=`: about 0.3 ms of 5 ms. Measured
  before and after in Section 7.
- **Numbers:** unchanged bit for bit, except (a) D2 when the integrand returns
  float64 to a float32 solver (item 2 above) and (b) D4 for `N_init_steps`
  ≥ 9,443,328.

## 2. Evidence

### 2.1 The code at `d11df47`

Construction (every `padaquad.integrate` call; once per solver otherwise):

- `padaquad/_frontend.py:192-193`: a new solver per call.
- `padaquad/methods/__init__.py:119-121` (`_get_method`, defined at 72):
  `registry[name].clone()` (a `copy.deepcopy`, `methods/_base.py:424-437`),
  then `to_device` (three host-to-device copies, `_base.py:87-89`) and
  `to_dtype`. Called from `AdaptiveQuadrature._set_solver_dtype`
  (`quadrature/base.py:526-549`), which also empties the solver's
  `_merge_weight_cache`.
- `padaquad/base.py:194-195, 200-204`: `atol`, `rtol` and the default `y0`
  are made with `torch.tensor(..., device=self.device)`: three host-to-device
  copies, each a host wait on the GPU.
- `SolverBase._set_dtype` (`base.py:311`) runs twice per one-shot call: once
  in `__init__` (205) and once from `set_dtype_by_input` (415-420), where it
  returns at once (341) because the dtype is unchanged. Nothing to do.
- `DistributedEnvironment.__init__` (`distributed.py:127`): no host wait in a
  single process. See Section 9 for distributed runs.
- `_restart_after_backward_oom` (`quadrature/memory.py`): binds the arguments
  with `inspect.signature` on every call; 0.02-0.03 ms. Left alone.

Before the loop (`quadrature/base.py`, `integrate`, 944-1210):

- `_setup_integral_bounds` (3364-3506): without a mesh, the two bound moves
  (3454-3455) and the order check `if not mesh_init < mesh_final` (3479).
  With a mesh: the mesh order check (3418), the bound moves (3472, when bounds
  are given), the order check (3479), two `torch.equal` (3500) and the
  boolean index `mesh[keep]` (3504), also when no bound is given and the cut
  changes nothing.
- `_setup_initial_mesh` (3508-3616): the float32 square root (3553-3555, on
  the CPU), the random mesh (about 15 small operations), two asserts gated by
  `runtime_checks` (3608, 3611; for T = 1 the second implies the first),
  `torch.ones(...).to(torch.bool)` and `mesh_trackers[-1] = False`
  (3612-3613), and `_get_mesh_indices` (3614, owned by `record-order`).
- `int(mesh_trackers.sum()) * self.C` (1177): a host wait for a number known
  on the host (`n_start_steps = len(mesh) - 1`, 1162); only without
  `max_batch` and `state`.

After the loop (1552-1675):

- Split statistics: two reads (1562, 1563). Final check: one read (1585).
- `_error_ratios_norm` 3880-3881 and `_error_ratios_failure_fraction`
  3971-3972: `self.atol.to(abs_err.device)` and the same for `rtol`; nothing
  in the loop and the final check, but two device-to-host copies in each of
  the two calls of `_get_optimal_mesh` when `result_device` is another device
  (the default `"cpu"` with a CUDA solver): four host waits.
- `_integration_on_device` (3187-3216): `self.method.to_device` twice, six
  host waits with `result_device="cpu"` on a CUDA solver.

### 2.2 Measurements of the planning stage (2026-11-25, commit `0590ccf`)

Node `nid008404` (A100 shared with another agent; EPYC 7763), gk21,
`atol=1e-8`, `rtol=0`, `error_norm="max"`, float64, `max_batch=1_000_000`,
`take_gradient=False`, `damped_sine` = exp(-t) sin(40 t) on [0, 1] (11
subintervals, one turn), `result_device` the default `"cpu"`. Wall times are
medians and indicative only (shared GPU); counts are exact.

| Part | One-shot (64 waits) | `state=` call with bounds (61 waits) | Owner |
|---|---|---|---|
| Construction: `atol`, `rtol`, `y0` | 3 | 0 | this plan |
| Construction: `to_device` | 3 | 0 | this plan |
| `_integration_on_device`: `to_device` twice | 6 | 6 | this plan (by the shared converted methods) |
| `_setup_integral_bounds` | 3 | 7 | this plan (partly) |
| `_setup_initial_mesh` (asserts, tracker) | 3 | 1 | this plan |
| Split statistics and final check | 3 | 3 | this plan |
| `atol`/`rtol` moves in `_get_optimal_mesh` | 4 | 4 | this plan |
| Moves into and out of `_get_optimal_mesh` (1612-1629) | 5 | 5 | `optimal-mesh-cost` |
| Record, dictionary, sorting, loop, mesh update | 33 | 34 | other plans |
| Outside PadaQuad's frames | 1 | 1 | unknown |

Time by phase (ms; device synchronized at each mark): GPU one-shot 0.325
construction, 0.688 before the loop, 2.026 loop, 2.618 after the loop; GPU
`state=` 0.010 / 0.648 / 2.014 / 2.250. Construction 0.296 ms on the GPU, of
which `copy.deepcopy` 0.28-0.32 ms (cProfile) and `to_device` 0.067 ms.
Merge weight vectors: 0.43-0.50 ms per new solver (not this plan).

### 2.3 Prototype of the planning stage

`proto/` (a copy of `0590ccf`) keyed its method cache by method NAME and
swapped the method object in `_integration_on_device`, with `atol`/`rtol` as
floats: construction 0.296 → 0.089 ms; one-shot 64 → 49 waits; `state=` call
61 → 51 waits, 4.93 → 4.72 ms; 24 compared cases identical. The design below
keys the cache by the registry OBJECT, which the prototype did not run; the
implementation is measured in Section 7.

## 3. The design

### 3.1 Method tables: converted once, shared, never changed in place

**Change** (`padaquad/methods/_base.py`, `padaquad/methods/__init__.py`).
Replace `MethodClass.clone`, the abstract `to_dtype`/`to_device`, their four
implementations and `_Tableau.to_dtype`/`to_device` by one non-changing
conversion, memoized per process:

```python
# padaquad/methods/_base.py
_CONVERTED: dict[tuple[MethodClass, torch.device, torch.dtype], MethodClass] = {}

class MethodClass(ABC):
    def to(self, device: str | torch.device, dtype: torch.dtype) -> Self:
        """This method with copies of its tensors on device in dtype; built once, shared."""
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        key = (self, device, dtype)
        if key not in _CONVERTED:
            _CONVERTED.setdefault(key, self._map_tensors(lambda t: t.to(device).to(dtype, copy=True)))
        return _CONVERTED[key]

    @abstractmethod
    def _map_tensors(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> Self:
        """A shallow copy with fn applied to each tensor."""
```

`UniformMethodClass._map_tensors` returns `copy.copy(self)` with `tableau =
_Tableau(fn(c), fn(b), fn(b_error))`; `VariableMethodClass._map_tensors`
returns `copy.copy(self)` with `c = fn(self.c)` (the module-level
`weight_function` is shared, as `clone` shared it). `_Tableau` becomes
`@dataclass(frozen=True)`. `_get_method` returns
`registry[name].to(device, dtype)`. `_integration_on_device` sets
`self.method = self.method.to(device, self.dtype)` and restores the original
object instead of calling `to_device` twice.

**Order of conversion.** `t.to(device).to(dtype)` keeps today's order (the
cast runs on the target device), so the tables have the same bits; the grid
harness's float32 cases on CUDA check it.

**Keys and lifetime.** The key is the source object itself (identity hash;
`MethodClass` defines no `__eq__`), not the name: a user can replace a
registry entry (`docs/development/adding_a_method.md`), and
`_integration_on_device` must convert the solver's own method. Converting a
converted method (`_integration_on_device`) adds a second level keyed by the
converted object. The dictionary keeps every source and converted method alive
for the life of the process: (methods used) × (devices) × (dtypes) × 2 small
objects, a few kilobytes each; a user who builds many methods (for example a
new `UniformMethodClass` per call) keeps all of them, with their tables on
every device used. Two threads may build the same entry at once;
`setdefault` makes both return the one stored. `copy.copy`,
`copy.deepcopy` and pickling of a method or a solver copy the converted
method; a copy is a new object, not in the dictionary, and its own
conversions are new entries. A dtype change of an existing solver
(`set_dtype_by_input` → `_set_dtype` → `_set_solver_dtype`) gets
`_get_method(...)` of the registry entry under the solver's method name, as
today. `adaptive_heun` is one object in both registries, so both solver types
share its converted copies.

**In-place writes.** Shared objects turn a write into a solver's method
(`solver.method.tableau.b.mul_(2)`) from a change of one solver into a change
of every solver of the same method, device and dtype in the process.
PyTorch has no read-only tensors (inference-mode tensors refuse in-place
writes but cannot be saved for backward). Bounded by: copies never sharing
memory with the registry (the registry is never changed through a solver);
`weights()` already documents that its views must not be modified; a test
that no integration writes into a method's tensors (version counters); a
sentence in `docs/development/architecture.md` and
`docs/development/adding_a_method.md`.

**Alternatives rejected.** *No cache, each solver converts*: saves the deep
copy, keeps three copies to the GPU per construction and six per call in
`_integration_on_device`. *Cache keyed by name* (the prototype): goes stale
when a user replaces a registry entry. *Keep `clone`/`to_dtype`/`to_device`
too* (D1 b): two ways to do one thing, and the in-place conversion stays
available.

### 3.2 Tolerances as Python floats (D2, its own commit)

`SolverBase.__init__` stores `self.atol = float(atol)`, `self.rtol =
float(rtol)` and checks that both are finite and ≥ 0 (`ValueError`; today a
negative or NaN tolerance is accepted and makes every error ratio negative or
NaN, so every subinterval is accepted; zero stays allowed and is used, for
example `rtol=0`). `_set_dtype` drops the two conversions (360-361);
`_error_ratios_norm` and `_error_ratios_failure_fraction` drop the moves
(3880-3881, 3971-3972) and use the floats, with `torch.maximum(atol, x)`
written as `torch.clamp(x, min=self.atol)` (3890, 3899; `torch.maximum` needs
two tensors).

**Same bits, and where not.** A Python float with a float32 tensor gives a
float32 result, as a 0-dimensional tensor does (PyTorch documentation,
"Type promotion" in *Tensor Attributes*,
https://docs.pytorch.org/docs/stable/tensor_attributes.html#type-promotion-doc),
and the scalar is rounded to the tensor's dtype before the operation, so with
values in the solver's dtype the results are the same bits (checked,
`impl/tol_bits_cs.py`, CPU and CUDA). With float64 values and a float32
solver (an integrand returning float64), the old tolerances were the float32
roundings of the given values and the new ones are the given values (Review,
item 2). `torch.clamp` and `torch.maximum` give zeros of different sign only
when `atol = 0` meets −0, which the norms of absolute values never produce;
NaN results may differ in their sign bit only, which nothing reads.

**Visible differences.** `solver.atol` and `solver.rtol` are floats, not
tensors (`float(solver.atol)` keeps working; `solver.atol.item()` breaks). A
negative or non-finite tolerance raises `ValueError`.

### 3.3 A mesh given without bounds is not cut

`_setup_integral_bounds` (3364-3506): when a mesh is given and neither bound
is, the bounds are the mesh's ends and the mesh is returned after its
validation and the order check, without the cut (3486-3505). Same values;
the returned mesh is the given tensor (moved to the solver's dtype and
device) instead of a concatenated copy. Nothing in `integrate` writes into it
(checked: the loop replaces `mesh` by `_adaptively_increase_mesh`'s new
tensors). For a given mesh that requires grad, the gradient now reaches it
directly instead of through the concatenation; the values are the same, and a
zero gradient may change sign (−0 + 0 = +0 in the old sum). With given bounds
the function is unchanged.

### 3.4 Starting mesh and the start of the loop

- Asserts (3606-3611, gated by `runtime_checks`): for T = 1 only the
  strictly-increasing assert (it implies the distinct-points one); for T > 1
  only the distinct-points one.
- Trackers (3612-3613): `mesh_trackers = torch.arange(len(mesh),
  device=self.device) < len(mesh) - 1`, the same boolean tensor without a copy
  from the host. Checked with `torch.cuda.set_sync_debug_mode("warn")`.
- 1177: `n_start_steps * self.C` instead of `int(mesh_trackers.sum()) *
  self.C` (the same number).
- The default `y0` (`base.py:203`): `torch.zeros(1, dtype=dtype,
  device=self.device)`, a fill on the device instead of a copy from the host.
- 3553-3555: `math.isqrt(int(N_init_steps))` (D4, its own commit).
- The random mesh arithmetic itself is not changed: any rewrite changes the
  starting mesh's rounding and so every result.

### 3.5 After the loop

- One read for the split statistics and the final check (1561-1585):
  `n_unsplit, max_split_depth, failed = torch.stack([(c == 0).sum(), c.max(),
  final_fails.any()]).tolist()` with `c = split_counts[:-1]` (the boolean
  converted to int64 by `stack`'s type promotion, checked). 3 → 1 waits; the
  same integers.
- 1612-1639 (moves into `_get_optimal_mesh`, `mesh_optimal`, the record) are
  left to `optimal-mesh-cost` and `record-order`.

## 4. Decisions (taken by the user, 2026-11-25)

- **D1 (a):** replace `clone`, `to_dtype` and `to_device` by the memoized
  `to(device, dtype)` and freeze `_Tableau`. `MethodClass` is importable from
  `padaquad.methods`, so code calling the removed methods breaks (CHANGELOG
  entry).
- **D2 (a):** `solver.atol` / `solver.rtol` as Python floats; a separate
  commit after D1, droppable without breaking D1.
- **D3 (a):** no reuse of solvers in `padaquad.integrate` (a reused solver
  would share per-call state between threads and nested integrals, and its
  options include tensors and callables, which are not reliable keys).
- **D4:** `math.isqrt` in its own commit (as `math.isqrt(int(N_init_steps))`,
  Review item 1). Differs for `N_init_steps` ≥ 9,443,328.
- **D5:** distributed construction is out of scope (Section 9).
- **D6:** done on `main` by pull request #74; removed.

## 5. Implementation steps (one commit each)

1. **Method conversion** (`padaquad/methods/_base.py`: `_Tableau`,
   `MethodClass.to`, `_map_tensors` and its two implementations, removal of
   `clone`, `to_dtype`, `to_device`; `padaquad/methods/__init__.py`:
   `_get_method` and the module docstring; `padaquad/quadrature/base.py`:
   `_set_solver_dtype` docstring, `_integration_on_device`).
   Tests: `tests/test_solver_isolation.py` rewritten (Section 6);
   `tests/test_base_and_methods.py::TestTableau`,
   `tests/test_methods_extra.py::TestTableauToDevice` and
   `::TestVariableMethodDtype` rewritten for `to`;
   `tests/test_methods_registry.py::test_get_method_returns_independent_clone`
   adapted; `tests/test_base_solver.py` loses `_save_tableau`/`_restore_tableau`
   (nothing can change the registry through a solver any more, and the restore
   would raise on the frozen tableau). Docs: CLAUDE.md "Dtype management",
   the test table, `docs/development/architecture.md`,
   `docs/development/adding_a_method.md`, `docs/reference/methods.md`,
   CHANGELOG (Removed and Changed).
2. **Tolerances as floats (D2)** (`padaquad/base.py`: `__init__`,
   `_set_dtype`; `padaquad/quadrature/base.py`: `_error_ratios_norm`,
   `_error_ratios_failure_fraction`). Tests: `tests/test_error_masks.py` gets a
   test that both helpers give the same bits as the formulas with
   0-dimensional tolerance tensors of the values' dtype (float32 and float64,
   CPU and CUDA, NaN, ±Inf, 0, subnormal values and tolerances); a test of
   the new `ValueError`s; the snapshot tests. CHANGELOG.
3. **Host waits of the starting mesh and the statistics**
   (`padaquad/base.py:203` the default `y0`; `quadrature/base.py`:
   `_setup_initial_mesh` asserts and trackers, `integrate` 1177 and
   1561-1585). Tests: the snapshot tests and `tests/test_split_statistics.py`
   unchanged; a CUDA test that `_setup_initial_mesh` and the construction make
   no host wait apart from the asserts and `_get_mesh_indices`.
4. **A mesh given without bounds is not cut** (`_setup_integral_bounds`).
   Tests: `tests/test_bug_regressions.py`'s interval rules unchanged; a test
   that `integrate(f, state=state)` starts from the state's mesh object, and
   that a mesh requiring grad still gets its gradient.
5. **D4** (`_setup_initial_mesh`). Test: the starting mesh has
   `isqrt(N) * (isqrt(N) + 1)` points for several `N`, including 13 (the
   default), a perfect square, `N = 3073² − 1` and a float.
6. **Measure** (Section 7) and record the results here.

## 6. Tests, and showing that results do not change

- **`tests/test_solver_isolation.py`.** Keep: two solvers at different dtypes
  stay independent; the registry entry is unchanged after building solvers at
  every dtype. New: `to` returns the same object for the same key and another
  for another device or dtype; `to` leaves its source unchanged and shares no
  memory with it; assigning a `_Tableau` field raises
  `dataclasses.FrozenInstanceError`; a replaced registry entry reaches new
  solvers while an existing solver keeps its own (also through
  `_integration_on_device`); no integration writes into a method's tensors
  (version counters); on CUDA, `integrate(result_device="cpu")` leaves the
  solver's method the same CUDA object.
- **Unchanged results.** (1) `tests/test_snapshots.py`. (2) The grid
  `impl/bitgrid_cs.py` (extends `record-order/bitgrid.py`): CPU and CUDA;
  float32 and float64; `result_device="cpu"` with a CUDA solver; the dtype
  switch by float32 bounds; second calls from the state with no bounds, equal
  bounds, bounds inside, outside and within rounding; a given mesh with
  `take_gradient=True`; failure_fraction with D > 1; `rtol=0`; one-shot calls
  with several `N_init_steps`; a float32 solver with a float64 integrand
  (expected to differ under D2 only). Compared with `impl/bitcompare_cs.py`
  against `d11df47`.

## 7. How to measure the gain

`impl/measure_cs.py {cuda|cpu}`: construction, one-shot `padaquad.integrate`
(default `result_device="cpu"`, and on the solver's device), a `state=` call
without bounds and with bounds; wall time (median and minimum of 41), host
waits per line, GPU launches, inclusive time of `_setup_integral_bounds`,
`_setup_initial_mesh`, `_get_optimal_mesh` and `_set_solver_dtype`. And
`/pscratch/sd/k/khegazy/padaquad_speedup_profile/profile_baseline.py` for the
four integrands. Before (`impl/base_d11df47`, `git archive d11df47`) and after,
on the same node and GPU in one session.

## 8. Documentation to update

- `CLAUDE.md`: "Dtype management" (methods converted once per process and
  shared, never changed in place); the merge weight cache line; the test
  table rows of `test_solver_isolation.py`, `test_base_and_methods.py`,
  `test_methods_extra.py`.
- `docs/development/architecture.md` (Methods), `docs/development/adding_a_method.md`
  (own copy), `docs/reference/methods.md` (own copy; `clone` in the members).
- `CHANGELOG.md` (the unreleased 2.0.0 section): Removed `MethodClass.clone`,
  `to_dtype`, `to_device`; Changed: faster construction and fewer host waits
  per call, `solver.atol`/`rtol` are floats (D2), the tolerance check (D2),
  the starting mesh's count (D4).
- Docstrings: `_get_method`, `MethodClass`, `UniformMethodClass`,
  `VariableMethodClass`, `_Tableau`, `_set_solver_dtype`,
  `_integration_on_device`, `SolverBase._set_dtype`,
  `_setup_integral_bounds`.

## 9. Risks and interactions

- **Shared objects** (Section 3.1, "In-place writes").
- **Distributed runs (found, not in this plan).** Under `torchrun` (`RANK`
  set) every construction calls `dist.init_process_group`
  (`distributed.py:316`) and every destruction `destroy_process_group`
  (`SolverBase.__del__`, `base.py:500`; `distributed.py:350`). Checked with
  gloo in one process: a second solver built while the first exists raises
  `ValueError: trying to initialize the default process group twice!`, and
  deleting a solver destroys the group, also one the user's program created.
  Recommend a separate issue: use an existing group and destroy only one the
  solver created.
- **`optimal-mesh-cost`**: owns the merge weight cache and the moves around
  `_get_optimal_mesh`. Interface given here: `_get_method(sampling, name,
  device, dtype)` and `method.to(device, dtype)` return the same object for
  the same source, device and dtype, never changed in place. If that plan
  removes `_integration_on_device`, its swap goes with it.
- **`startup-evaluations`**: owns the shape-check call (1118) and
  `_initialize_max_batch` (1173); this plan only replaces the argument at
  1177.
- **`record-order`**: owns `_get_mesh_indices` (3614), `_sort_record` and the
  record moves.
- **`turn-device-reads`**: its D1 (what `runtime_checks` covers) decides
  whether the starting-mesh assert stays gated.
- **`error-ratio-work`**: edits the same two error-ratio helpers; merge
  conflicts only.
- **`compiled-turn`**: Python-float tolerances are constants to a compiler.
- **`endpoint-evaluations`** (being implemented at the same time): changes
  the Gauss-Kronrod tables, `_check_node_fractions`, `_RK_integral`,
  `_flatten_output`, the merge prediction and `_merge_excess_nodes`. This plan
  touches `methods/_base.py` only in `_Tableau`, the conversion methods and
  `clone`, and `quadrature/base.py` only in the functions named in Section 5.

## 10. Confidence

- **Sure (measured or read):** the call sites and host-wait sources above;
  D4's first difference on this CPU; the tolerance arithmetic with values in
  the solver's dtype; the distributed behaviour in one gloo process.
- **Less sure:** wall-time gains (shared GPU nodes, spread about as large as
  single items); whether CUDA's float32 square root and casts agree with the
  CPU's (the grid compares each device with itself only).

## Implemented (2026-11-26, branch `call_setup_cost`)

| Step | Commit | What |
|---|---|---|
| Plan review | `be1bf80` | This file, reviewed and corrected |
| 1. Method conversion (D1) | `314db3c` | `MethodClass.to`, `_map_tensors`, `_CONVERTED`, frozen `_Tableau`; `clone`, `to_dtype`, `to_device` removed; `_get_method` and `_integration_on_device` use `to`; tests and docs |
| 2. Tolerances as floats (D2) | `787e232` | `SolverBase.__init__` (floats, `ValueError`), `_set_dtype`, both error-ratio helpers (`torch.clamp`); tests in `test_error_masks.py` and `test_base_solver.py` |
| 3. Host waits of the start and end of a call | `8f40aa6`, `7350a3f` | default `y0`, trackers, the fresh-mesh check, `n_start_steps * self.C`, one read for the split statistics and the check; CUDA host-wait tests (`7350a3f` fixes their count) |
| 4. A mesh without bounds is not cut | `58d3411` | `_setup_integral_bounds`; test `test_a_mesh_alone_is_used_as_given` |
| 5. D4 | `d58a45e` | `math.isqrt(int(N_init_steps))`; test `test_starting_mesh_has_isqrt_segments` |
| 6. Measurements, CHANGELOG, this section | the last commit | |

Dropping D2 alone (`git revert 787e232`) leaves D1 intact; it needs two
small edits: the CHANGELOG lines next to its entry, and the expected count of
`TestHostWaits.test_construction_waits_for_nothing` (2 instead of 0: the
tolerance tensors).

**Measured** on `nid200417` (A100-SXM4-80GB, GPU 0; `impl/measure_cs.py`,
median of 41 calls, two rounds alternating `d11df47` and the branch on the
same GPU; `impl/logs/measure_*`). `damped_sine` on [0, 1], gk21, float64,
`atol=1e-8`, `rtol=0`, `error_norm="max"`.

| Case | `d11df47`: ms (host waits, GPU launches) | branch |
|---|---|---|
| Build a solver | 0.320, 0.325 (6, 6) | 0.076, 0.076 (0, 1) |
| One-shot `padaquad.integrate` (`result_device="cpu"`) | 6.70, 6.69 (61, 267) | 6.06, 6.08 (41, 248) |
| One-shot, `result_device` on the GPU | 9.46, 9.26 (63, 508) | 8.84, 8.86 (53, 499) |
| `state=`, no bounds | 5.79, 5.78 (53, 252) | 5.27, 5.27 (39, 224) |
| `state=` with the bounds equal to the mesh ends | 5.91, 5.91 (57, 260) | 5.69, 5.66 (44, 250) |

`_setup_integral_bounds` (inclusive, synchronized): 0.43 → 0.16 ms in the
`state=` call without bounds; unchanged (0.56 ms, 5 host waits) with bounds.
CPU of the same node (8 threads): building 0.198 → 0.043 ms; one-shot 3.32 →
3.15 ms; `state=` 2.60 → 2.53 ms; `state=` with bounds 2.60 → 2.63 ms (no
change within noise). `profile_baseline.py` (median of 5, two rounds):
`damped_sine` 6.2/6.1 → 5.5/5.6 ms, `many_outputs` 13.7/14.8 → 14.3/13.0 ms
(noise as large as the change), `chirp` 69.3/69.2 → 68.7/68.4 ms,
`oscillatory` 226/227 → 220/214 ms; 20 fewer host waits in each.

**Checks** (same node; `impl/logs/`). Full suite, `d11df47` → branch: CPU
3968 passed, 315 skipped → 3997 passed, 337 skipped, no failure on either;
CUDA 4258 passed, 25 skipped → 4308 passed, 25 skipped, 1 failed
(`TestHostWaits.test_construction_waits_for_nothing`, which counted PyTorch's
notice that the sync debug mode is a prototype; fixed in `7350a3f`, after
which its file and the other changed test files pass on CUDA: 293 passed, 2
skipped). Bit-identity grid (`impl/bitgrid_cs_parts.py`, 1,288 integrations
per device, 16,096 tensors): CPU and CUDA each differ only in `error_ratios`
of the two cases with a float64 integrand on a float32 solver (D2; at most
4.8e-11 for gk21 and 2.5e-8 for `interpolatory3_variable`, both relative to
ratios of order 1; integral, error, mesh, nodes and values identical). The
grid's variable-solver `adaptive_heun` part did not finish within the
allocation and is not compared. Tolerance arithmetic (`impl/tol_bits_cs.py`):
2,700 same-dtype cases identical on the CPU and on CUDA. Docs build with
`-W`: passes. `ruff`, `mypy`, `numpydoc lint`: clean.

## Not done

- **The combined read for given bounds** (old Section 3.3): a call with
  bounds still spends 0.56 ms and 5 host waits in `_setup_integral_bounds` on
  the GPU (two copies of the CPU bounds, the order check, two `torch.equal`
  and the boolean index). Combining them would save perhaps 0.2-0.3 ms there
  but rewrites the validation of user input; left for a decision.
- **The merge weight cache on the method object** (Review, item 8).
- **D5** (distributed construction): a separate issue.
