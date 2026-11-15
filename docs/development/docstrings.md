# Writing docstrings

padaquad's docstrings follow the [numpydoc](https://numpydoc.readthedocs.io/)
format and these rules. The documentation site is built from them, so they are
written for users first.

1. **One word per concept.** Use the words of the {doc}`glossary <../glossary>`:
   *subinterval* (never "step" or "panel" for a piece of the domain), *mesh*
   (never "barrier"), *node*, *method* (never "rule"), *weights*, *degree*,
   *order*. Call `t` "the integration variable `t`", never "time". Say
   "Runge-Kutta" only for the four Runge-Kutta methods. Code names such as
   `mesh_trackers` or `max_split_depth` keep their names; this rule is about
   prose.
2. **The summary line is one sentence** saying what the object does.
3. **Each parameter gets at most three lines**: what it controls, its shape or
   units, and its default behavior. A longer explanation goes in the user guide,
   and the docstring links to it with ``:ref:`label` ``. The labels are listed
   below.
4. **Public docstrings do not mention** bug IDs ("B4"), "Phase N",
   "historical", names of private functions or attributes (anything starting
   with `_`), or removed features. Private docstrings may name private code.
5. **Formulas** use ``:math:`...` `` or double-backtick literals. Bare `|x|`,
   `*args` and `_name_` in prose break the documentation build.
6. **Types** come from the docstring (`name : type`), because the site does not
   print annotations. Write `torch.Tensor or None` rather than
   `torch.Tensor | None` in public docstrings.
7. **Citations** use ``:cite:p:`key` `` with a key of `docs/references.bib`.
   A source is added there only after it has been opened and the cited number,
   table or equation found in it. A formula that is padaquad's own is labeled
   "(padaquad's own derivation)" and derived where it is used.
8. **Examples** sections run as doctests. They pass `device="cpu"`, an explicit
   `max_batch` (this skips the memory benchmark) and float64 bounds, print
   rounded Python floats rather than tensors, and never print a mesh size,
   because the starting mesh is random.

## User-guide labels for links from docstrings

| Label | Page and section |
|---|---|
| `ug-shapes` | Concepts: the integrand contract and the shape letters |
| `ug-error-control` | Error control: tolerances, error ratios and the rounding floor |
| `ug-error-norm` | Error control: vector-valued integrands (`error_norm`, `mesh_failure_tolerance`) |
| `ug-error-reference` | Error control: absolute and cumulative tolerance, `error_integral_reference` |
| `ug-mesh` | How it works: the starting mesh (`N_init_steps`, `random_initial_mesh`) |
| `ug-optimal-mesh` | How it works: the optimal mesh and the merge rule (`remove_cut`) |
| `ug-same-f` | Repeated calls: warm starts with `same_f` |
| `ug-autodiff` | Gradients: `take_gradient` and `loss_fxn` |
| `ug-memory` | Devices and memory: the memory benchmark, `max_batch`, `get_max_batch`, `total_mem_usage`, `conserve_memory` |
| `ug-convergence` | Troubleshooting: `converged`, `max_adaptive_splits` and the split statistics |
| `ug-tracked-variables` | Repeated calls: tracked variables |
