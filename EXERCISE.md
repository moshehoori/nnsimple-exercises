# F-02 — Add `nnsimple.neg` with a declarative canonicalization

**Concepts**: TableGen op definition, declarative rewrite rules (DRR), canonicalization.

**Time**: ~1.5h.

## Task

Add an element-wise negation op, then write a declarative rewrite rule that collapses double-negation: `neg(neg(x)) → x`.

```mlir
// Before -canonicalize:
%0 = nnsimple.neg %x : ...
%1 = nnsimple.neg %0 : ...
// After -canonicalize:
// (nothing — both negs are eliminated; uses of %1 are redirected to %x)
```

## Files to edit

| File | What to do |
|---|---|
| `include/NNSimple/NNSimpleOps.td` | Define `NNSimple_NegOp` and a `Pat<>` named `NegNegElimination`. Stubs show where. |
| `lib/NNSimple/NNSimpleOps.cpp` | Register the generated pattern with `NegOp::getCanonicalizationPatterns`. Stub shows where. |

## Hints

- Look at `NNSimple_ReluOp` in `NNSimpleOps.td` (lines ~50-64) — your op is structurally identical: single operand, single result, Pure, `hasCanonicalizer = 1`.
- Look at `ReluReluElimination` in the same file (the `Pat<>` block near the bottom). Your pattern's RHS uses the DRR builtin `replaceWithValue` to replace the outer op with the inner op's original input:
  ```tablegen
  def NegNegElimination : Pat<
    (NNSimple_NegOp (NNSimple_NegOp $input)),
    (replaceWithValue $input)>;
  ```
  Without `replaceWithValue`, you'd create a new op; this builtin just reuses the SSA value.
- The generated include `NNSimple/NNSimpleCanonicalization.inc` is already pulled into `NNSimpleOps.cpp` — the pattern class name `NegNegElimination` will exist as a C++ symbol once the `.td` pattern is defined.
- Look at `ReluOp::getCanonicalizationPatterns` (near the bottom of `NNSimpleOps.cpp`) — yours is one-liner identical.

## Done when

```bash
cd build && ninja check-nnsimple
```

passes. The test file is `test/NNSimple/neg-canonicalize.mlir` — a positive test (`neg(neg(x))` is gone after `-canonicalize`) and a sanity test (a single `neg` survives).

Single-file:
```bash
llvm-lit -v ../test/NNSimple/neg-canonicalize.mlir
```

## Stretch (optional)

- Add a folder: `neg(const)` folds to a new constant with all elements negated. Mirror `AddOp::fold`.
- Add a DRR for `neg(sub(a, b)) → sub(b, a)` (requires F-01's `sub` op merged in).
