# C-02 — `-nnsimple-dce` dead-op elimination pass

**Concepts**: op traits (`Pure`), `op.use_empty()`, IR walks, safe erasure during traversal.

**Time**: ~2h.

## Task

Write a dead-code-elimination pass that walks a function and erases every op that is **both**:
- `Pure` (no side effects) — side-effecting ops cannot be safely removed even if unused.
- Uses-empty (`op->use_empty()`) — nobody reads its result.

```mlir
// Before -nnsimple-dce:
func.func @f(%a: !nnsimple.tensor<f32, [4], NCHW>, %b: ...) {
    %0 = nnsimple.add %a, %b : ... // dead — %0 unused
    return
}
// After: only `return` remains.
```

Don't touch non-`Pure` ops even if their result is unused (e.g. `func.call` to a function with side effects must not be removed).

## Files to edit

| File | What |
|---|---|
| `lib/NNSimple/NNSimpleDCE.cpp` | Implement `runOnOperation`. Starter skeleton provided. |

The pass is already declared in `NNSimplePasses.td` and wired into CMake — auto-registered with `nnsimple-opt`.

## Hints

- Walking the IR:
  ```cpp
  // #include "mlir/Interfaces/SideEffectInterfaces.h"  — for mlir::isPure
  SmallVector<Operation *> toErase;
  func.walk([&](Operation *op) {
    // mlir::isPure(op) is the free-function way to query the Pure trait.
    // Also guard against ops with no results (like func.return) — they
    // satisfy use_empty() trivially but you don't want to erase terminators.
    if (mlir::isPure(op) && op->use_empty() && op->getNumResults() > 0)
      toErase.push_back(op);
  });
  for (Operation *op : toErase) op->erase();
  ```
- **Collect-then-erase, in a fixed-point loop**: `walk()` misbehaves if you erase during traversal. Collect first, erase after. And because erasing one op can leave its operands dead, you need a fixed-point loop (the test's third case — `drop_dead_chain` — has a `relu` whose operand `add` is dead; one pass erases the `relu`, the second erases the `add`):
  ```cpp
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> toErase;
    func.walk([&](Operation *op) {
      if (mlir::isPure(op) && op->use_empty() && op->getNumResults() > 0)
        toErase.push_back(op);
    });
    for (Operation *op : toErase) { op->erase(); changed = true; }
  }
  ```
- `func::FuncOp` (the pass's target op) comes from `getOperation()`.

## Done when

```bash
cd build && ninja check-nnsimple
```

passes. `test/NNSimple/dce.mlir` has 3 cases: simple dead op, used op preserved, chain of dead ops (requires the loop).

## Stretch

- Add a pass option `--preserve-ops=nnsimple.const` that skips ops of named kinds even when dead.
- Compare your implementation to MLIR's built-in `-canonicalize` (which does DCE as part of its work). What does `-canonicalize` do differently?
- Why didn't we use `OpRewritePattern` for this? (Hint: erasing an op isn't a *rewrite* — it's a deletion. Greedy driver can do both, but a hand-written walk makes the intent clearer for DCE.)
