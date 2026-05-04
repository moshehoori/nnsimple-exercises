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

The test for this exercise (`test/NNSimple/dce.mlir`) is already in the repo — don't edit it. Just make it pass.

The pass is already declared in `NNSimplePasses.td` and wired into CMake — auto-registered with `nnsimple-opt`.

## Hints

- The pass's target op comes from `getOperation()` — for this pass that's a `func::FuncOp`.
- Read [Understanding the IR Structure — Operation Walkers](https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/#operation-walkers) for how to traverse the IR with `op.walk(...)`.
- To query the `Pure` trait, `#include "mlir/Interfaces/SideEffectInterfaces.h"` and call the free function `mlir::isPure(op)`.
- To check whether an op's result is consumed: `op->use_empty()`.
- Two gotchas to think through yourself:
  - If you call `op->erase()` from inside `walk()`, things break. Why? How would you work around it?
  - `func.return` satisfies `use_empty()` trivially (it has no results). You don't want to erase terminators. What extra check keeps it alive?
- **Fixed-point loop**: the third test case (`drop_dead_chain`) has a `relu` whose operand `add` is also dead. A single DCE pass erases the `relu`, but at that point the `add` still existed when you collected dead ops, so it survives that iteration. Loop until nothing changes.

## Done when

`test/NNSimple/dce.mlir` goes from red to green. It has 3 cases: simple dead op, used op preserved, chain of dead ops (requires the fixed-point loop).

```bash
cd build && ninja nnsimple-opt
llvm-lit -v ../test/NNSimple/dce.mlir   # should PASS
```

Or manually:
```bash
./bin/nnsimple-opt ../test/NNSimple/dce.mlir -nnsimple-dce \
    | /path/to/llvm-project/build/bin/FileCheck ../test/NNSimple/dce.mlir
```

## Stretch

- Add a pass option `--preserve-ops=nnsimple.const` that skips ops of named kinds even when dead.
- Compare your implementation to MLIR's built-in `-canonicalize` (which does DCE as part of its work). What does `-canonicalize` do differently?
- Why didn't we use `OpRewritePattern` for this? (Hint: erasing an op isn't a *rewrite* — it's a deletion. Greedy driver can do both, but a hand-written walk makes the intent clearer for DCE.)
