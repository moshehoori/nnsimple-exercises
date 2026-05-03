# NNSimple MLIR Exercises — 2-day curriculum

A hands-on introduction to MLIR compiler development for a team of 12, built on top of the `nnsimple` dialect. Learn by doing — no grading, no lectures beyond a kickoff.

## Schedule

| When | What | Who |
|---|---|---|
| Pre-event | Laptop setup: build MLIR, build nnsimple, run `ninja check-nnsimple` green | Everyone, solo |
| Day 1 AM (~3h) | Foundation exercises F-01, F-02 | Everyone, solo |
| Day 1 lunch | Show-and-tell: 2-3 people walk their diff | Everyone |
| Day 1 PM (~4h) | Form groups of 3, each group picks a track, start exercise #1 | 4 groups |
| Day 2 AM (~4h) | Finish exercise #1, start exercise #2 | Groups |
| Day 2 PM (~3h) | Finish exercise #2 | Groups |
| Day 2 end (~1h) | Each group does a 5-min demo of what they built | Everyone |

## How the repo is organized

- **`master`** — pristine nnsimple snapshot plus these curriculum docs. Never committed to during the exercise. This is your reference.
- **`starter/<id>`** — one branch per exercise. Each contains a failing FileCheck test and an `EXERCISE.md` describing the task. Check out the branch, make it pass, you're done.
- **`reference/<id>`** — (optional) completed solutions. Don't look unless stuck.

## How to do an exercise

```bash
# Pick the exercise you want
git checkout starter/F-01

# Read the task
cat EXERCISE.md

# See the failing test (look for the exercise's own test failing)
cd build && ninja check-nnsimple

# Edit files. When done, the exercise's test goes green:
ninja check-nnsimple

# Optional: save your work on a personal branch before moving on
git checkout -b $USER/F-01
git commit -am "F-01 done"
```

Run a single exercise's test only (faster feedback loop):
```bash
llvm-lit -v ../test/NNSimple/sub-ops.mlir
```

### A note on baseline failures

`ninja check-nnsimple` has **7 pre-existing failures** (same on `master`, unrelated to exercises) caused by API drift between upstream nnsimple and the pinned MLIR build:
- `CAPI/nnsimple-capi-test.c` — CAPI binary not built in this configuration
- `NNSimple/add-canonicalize.mlir`, `basic_ops.mlir`, `dummy.mlir`, `nnsimple-fuse-test.mlir` — upstream printer drops `!nnsimple.tensor` prefix inside `(...)`
- `NNSimple/nnsimple-plugin.mlir`, `nnsimple-pass-plugin.mlir` — plugin only builds with in-tree MLIR

Your goal for each exercise is to turn **your exercise's test** green (e.g. `sub-ops.mlir` for F-01). Ignore the 7 baseline reds. You can filter to just your test:
```bash
llvm-lit -v ../test/NNSimple/sub-ops.mlir   # F-01
llvm-lit -v ../test/NNSimple/neg-canonicalize.mlir   # F-02
# ... etc.
```

## Exercises

### Foundation — everyone, solo (~3h)

| ID | Task | Concepts | Files |
|---|---|---|---|
| F-01 | Add `nnsimple.sub` op with verifier | TableGen op def, C++ verifier, Pure trait | `include/NNSimple/NNSimpleOps.td`, `lib/NNSimple/NNSimpleOps.cpp`, `test/NNSimple/sub-ops.mlir` |
| F-02 | Add `nnsimple.neg` op + DRR canonicalization `neg(neg(x))→x` | Declarative rewrite rules, canonicalization | same three files |

### Tracks — groups of 3 pick one (~7h over day 1 PM + day 2)

**Track A — Ops & Type System**
| ID | Task | Mirror |
|---|---|---|
| A-01 | `nnsimple.matmul` op with shape-compatibility verifier | `AddOp::verify` in `NNSimpleOps.cpp:35-47` |
| A-02 | `!nnsimple.quantized<elementType, shape, scale, zero_point>` type | `TensorType` in `NNSimpleTypes.td:48-67` |

**Track B — Folders & Canonicalization**
| ID | Task | Mirror |
|---|---|---|
| B-01 | `MulOp::fold` for constant × constant | `AddOp::fold` in `NNSimpleOps.cpp:50-69` |
| B-02 | C++ canonicalizations `mul(x,1)→x` and `mul(x,0)→0` | `AddZeroElimination` in `NNSimpleOps.cpp:72-98` |

**Track C — Transformation Passes**
| ID | Task | Mirror |
|---|---|---|
| C-01 | `-nnsimple-fuse-mul-add` pass producing a new `FusedMulAddOp` | `NNSimpleFuseAddReluRewriter` in `NNSimplePasses.cpp:24-42` |
| C-02 | `-nnsimple-dce` dead-op elimination using the `Pure` trait | Same file; `op->hasTrait<OpTrait::IsPure>()` + `op->use_empty()` |

**Track D — Lowering & E2E JIT**
| ID | Task | Mirror |
|---|---|---|
| D-01 | Lower `nnsimple.sub` (from F-01) to `linalg.sub` | `AddOpLowering` in `NNSimpleLinalgLowering.cpp:59-76` |
| D-02 | E2E JIT: run `add+relu` end-to-end with `mlir-cpu-runner`, CHECK output | upstream `mlir/test/Integration/Dialect/Linalg/CPU/` examples |

## Build quick-reference

```bash
mkdir -p build && cd build
cmake -G Ninja .. \
  -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DMLIR_INCLUDE_TESTS=ON
ninja nnsimple-opt
ninja check-nnsimple
```

## Tips

- TableGen changes only rebuild if CMake re-runs after adding files — `ninja` picks up `.td` edits automatically.
- `nnsimple-opt -help` lists all registered passes.
- `nnsimple-opt -debug` prints pattern application traces — useful when a rewrite doesn't fire.
- FileCheck docs: https://llvm.org/docs/CommandGuide/FileCheck.html
- MLIR op definition spec: https://mlir.llvm.org/docs/DefiningDialects/Operations/
