# NNSimple MLIR Exercises — 2-day curriculum

A hands-on introduction to MLIR compiler development for a team of 12, built on top of the `nnsimple` dialect. Learn by doing — no grading, no lectures beyond a kickoff.

**For the detailed per-exercise spec (task, files, hints, done-criteria) see [EXERCISES.md](EXERCISES.md).**

## Format

Everyone does **all 10 exercises individually**. No groups, no tracks — every learner works through the same sequence.

The 10 exercises are ordered roughly from easiest to hardest:

```
Foundation   →   Ops & Types   →   Folders/Canon   →   Passes       →   Lowering
 F-01, F-02       A-01, A-02        B-01, B-02         C-01, C-02       D-01, D-02
```

## Schedule

| When | What |
|---|---|
| Pre-event | Laptop setup: build MLIR, build nnsimple, run `ninja check-nnsimple` green |
| Day 1 AM (~3h) | F-01, F-02 (foundation) |
| Day 1 PM (~4h) | A-01, A-02, B-01 (ops/types, first folder) |
| Day 2 AM (~4h) | B-02, C-01, C-02 (canonicalization, passes) |
| Day 2 PM (~3h) | D-01, D-02 (lowering, E2E) |
| Day 2 end (~1h) | Show-and-tell: 2-3 people walk their favorite diff |

**Heads up on pacing**: 10 exercises in 2 days is tight for some learners (especially if it's your first contact with MLIR). That's intentional — the goal is exposure to every major concept, not polish. If you don't finish D-02, that's fine; come back to it later. If you fly through, the `## Stretch` sections in each `EXERCISE.md` have deeper follow-ons.

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

Each exercise lives on its own `starter/<id>` branch. See [EXERCISES.md](EXERCISES.md) for full per-exercise specs (task, files, hints, done-criteria).

**Foundation**
| ID | Task | Concepts |
|---|---|---|
| F-01 | Add `nnsimple.sub` op with verifier | TableGen op def, C++ verifier, Pure trait |
| F-02 | Add `nnsimple.neg` op + DRR canonicalization `neg(neg(x))→x` | Declarative rewrite rules |

**Ops & Type System**
| ID | Task | Mirror |
|---|---|---|
| A-01 | `nnsimple.matmul` with shape-compatibility verifier | `AddOp::verify` cpp:35-47 |
| A-02 | `!nnsimple.quantized<elementType, shape, scale, zero_point>` type | `TensorType` td:48-67 |

**Folders & Canonicalization**
| ID | Task | Mirror |
|---|---|---|
| B-01 | `MulOp::fold` for constant × constant | `AddOp::fold` cpp:50-69 |
| B-02 | C++ canonicalizations `mul(x,1)→x`, `mul(x,0)→0` | `AddZeroElimination` cpp:72-98 |

**Transformation Passes**
| ID | Task | Mirror |
|---|---|---|
| C-01 | `-nnsimple-fuse-mul-add` pass (new `FusedMulAddOp`) | `NNSimpleFuseAddReluRewriter` cpp:24-42 |
| C-02 | `-nnsimple-dce` dead-op elimination using the `Pure` trait | `mlir::isPure(op)` + `op->use_empty()` |

**Lowering & E2E**
| ID | Task | Mirror |
|---|---|---|
| D-01 | Lower `nnsimple.sub` to `linalg.sub` | `AddOpLowering` cpp:59-76 |
| D-02 | Lower `add+relu` all the way to LLVM dialect | upstream conversion passes |

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
