# GOW Command-Line Interface Guide

This document describes how to run the **Generic Optimization Workflow (GOW)** from the command line, with particular emphasis on the **FireWorks backend on HTC/HPC clusters using a batch scheduler such as Slurm**.

It complements `docs/User_Reference_Manual.md`, which documents the optimization specification and evaluator contract.

> **Scope**
>
> The commands and behavior described here correspond to the current `feature/fw-htc-grouping` implementation, including grouped FireWorks, queue waiting/retries, per-generation post-processing, launch manifests, and asynchronous generation archiving.

---

## 1. Installation

### 1.1 Local execution only

From the repository root:

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

The `gow` executable is installed by the package entry point:

```text
gow = gow.cli:app
```

Check the installation with:

```bash
gow info
```

### 1.2 FireWorks execution

For the FireWorks backend install the optional FireWorks dependencies:

```bash
pip install -e '.[fireworks]'
```

This installs the Python dependencies required to communicate with a FireWorks `LaunchPad`/MongoDB database and to use FireWorks queue launchers.

### 1.3 `zstd` for HTC generation archiving

The asynchronous FireWorks archiver creates `.tar.zst` archives by calling the system `zstd` executable. No additional Python zstd package is required.

Check that it is available:

```bash
which zstd
zstd --version
```

On Rocky/RHEL systems it is normally provided by the `zstd` package. A module, Spack, Conda, or another site installation is also valid as long as `zstd` is available in `PATH` to the process running `gow fw run`.

`zstd` is required only when FireWorks generation archiving is enabled with `--archive-generations`.

---

## 2. Command overview

The main CLI commands are:

| Command | Purpose |
|---|---|
| `gow info` | Check that the CLI is installed and commands are registered. |
| `gow evaluate` | Evaluate one candidate locally. Useful for evaluator debugging. |
| `gow run` | Run a complete optimization locally, without FireWorks. |
| `gow best` | Read completed JSONL results and print the best candidate(s). |
| `gow merge-runs` | Merge the completed `results.jsonl` files from several runs. |
| `gow archive-generation` | Manually archive candidate work directories from one generation. |
| `gow fw evaluate` | Submit one candidate evaluation to FireWorks, optionally launching it. |
| `gow fw run` | Run a complete optimization using FireWorks; this is the main HTC/HPC command. |

Use Typer's built-in help for the exact options supported by the installed version:

```bash
gow --help
gow run --help
gow evaluate --help
gow fw --help
gow fw evaluate --help
gow fw run --help
```

---

## 3. Result directory selection

Commands that take an optimization specification resolve the output directory in this order:

1. explicit `--outdir` / `-o`;
2. environment variable `GOW_OUTDIR`;
3. `<configuration-directory>/results`.

For example:

```bash
export GOW_OUTDIR=/scratch/$USER/gow/results
```

or:

```bash
gow run optimization_specs.yaml --outdir /scratch/$USER/gow/results
```

The output directory is a **problem-level results directory**. Individual runs are stored below:

```text
<outdir>/
  results.jsonl
  summary.json
  runs/
    <run_id>/
      ...
```

For production runs, explicitly setting both `--outdir` and `--run-id` makes provenance and recovery easier.

---

## 4. Run and candidate identifiers

If `--run-id` is not supplied to a full run, GOW generates a UUID.

Automatic optimization candidates receive run-aware identifiers containing their generation and global candidate index. A concrete execution also has an `attempt_index`/`attempt_id`.

This distinction is important for HTC recovery:

- `candidate_id` identifies the logical candidate;
- `attempt_index` identifies an execution attempt of that candidate;
- queue retries keep the same logical `candidate_id` and increment the attempt index.

---

# Part I — Local CLI

## 5. Evaluate one candidate locally

Use `gow evaluate` to debug an evaluator without FireWorks:

```bash
gow evaluate examples/toy/optimization_specs.yaml \
  --run-id debug-local \
  --candidate-id candidate-test \
  --param x=0.5 \
  --param y=-0.1
```

Parameter overrides can be repeated:

```bash
-p x=0.5 -p y=-0.1
```

or loaded from a JSON object:

```bash
gow evaluate optimization_specs.yaml \
  --params-file candidate.json
```

Useful options:

| Option | Meaning |
|---|---|
| `--outdir`, `-o` | Results directory. |
| `--run-id` | Run namespace. Default for this command: `manual`. |
| `--candidate-id` | Explicit candidate identifier. |
| `--generation-id` | Generation metadata. |
| `--candidate-index` | Global candidate sequence index. |
| `--attempt-index` | Attempt number, starting at `0`. |
| `--param`, `-p` | `NAME=VALUE` override; repeatable. |
| `--params-file` | JSON object containing parameter overrides. |

If `--candidate-id` is omitted but both `--generation-id` and `--candidate-index` are given, GOW generates a canonical candidate id. Otherwise the fallback id is `manual`.

The command prints the work directory, status, objective, metrics, return code, and wall time.

---

## 6. Run a complete optimization locally

```bash
gow run optimization_specs.yaml \
  --outdir /scratch/$USER/gow/results \
  --run-id local-test
```

The optimizer configuration (`batch_size`, `max_evaluations`, optimizer settings, etc.) is read from the YAML/JSON optimization specification.

Optional generation archiving is available:

```bash
gow run optimization_specs.yaml \
  --run-id local-test \
  --archive-generations \
  --delete-archived-workdirs
```

In the current local runner this manual/local archiving path uses generation `.tar.gz` archives. The FireWorks HTC archiver described later uses transactional `.tar.zst` archives and also archives FireWorks launcher directories.

---

# Part II — FireWorks configuration

## 7. LaunchPad configuration

`gow fw ...` needs a FireWorks `LaunchPad` configuration (`my_launchpad.yaml`) that points to the MongoDB database used by FireWorks.

The path is resolved in this order:

1. `--launchpad /path/to/my_launchpad.yaml`;
2. `--launchpad /path/to/config-directory`;
3. `FW_LAUNCHPAD_FILE`;
4. `$FW_CONFIG_DIR/my_launchpad.yaml`;
5. search for `my_launchpad.yaml` from the current directory upwards;
6. FireWorks `LaunchPad.auto_load()` as a last resort.

Example:

```bash
export FW_CONFIG_DIR=$HOME/.fireworks
```

with:

```text
$HOME/.fireworks/my_launchpad.yaml
```

The MongoDB server must be reachable from the process that submits/monitors FireWorks and from any FireWorks components that require database access.

---

## 8. Queue adapter configuration

Cluster submission with `--launcher queue` needs a valid FireWorks queue adapter, normally `my_qadapter.yaml`.

It is resolved in this order:

1. `--qadapter /path/to/my_qadapter.yaml`;
2. `--qadapter /path/to/config-directory`;
3. `FW_QADAPTER_FILE`;
4. `$FW_CONFIG_DIR/my_qadapter.yaml`;
5. search for `my_qadapter.yaml` from the current directory upwards.

A convenient cluster setup is therefore:

```bash
export FW_CONFIG_DIR=$HOME/.fireworks
```

with both files in the same directory:

```text
$HOME/.fireworks/
  my_launchpad.yaml
  my_qadapter.yaml
```

Scheduler details such as Slurm partition, account, wall time, resources, and the FireWorks rocket command belong in the FireWorks queue adapter configuration. GOW selects the queue launcher but does not replace the scheduler-specific adapter.

---

# Part III — FireWorks commands

## 9. Submit/evaluate a single candidate with FireWorks

`gow fw evaluate` is useful for validating FireWorks and cluster configuration before running an optimization.

### 9.1 Submit only

```bash
gow fw evaluate optimization_specs.yaml \
  --run-id fw-debug \
  --candidate-id test-001 \
  --param x=0.5 \
  --no-launch
```

This adds the workflow to the LaunchPad but does not launch it.

### 9.2 Launch locally through FireWorks

```bash
gow fw evaluate optimization_specs.yaml \
  --run-id fw-debug \
  --candidate-id test-001 \
  --param x=0.5 \
  --launch \
  --launcher local
```

### 9.3 Submit through the cluster queue

```bash
gow fw evaluate optimization_specs.yaml \
  --run-id fw-debug \
  --candidate-id test-001 \
  --param x=0.5 \
  --launch \
  --launcher queue \
  --qadapter $HOME/.fireworks/my_qadapter.yaml \
  --njobs-queue 1
```

Important options:

| Option | Meaning |
|---|---|
| `--launch/--no-launch` | Immediately invoke FireWorks rapidfire after submission. Default: no launch for `fw evaluate`. |
| `--launcher local\|queue` | Use local FireWorks rockets or the FireWorks queue launcher. |
| `--launchpad` | Explicit LaunchPad YAML or directory. |
| `--qadapter` | Queue adapter YAML or directory. |
| `--njobs-queue` | Maximum number of queued jobs passed to FireWorks queue rapidfire. |
| `--reserve` | Ask FireWorks to reserve jobs before launching them. |
| `--launch-dir` | Override the directory where FireWorks creates launcher directories. |
| `--nlaunches` | FireWorks rapidfire launch limit; `0` means until its queue is empty. |
| `--sleep` | Delay between FireWorks rapidfire launches. |

`fw evaluate` is a submission/debug command. In queue mode, do not treat its return as proof that the candidate itself has completed; inspect FireWorks/result state as appropriate.

---

# Part IV — Full HTC/HPC optimization

## 10. `gow fw run`: recommended cluster command

For a full closed-loop optimization on a Slurm/HTC cluster, use `gow fw run` with the FireWorks queue launcher.

Example:

```bash
gow fw run optimization_specs.yaml \
  --outdir /scratch/$USER/gow/results \
  --run-id mica-scan-001 \
  --launcher queue \
  --launchpad $HOME/.fireworks/my_launchpad.yaml \
  --qadapter $HOME/.fireworks/my_qadapter.yaml \
  --group-size auto \
  --njobs-queue 100 \
  --queue-wait \
  --queue-poll-seconds 10 \
  --queue-wait-timeout 0 \
  --max-missing-result-retries 2 \
  --archive-generations \
  --delete-archived-workdirs \
  --delete-archived-launchers \
  --archive-backlog 1
```

The optimizer's `batch_size`, `max_evaluations`, and algorithm-specific settings still come from `optimization_specs.yaml`.

---

## 11. How grouped FireWorks work

`--group-size N` controls how many candidate evaluations are placed in one FireWork.

For example:

```text
batch_size = 10000
group_size = 100
```

produces approximately:

```text
100 FireWorks per generation
```

Each grouped FireWork evaluates its candidates sequentially inside that FireWork. Different FireWorks can be processed concurrently by different FireWorks workers/queue jobs.

### `--group-size 0`

`0` means "use the complete optimizer batch as one FireWork".

For large HTC batches this is usually **not** the desired setting. A very large grouped FireWork:

- reduces available parallelism;
- increases the FireWork JSON/BSON payload;
- makes a single FireWork responsible for too much work;
- can approach MongoDB document limits.

The current slim-batch implementation avoids sending complete `batch_results` back through MongoDB, so the practical limit can be much larger than in earlier versions.

### `--group-size auto`

`auto` asks GOW to select the group size for each generation using **both** the real MongoDB BSON limit and the amount of queue parallelism requested with `--njobs-queue`.

The automatic selection has two constraints:

1. **MongoDB/FireWorks BSON safety**.
2. **HTC queue occupancy**.

GOW queries the MongoDB server used by the active FireWorks LaunchPad with the MongoDB `hello` command and reads values such as:

```text
maxBsonObjectSize
maxMessageSizeBytes
maxWriteBatchSize
```

The relevant BSON threshold is derived from the server's real `maxBsonObjectSize` and the configured safety fraction:

```text
safe_BSON_limit = maxBsonObjectSize × bson_safety_fraction
```

The default safety fraction is:

```text
0.75
```

Thus a MongoDB server advertising the usual 16 MiB `maxBsonObjectSize` produces a default GOW safety threshold of approximately 12 MiB.

GOW serializes the actual FireWork/Workflow database representation and measures the BSON payload before submission. The BSON check is repeated immediately before `LaunchPad.add_wf()`.

The second limit is the queue-parallelism target. For a generation containing `N` pending candidates and queue capacity `J`:

```text
HTC target group size = ceil(N / J)
```

For example:

```text
generation candidates = 200000
njobs_queue            = 400
HTC target group size  = 500
```

If the BSON-safe group size is at least 500, `auto` selects approximately 500 and produces about 400 grouped FireWorks.

If MongoDB allows only 300 candidates per FireWork:

```text
BSON-safe group size   = 300
HTC target group size  = 500
selected group size    = 300
FireWorks generated    = ceil(200000 / 300) = 667
```

This is intentional: it is preferable to create more FireWorks than queued jobs so that the available workers remain busy, rather than exceed the BSON safety limit.

Conceptually:

```text
selected_group_size = min(
    bson_safe_group_size,
    ceil(candidates_pending / njobs_queue)
)
```

The calculation is repeated for each generation and for retry subsets, so the selected group size can decrease near the end of a run or when only a small number of missing candidates must be resubmitted.

### BSON preflight without running the optimization

The automatic/fixed group size can be checked before a production run:

```bash
gow fw group-size-preflight optimization_specs.yaml \
  --launchpad $HOME/.fireworks/my_launchpad.yaml \
  --group-size auto \
  --njobs-queue 400
```

The preflight connects to the real MongoDB server to obtain its advertised limits and constructs representative FireWorks, but it does **not** submit a Workflow to MongoDB.

For production HTC runs, the recommended form is:

```bash
gow fw run optimization_specs.yaml \
  --launcher queue \
  --group-size auto \
  --njobs-queue 400 \
  --bson-safety-fraction 0.75
```

A fixed numeric group size remains supported. Even in that case, GOW performs the BSON safety check before submission and aborts before PyMongo if the FireWork would exceed the configured safe limit.

---

## 12. `--njobs-queue` versus `--group-size`

These parameters control different things:

- `--group-size`: candidates per FireWork, or `auto` for BSON- and queue-aware selection;
- `--njobs-queue`: maximum queued jobs controlled by FireWorks queue rapidfire and, in `auto` mode, the desired parallelism used to size grouped FireWorks.

For a fixed group size:

```text
number of grouped FireWorks ≈ ceil(batch_size / group_size)
```

Example:

```text
batch_size      = 10000
group_size      = 100
FireWorks/gen   = 100
njobs_queue     = 100
```

This configuration makes up to roughly one queued worker/job per grouped FireWork available, subject to the queue adapter, scheduler policy, and FireWorks behavior.

For `--group-size auto`, the relationship is inverted: GOW first tries to create enough grouped FireWorks to keep up to `njobs_queue` jobs useful, and then reduces the group size further if required by MongoDB BSON limits.

Increasing `njobs_queue` cannot create more useful parallel work than the number of FireWorks available in the current generation.

---

## 13. Synchronous generation control and active waiting

Although cluster jobs are asynchronous, the optimizer loop itself is generation-synchronous:

```text
ask candidates
    ↓
submit grouped FireWorks
    ↓
launch through queue
    ↓
wait for candidate result.json files
    ↓
retry missing candidates if necessary
    ↓
optimizer.tell(...)
    ↓
finalize generation
    ↓
next generation
```

Keep the default:

```bash
--queue-wait
```

for normal HTC optimization.

### `--queue-poll-seconds`

Controls how often GOW checks for completed individual `result.json` files and FireWork terminal states.

Example:

```bash
--queue-poll-seconds 10
```

For millisecond evaluators there is normally no benefit in polling extremely frequently because queue/scheduler latency dominates.

### `--queue-wait-timeout`

```bash
--queue-wait-timeout 0
```

means no explicit GOW timeout. GOW can still stop waiting when the corresponding FireWorks are known to be terminal and candidate results are missing.

A finite timeout can be useful if cluster jobs can remain in a stale/non-terminal state indefinitely:

```bash
--queue-wait-timeout 7200
```

Choose it with scheduler queue time and job wall time in mind.

### Do not normally use `--no-queue-wait`

For a closed-loop optimization, `optimizer.tell()` must not run before the current generation's results exist. `--no-queue-wait` is therefore an advanced/debug option and is not appropriate for a normal `gow fw run --launcher queue` workflow unless another orchestration layer guarantees result completion before the optimizer proceeds.

---

## 14. Automatic recovery of missing HTC results

The queue runner distinguishes a **logical candidate** from its execution attempts.

If a grouped FireWork/job terminates but some candidates did not produce their individual:

```text
runs/<run_id>/<candidate_id>/result.json
```

GOW resubmits only the missing candidates.

Control the number of retries with:

```bash
--max-missing-result-retries 2
```

The candidate keeps the same logical `candidate_id`; the execution attempt index is incremented.

This is intended to recover from infrastructure failures such as:

- a failed Slurm node;
- a `FIZZLED` FireWork;
- a killed worker/job;
- partial completion of a grouped FireWork.

If all configured retries are exhausted, GOW creates a synthetic failed `result.json` with:

```text
failure_kind = missing_result_after_retries
```

This allows the generation to close consistently rather than hanging forever. The failed fitness is passed to the optimizer as a failed evaluation.

---

## 15. Per-generation result pipeline

Workers do **not** append concurrently to a shared `results.jsonl`. Each candidate writes its own `result.json`.

After all candidates in the generation are accounted for, the coordinator calls generation finalization:

1. verify every expected candidate has an individual `result.json`;
2. build the generation shard;
3. update run progress/summary information;
4. write the generation FireWorks launch manifest;
5. optionally queue that completed generation for background archiving;
6. continue with the next generation.

The generation shard is:

```text
<outdir>/runs/<run_id>/generations/g000000.jsonl
<outdir>/runs/<run_id>/generations/g000001.jsonl
...
```

This avoids making a huge shared JSONL file part of the workers' critical path.

---

## 16. Progress and summaries

During the run, GOW maintains:

```text
runs/<run_id>/summary.jsonl
runs/<run_id>/summary.json
```

### `summary.jsonl`

Contains one row per finalized generation, including:

- generation id;
- expected/actual result count;
- generation results shard;
- best candidate of that generation.

It is useful for monitoring a long run without scanning millions of individual candidate directories.

### `runs/<run_id>/summary.json`

Contains the current run-level snapshot, including the **best result seen so far**, number of completed generations/evaluations, and whether the generation pipeline considers the run finalized.

### `<outdir>/summary.json`

Represents the latest/current problem-level summary written by GOW. If several runs share the same `outdir`, do not treat this as a historical database of all summaries; use each run's own summary files.

For the final optimum of a particular run, prefer:

```text
runs/<run_id>/summary.json
```

or:

```bash
gow best <outdir> --run-id <run_id> --config optimization_specs.yaml
```

---

## 17. FireWorks launch manifests

For each finalized generation GOW records the FireWork ids, including retry FireWorks, and their launch metadata in:

```text
runs/<run_id>/generations/g000000.launches.json
runs/<run_id>/generations/g000001.launches.json
...
```

The manifest records compact information such as:

- `fw_id`;
- FireWork terminal state;
- launch ids;
- launch directories;
- archived/live launch status.

This mapping is what allows the archiver to operate on the correct FireWorks leaf launch directories without guessing from directory names.

---

# Part V — Archiving and inode control

## 18. Enable background archiving in FireWorks mode

Generation archiving is disabled by default. Enable it with:

```bash
--archive-generations
```

In `gow fw run`, this starts **one background archiver thread**. Completed generations are submitted to that worker while the optimization coordinator proceeds with subsequent generations.

The archiver queue is bounded with:

```bash
--archive-backlog N
```

`--archive-backlog` is a **maximum number of completed generations allowed to wait for the single background archiver**. It is not a recommended retention count.

This option has a direct effect on peak inode consumption because candidate work directories are deleted only after their generation archive has been created and validated.

With the current candidate layout, one candidate normally contributes approximately six filesystem objects:

```text
candidate directory
input.json
output.json
result.json
stderr.txt
stdout.txt
```

Therefore a generation of 100,000 candidates can temporarily consume about:

```text
100000 × 6 ≈ 600000 inodes
```

If `--archive-backlog 1` permits three completed generations to accumulate while the archiver is busy, those waiting generations alone can represent approximately:

```text
3 × 600000 ≈ 1.8 million inodes
```

and the true peak can be higher because it may also include:

- the generation currently being archived;
- the generation currently being evaluated;
- FireWorks launcher directories and files.

For inode-constrained filesystems, prefer:

```bash
--archive-backlog 1
```

This applies backpressure to the coordinator much earlier. A larger backlog should only be used when the filesystem has enough inode headroom and overlapping compression with computation is worth the additional live filesystem footprint.

A useful first-order estimate is:

```text
candidate_inodes_per_generation ≈ candidates_per_generation × 6
queued_candidate_inodes ≈ archive_backlog × candidate_inodes_per_generation
```

This estimate excludes the archive-in-progress generation, the currently evaluated generation, and FireWorks launcher artifacts. Always leave additional safety margin.

For very large generations, the inode pressure exists **before** the generation can be archived. Reducing `--archive-backlog` limits the number of completed generations retained, but it cannot reduce the intrinsic inode footprint of the currently active generation.

---

## 19. Candidate workdir archives

For each completed generation the FireWorks archiver can create:

```text
runs/<run_id>/archives/candidates/g000000.tar.zst
runs/<run_id>/archives/candidates/g000001.tar.zst
...
```

Use:

```bash
--delete-archived-workdirs
```

to delete candidate work directories **only after** the corresponding archive has been created and validated.

Without this flag the archive is created but the candidate directories remain.

---

## 20. FireWorks launcher archives

Completed terminal leaf launch directories can be archived per generation into:

```text
runs/<run_id>/archives/launchers/g000000.tar.zst
runs/<run_id>/archives/launchers/g000001.tar.zst
...
```

Deletion of safely archived terminal leaf launcher directories is enabled by default when the asynchronous generation archiver is active:

```bash
--delete-archived-launchers
```

To keep them after archiving:

```bash
--keep-archived-launchers
```

GOW does not blindly delete the parent Slurm/pilot launcher tree while it may still be in use. It uses the generation launch manifest and FireWorks terminal state to identify safe leaf launch directories.

A launch directory containing:

```text
FW_offline.json
```

is skipped to avoid interfering with FireWorks offline recovery.

---

## 21. Transactional `.tar.zst` creation

The HTC archiver uses this sequence:

```text
sources
  ↓
.tar.zst.partial
  ↓
zstd integrity test
  ↓
atomic rename
  ↓
.tar.zst
  ↓
optional source deletion
```

Conceptually:

```text
tar stream → zstd -T0 -1 → archive.tar.zst.partial
zstd -t archive.tar.zst.partial
rename archive.tar.zst.partial → archive.tar.zst
```

Source directories are deleted only after the validated archive has been atomically published.

`zstd` thread tuning is currently left to the implementation default (`-T0`). If CPU contention becomes relevant in the future, this can be made configurable without changing the archive layout.

---

## 22. Final launcher cleanup

After the optimization loop finishes, the background archiver is drained.

If launcher deletion is enabled, GOW then checks all FireWorks referenced by generation manifests. Final cleanup is refused if:

- any FireWork is still non-terminal/uninspectable; or
- any `FW_offline.json` marker remains.

The remaining parent/pilot launcher tree is archived to:

```text
runs/<run_id>/archives/launchers/run-final.tar.zst
```

with a report such as:

```text
runs/<run_id>/archives/launchers/run-final.json
```

Only after successful archival can the remaining `runs/<run_id>/launchers/` tree be removed.

---

## 23. Archive reports

Per-generation archive metadata is stored under:

```text
runs/<run_id>/archives/manifests/g000000.archive.json
```

These reports record the generated archives, counts of launcher directories, skipped launchers, and deletion policy. They are useful when diagnosing why a particular launcher directory was intentionally kept.

---

# Part VI — Final result consolidation

## 24. `results.jsonl` is rebuilt, not hot-written by workers

At the end of a complete run GOW verifies that the expected number of results exists and rebuilds:

```text
runs/<run_id>/results.jsonl
```

from the per-generation result shards.

It then rebuilds the problem-level:

```text
<outdir>/results.jsonl
```

from run-level results.

This means the large JSONL files are **post-processed indexes**, not shared files that hundreds of workers append to concurrently.

For a running large HTC campaign, use generation shards and run summary files for progress monitoring instead of expecting the final `results.jsonl` to be continuously current.

---

## 25. Show the best candidate(s)

For one specific run:

```bash
gow best /scratch/$USER/gow/results \
  --run-id mica-scan-001 \
  --config optimization_specs.yaml
```

Show the best ten:

```bash
gow best /scratch/$USER/gow/results \
  --run-id mica-scan-001 \
  --config optimization_specs.yaml \
  --top 10
```

Without `--run-id`, the command reads the problem-level `results.jsonl`, which may contain several runs.

If no configuration is supplied, the default objective direction used by this command is `minimize`; use `--config` or explicitly pass:

```bash
--direction maximize
```

when required.

---

# Part VII — Multiple runs and reuse

## 26. Merge completed runs

Several completed run-level `results.jsonl` files can be combined with:

```bash
gow merge-runs /scratch/$USER/gow/results \
  --target-run-id combined-2026 \
  --source-run-id scan-day-1 \
  --source-run-id scan-day-2 \
  --source-run-id scan-day-3
```

The output is:

```text
runs/combined-2026/results.jsonl
```

Duplicate records are filtered using run/attempt/candidate identifiers and records are sorted by generation/candidate metadata.

### Important limitation

`merge-runs` is a **result aggregation** operation. It does not reconstruct or resume an optimizer's internal state, and it does not turn several independent optimizer trajectories into one mathematically continuous optimization run.

It is useful for:

- combined analysis;
- reusing completed evaluations as a result corpus;
- comparing/aggregating campaigns launched on different days.

True optimizer checkpoint/restart would require optimizer-specific state persistence and is a separate capability.

---

## 27. Manually archive one generation

The standalone command:

```bash
gow archive-generation /scratch/$USER/gow/results \
  --run-id mica-scan-001 \
  --generation-id 42 \
  --delete-source
```

archives candidate workdirs discovered for that generation.

This is mainly a manual maintenance/recovery utility. The production FireWorks path should normally use `gow fw run --archive-generations`, because the asynchronous HTC archiver also handles FireWorks launch metadata and launcher directories safely.

---

# Part VIII — Recommended Slurm/HTC workflow

## 28. Preflight

Before a large run:

```bash
source venv/bin/activate
export FW_CONFIG_DIR=$HOME/.fireworks
export GOW_OUTDIR=/scratch/$USER/gow/my-problem

which zstd
```

Confirm MongoDB/LaunchPad connectivity with a small FireWorks evaluation or small run before launching millions of evaluations.

---

## 29. Small cluster validation

Use the production queue path but with a small optimization configuration:

```bash
gow fw run optimization_specs-small.yaml \
  --run-id validation-001 \
  --launcher queue \
  --group-size 10 \
  --njobs-queue 4 \
  --queue-wait \
  --max-missing-result-retries 1
```

Check:

```text
runs/validation-001/generations/
runs/validation-001/summary.json
runs/validation-001/summary.jsonl
```

and FireWorks/MongoDB states.

---

## 30. Large HTC run template

A typical high-throughput invocation is:

```bash
gow fw run optimization_specs.yaml \
  --run-id production-001 \
  --launcher queue \
  --group-size auto \
  --njobs-queue 100 \
  --queue-wait \
  --queue-poll-seconds 10 \
  --queue-wait-timeout 0 \
  --max-missing-result-retries 2 \
  --archive-generations \
  --delete-archived-workdirs \
  --delete-archived-launchers \
  --archive-backlog 1
```

If `FW_CONFIG_DIR` is not set, add explicit `--launchpad` and `--qadapter` paths.

### Parameters to tune first

1. `optimizer.batch_size` in the optimization specification;
2. `--group-size`;
3. `--njobs-queue`;
4. scheduler resources/wall time in `my_qadapter.yaml`;
5. `--archive-backlog` if filesystem pressure or archive throughput requires it.

Do not increase all of these independently without checking MongoDB payload size, Slurm scheduling behavior, filesystem load, and evaluator runtime.

---

# Part IX — Output layout

## 31. Typical FireWorks run before/while archiving

```text
<outdir>/
├── results.jsonl                    # rebuilt at end across runs
├── summary.json                     # latest/current problem summary
└── runs/
    └── <run_id>/
        ├── summary.json             # run-level current/final summary
        ├── summary.jsonl            # one progress record per generation
        ├── results.jsonl            # rebuilt at run completion
        ├── generations/
        │   ├── g000000.jsonl
        │   ├── g000000.launches.json
        │   ├── g000001.jsonl
        │   ├── g000001.launches.json
        │   └── ...
        ├── archives/
        │   ├── candidates/
        │   │   ├── g000000.tar.zst
        │   │   └── ...
        │   ├── launchers/
        │   │   ├── g000000.tar.zst
        │   │   ├── ...
        │   │   ├── run-final.tar.zst
        │   │   └── run-final.json
        │   └── manifests/
        │       ├── g000000.archive.json
        │       └── ...
        ├── launchers/                # live FireWorks tree; removable after final safe archive
        └── <candidate_id>/            # individual workdir; optional deletion after archive
            ├── input.json
            ├── output.json
            ├── result.json
            ├── stdout.txt
            └── stderr.txt
```

If candidate workdir deletion and launcher deletion are enabled, the large inode-heavy directory trees disappear after their validated archives have been produced.

---

# Part X — Operational notes and troubleshooting

## 32. Run appears stuck waiting for a generation

Check, in this order:

1. how many expected candidate `result.json` files exist;
2. FireWork states in LaunchPad/MongoDB;
3. Slurm job states;
4. whether missing candidates are being retried;
5. `--queue-wait-timeout` behavior;
6. evaluator stderr/stdout for candidates that repeatedly fail.

A node failure should normally cause missing candidates to be retried after their FireWorks reach terminal state or the configured wait timeout is reached.

---

## 33. MongoDB `DocumentTooLarge`

Use grouped FireWorks carefully. The current slim-batch implementation deliberately avoids writing complete `batch_results` back to the FireWork spec/MongoDB, which substantially reduces result-side BSON growth.

However, the **input FireWork definition still contains the candidate parameter payloads**. Very large groups can therefore still produce large FireWorks documents.

The preferred protection is:

```bash
--group-size auto
```

or, before a production run:

```bash
gow fw group-size-preflight optimization_specs.yaml \
  --launchpad $HOME/.fireworks/my_launchpad.yaml \
  --group-size auto \
  --njobs-queue 400
```

GOW queries the actual MongoDB server with `hello`, applies `--bson-safety-fraction`, and measures the serialized FireWork/Workflow BSON before submission.

If a fixed numeric `--group-size` is used, the same BSON safety check is still performed immediately before `LaunchPad.add_wf()`.

If MongoDB nevertheless reports:

```text
pymongo.errors.DocumentTooLarge
```

record the server's `hello` values and the measured FireWork size. This can indicate that the oversized document is being generated by a different FireWorks/PyMongo update path rather than by the initial grouped FireWork definition.

---

## 34. Too many filesystem inodes

For large HTC scans, use:

```bash
--archive-generations \
--delete-archived-workdirs \
--delete-archived-launchers
```

This allows completed candidate directories and terminal FireWorks launch directories to be replaced by a small number of `.tar.zst` files per generation.

If archiving cannot keep up with generation production, `--archive-backlog` limits the pending generation queue and applies backpressure to the coordinator.

Do not interpret a larger backlog as safer. With approximately six filesystem objects per candidate, a 100,000-candidate generation is already about 600,000 candidate-workdir inodes before launcher artifacts. On inode-constrained storage, `--archive-backlog 1` is the recommended starting point. A backlog of 3 can allow roughly 1.8 million candidate-workdir inodes to wait for archival, plus the active and archive-in-progress generations.

---

## 35. Archiver fails

The asynchronous archiver propagates its error back to the main process on subsequent submission/close rather than silently deleting data.

Check:

```bash
which zstd
```

and available filesystem space/permissions.

Because archives are validated before publication and deletion, a compression failure should leave the source directories intact and remove/avoid publishing an invalid final archive.

---

## 36. Launcher directories were intentionally not deleted

Inspect:

```text
runs/<run_id>/archives/manifests/gXXXXXX.archive.json
```

Possible reasons include:

- launch not in a terminal state;
- missing/unknown launch state;
- launch path outside the run launcher root;
- `FW_offline.json` present for offline recovery protection.

Final launcher cleanup is also refused if any FireWork from the run manifests is still active or cannot be safely inspected.

---

## 37. Results during a very large run

Do not repeatedly scan millions of candidate directories to monitor progress.

Prefer:

```text
runs/<run_id>/summary.json
runs/<run_id>/summary.jsonl
runs/<run_id>/generations/gXXXXXX.jsonl
```

The generation pipeline exists specifically so progress and best-so-far information can be obtained from compact files while individual workdirs are archived/deleted.

---

# Part XI — Quick reference

## 38. Local run

```bash
gow run optimization_specs.yaml -o results --run-id run-001
```

## 39. Local candidate debug

```bash
gow evaluate optimization_specs.yaml --run-id debug -p x=1.0 -p y=2.0
```

## 40. FireWorks local backend

```bash
gow fw run optimization_specs.yaml \
  -o results \
  --run-id fw-local-001 \
  --launcher local \
  --group-size 10
```

## 41. FireWorks + Slurm/queue backend

```bash
gow fw run optimization_specs.yaml \
  -o /scratch/$USER/gow/results \
  --run-id htc-001 \
  --launcher queue \
  --group-size auto \
  --njobs-queue 100 \
  --queue-wait \
  --max-missing-result-retries 2 \
  --archive-generations \
  --delete-archived-workdirs \
  --delete-archived-launchers
```

## 42. Best result from one run

```bash
gow best results --run-id htc-001 --config optimization_specs.yaml
```

## 43. Merge completed runs

```bash
gow merge-runs results \
  --target-run-id merged \
  --source-run-id run-a \
  --source-run-id run-b
```

---

## 44. Recommended production defaults for HTC

For short evaluator tasks where FireWorks/Slurm overhead dominates, a reasonable initial operational profile is:

```text
launcher                    queue
queue_wait                  enabled
group_size                  auto (or benchmark a fixed value when required)
njobs_queue                 match useful concurrent grouped FireWorks / site limits
queue_poll_seconds          10
max_missing_result_retries  2
archive_generations         enabled
delete_archived_workdirs    enabled for inode-constrained runs
delete_archived_launchers   enabled
archive_backlog             1 on inode-constrained filesystems
```

These are operational starting points, not universal performance constants. Benchmark `group_size` and queue concurrency with the actual evaluator, MongoDB deployment, filesystem, and scheduler policy before launching very large campaigns.
