# Unit and Integration Tests for Torch-Based Features in itwinai

Matteo Bunino (CERN)

In this subdirectory, we store tests for the itwinai logic concerning Torch-related features,  
such as the itwinai `TorchTrainer`.

## Testing Distributed Features

Some tests here are more advanced, as they are designed to validate the implementation of
parallel computations, such as distributed machine learning and hyperparameter optimization
(HPO). In general, testing distributed features involves the following steps:

1. **Launching multiple worker processes**, which is done using different distributed execution
   frameworks or cluster job schedulers, such as `torchrun`, `srun`, `mpirun`, or by
   initializing a Ray cluster.
2. **Running the tests independently on each worker** using `pytest`.
3. **Establishing inter-worker communication** within the test cases. For simplicity,
   communication is set up once and shared across all tests, but this is an implementation
   detail.

### Matching the Right Tests with the Right "Launcher"

By "launcher," we refer to the utility that initializes the distributed environment, setting up
the communication backend or cluster infrastructure so that worker processes can discover each
other and correctly exchange information during a distributed job. Examples include `torchrun`,
`srun`, `mpirun`, and initializing a Ray cluster.

Different distributed computing strategies in Python require different launchers. In the test
suite, each distributed test is marked with two pytest markers:

- **`@pytest.mark.hpc`**: Specifies that the test must run on HPC resources and should be
  deselected when executed in other environments.
- One of the following to specify the distributed strategy used in the test:
  - **`@pytest.mark.torch_dist`** for **Torch Distributed Data Parallel (DDP)**
  - **`@pytest.mark.horovod_dist`** for **Horovod**
  - **`@pytest.mark.deepspeed_dist`** for **DeepSpeed**
  - **`@pytest.mark.ray_dist`** for **Ray Distributed**

When running the tests, ensure the correct launcher is used for the corresponding pytest
marker.  

For example, the **DeepSpeed** distributed strategy can be executed using `torchrun`, `srun`,
or `mpirun`. Thus, the tests can be launched in any of the following ways:

```bash
# Use torchrun
torchrun ... pytest -m deepspeed_dist .

# Use srun (SLURM job scheduler)
srun ... pytest -m deepspeed_dist .

# Use mpirun (MPI-based execution)
mpirun ... pytest -m deepspeed_dist .
```

On the other hand, Horovod is not compatible with the `torchrun` launcher, so the following is
incorrect and will fail:

```bash
torchrun ... pytest -m horovod_dist .
```

For simplicity, arguments for `torchrun`, `srun`, and `mpirun` have been omitted.

For more details, please refer to `slurm.vega.sh` and `slurm.jsc.sh` to see how each set of
tests is lauched using different launchers.

### Overview of Distributed Launchers

| Launcher  | Description |
|-----------|-------------|
| **`torchrun`** | Native PyTorch launcher for distributed training. Sets up communication via Gloo or NCCL. |
| **`srun`** | SLURM workload manager’s parallel execution tool. Used in HPC clusters for launching distributed jobs. |
| **`mpirun`** | OpenMPI launcher that initializes and runs MPI-based distributed applications. |
| **Ray Cluster** | A distributed computing framework for parallel and distributed workloads, managing worker nodes dynamically. |

### Debugging Tips

Distributed tests are automatically executed on HPC clusters as part of each release cycle
using our **Dagger** pipeline. This pipeline relies on **interLink** to connect to HPC
environments, where tests are run inside **Singularity/Apptainer** containers.

However, if the automatic tests fail, you'll need to fix them and redo the release. Here are
some debugging tips to (hopefully) address the failing tests quickly.

- **Debugging within containers can be slow**, as every change requires rebuilding and
  redeploying the container.  
  A SLURM job script for JSC is available in this directory to facilitate launching the tests
  using a Python virtual environment instead. Another SLURM job script for Vega shows how to
  run the test using containers. It is generally good to start debugggin with pyhon virtual
  environments, and move to the containerized environments later.
- **Pytest output is shown only on the main worker**, while logs from other workers are
  redirected to files.  
  - If all tests pass but failures appear in `stderr`, or if a test fails without a clear
    reason, **⚠️ check the logs of all workers!**. Worker logs can be found in `logs_torchrun`,
    `logs_srun`, and `logs_mpirun` directories. On the other hand, Ray usually tries to report
    workers info on the stdout and stderr of the main worker (user `-s` option for pytest).
- **Ray workers may be slow to start**: if the tests fails to correctly connect to the Ray
  cluster, consider that sometimes it may just due to the fact that Ray Head node and Workers
  are slow at starting... Try to wait a bit more (increasing the sleep time) for the Ray
  cluster to be ready, before launching any job.
- **Increase pytest verbosity**. By default pytest hides stdout from tests, which may be useful
  when debugging. Consider adding `-o log_cli=true -o log_cli_level=INFO` and `-s` options to
  get more verbose outputs.
