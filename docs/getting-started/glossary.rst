Glossary
========

**Author**: Linus Eickhoff (CERN)

This page summarizes key terms used in distributed training and high-performance computing (HPC),
providing quick reference for terminology relevant to the itwinai project, its documentation, and codebase.

HPC Terms
---------

* **accelerator**

  Specialized compute device (e.g., GPU, TPU, FPGA) designed to speed up specific workloads.

* **collectives / collective operations / collective communications**

  Concurrency primitives that involve all ranks in a communicator to move or reduce data in a single operation. Read more `here <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html>`_.

  * **all-gather**
    
    Each rank sends its local tensor and receives the concatenated result from every rank.

  * **all-reduce**
  
    All ranks combine their tensors with an element-wise reduction (e.g., sum) and obtain the identical reduced result.

  * **broadcast**

    A single source rank transmits a tensor that every other rank receives unchanged.


* **directive**

  Compiler instruction embedded that modifies compilation or execution behavior, such as enabling parallelism or optimization, without altering program logic.

* **job**
  
  Submitted unit of work that the scheduler runs with allocated resources.

* **InfiniBand**
  
  High-bandwidth, low-latency network interconnect widely used in HPC clusters.

* **node**
  
  Physical server in a cluster containing CPUs, memory, and often accelerators.

* **NUMA**
  
  Non-Uniform Memory Access architecture in which memory latency depends on the socket that owns the memory.

* **NVLink**
  
  Point-to-point GPU interconnect from NVIDIA that offers higher bandwidth than PCIe.

* **NVSwitch**
  
  On-package switch fabric providing all-to-all NVLink connectivity among multiple GPUs in a node.

* **PCIe**
  
  Serial expansion bus standard that connects CPUs, GPUs, Network Interface Controllers (NICs), and storage devices.

* **rank**

  Index assigned to each device (e.g. GPU) in a distributed group, identifying its position in collective operations.

* **RDMA**
  
  Remote Direct Memory Access lets one host read or write another host’s memory without CPU intervention.

* **RPC (Remote Procedure Call)**

  Protocol that allows a program to execute a procedure or function on a remote system as if it were local.

* **scheduler (e.g. SLURM)**
  
  Software that queues jobs and assigns cluster resources according to policy and priority.

* **straggler**
  
  Task or node that runs significantly slower than its peers, delaying synchronous operations.

* **task (SLURM)**
  
  Smallest schedulable execution unit within a job, typically a process or thread.

* **wall time / wall-clock time**
  
  Real-world elapsed time from job start to finish.


Distributed ML Terms
--------------------

* **data parallelism**
  
  Replicates the full model on every device and synchronizes gradients in or after mini-batch.

* **HPO (Hyperparameter Optimization)**

  Process of systematically searching for the best hyperparameter values to maximize model performance.

* **model parallelism**
  
  Splits model layers or parameter shards across devices so a single forward pass spans multiple accelerators.

* **tensor parallelism**
  
  Partitions individual tensors along dimensions, letting different accelerators compute slices of the same layer.

* **trial**

  Single train run of a model with a specific set of hyperparameters, evaluated independently.

* **world size**

  Total number of devices participating in the current distributed run.


Libraries for Distributed Computing
-----------------------------------

* **CUDA**

  NVIDIA’s GPU-computing platform and runtime for C, C++, and Python kernels.

* **gRPC**

  High-performance RPC framework using HTTP/2 and Protocol Buffers for language-agnostic services.

* **Kubernetes**

  Cluster-orchestration system for scheduling and managing containerized applications.

  * **helm**

    Package manager that deploys and upgrades Kubernetes applications via declarative charts.

  * **pod**

    Smallest deployable Kubernetes object, grouping one or more tightly coupled containers.

* **MPI (Message Passing Interface)**

  Family of libraries implementing the Message Passing Interface standard for distributed communication.
  Used for point-to-point and collective operations in distributed applications.

  * **MPI**

    The MPI specification defining point-to-point and collective semantics for parallel programs.

  * **OpenMPI**

    Popular open-source, production-grade implementation of the MPI standard.

* **NCCL**
  
  NVIDIA Collective Communications Library optimized for intra- and inter-node GPU collectives.

* **OpenMP**
  
  Compiler-directive API for shared-memory parallelism on multicore CPUs.

* **RCCL**
  
  AMD’s Radeon Collective Communications Library, drop-in compatible with NCCL for AMD GPUs.

* **ROCm**

  AMD’s open-source GPU-computing stack analogous to CUDA.

* **Singularity**
  
  Container runtime tailored to HPC that runs unprivileged, reproducible images (similar to Docker).

* **SLURM**
  
  Open-source workload manager that queues jobs and allocates nodes on HPC systems.


Libraries for ML
----------------

* **DDP**

  PyTorch’s DistributedDataParallel wrapper enabling synchronous data-parallel training across ranks.

* **DeepSpeed**

  Microsoft library that extends PyTorch with memory-efficient optimizers, ZeRO sharding, and kernel fusions.

  * **shard**

    A slice of parameters or optimizer states stored on a specific rank in ZeRO.

  * **ZeRO**

    Optimization algorithm that partitions optimizer states, gradients, and parameters to fit massive models.

* **Horovod**

  Framework providing MPI/NCCL-backed data-parallel training APIs across major DL frameworks.

* **Ray**

  Distributed execution framework offering HPO and task, actor, and object store abstractions for Python.

  * **placement group**

    Ray construct for requesting a set of resources that are grouped or located together on the same machine or nearby.

  * **KubeRay**

    Kubernetes operator that provisions and manages Ray clusters as native resources.

  * **Ray Tune**

    Ray’s HPO library that supports distributed trials and advanced search and HPO-scheduling algorithms.

