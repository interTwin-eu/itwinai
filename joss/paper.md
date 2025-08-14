---
title: "itwinai: A Python Toolkit For Scalable Scientific Machine Learning On HPC"
tags:
  - Python
  - Digital Twins
  - Distributed Training
  - Hyperparameter Optimization
  - High Performance Computing
authors:
  - given-names: Matteo
    surname: Bunino
    orcid: 0009-0008-5100-9300
    affiliation: 1
    corresponding: true 
  - given-names: Jarl Sondre
    surname: Sæther
    orcid: 0009-0002-7971-2213
    affiliation: 1
  - given-names: Linus Maximilian
    surname: Eickhoff
    orcid: 0009-0006-6691-2821
    affiliation: 1
  - given-names: Anna Elisa
    surname: Lappe
    email: anna.elisa.lappe@cern.ch
    affiliation: 1
  - given-names: Kalliopi
    surname: Tsolaki
    email: kalliopi.tsolaki@cern.ch
    affiliation: 1
  - given-names: Killian
    surname: Verder
    email: killian.verder@cern.ch
    affiliation: 1
  - given-names: Henry
    surname: Mutegeki
    email: henry.mutegeki@cern.ch
    affiliation: 1
    orcid: 0009-0001-9940-1167
  - given-names: Roman
    surname: Machacek
    email: roman.machacek@cern.ch
    affiliation: 1
  - given-names: Oleksandr
    surname: Krochak
    orcid: 0009-0007-2245-9452
    affiliation: 2
  - given-names: Mario
    surname: Rüttgers
    orcid: 0000-0003-3917-8407
    affiliation: "2, 3"
  - given-names: Rakesh
    surname: Sarma
    orcid: 0000-0002-7069-4082
    affiliation: 2
  - given-names: Maria
    surname: Girone
    affiliation: 1
    orcid: 0000-0003-0261-8392
  - given-names: Andreas
    surname: Lintermann
    orcid: 0000-0003-3321-6599
    affiliation: 2
affiliations:
 - name: European Organization for Nuclear Research (CERN), Espl. des Particules 1, 1217 Genève, Switzerland
   index: 1
 - name: Forschungszentrum Jülich, Jülich Supercomputing Center, Germany
   index: 2
 - name: Data-Driven Fluid Engineering (DDFE) Laboratory, Inha University, Incheon, Republic of Korea
   index: 3
date: 16 August 2025
bibliography: paper.bib

---

<!-- markdownlint-disable MD025 -->

# Summary

The integration of Artificial Intelligence (AI) into scientific research has
expanded significantly over the past decade, driven by large-scale datasets and
specialized hardware accelerators. Graphics Processing Units (GPUs) have become
essential for AI workloads, with modern High-Performance Computing (HPC)
facilities housing thousands of accelerators: the JUWELS Booster system
[@JUWELS] at Jülich Supercomputing Center (JSC) operates over 3,700 NVIDIA A100
GPUs, while Finland's LUMI supercomputer provides access to more than 11,000 AMD
MI250X GPUs.

This growth has enabled development of numerous domain-specific AI solutions
across scientific fields, offering the potential for faster experimentation and
shorter feedback cycles. However, many researchers face significant barriers
when deploying AI workflows on HPC systems, including the complexity of
distributed training, Hyperparameter Optimization (HPO) at scale, and the
specialized knowledge required for efficient accelerator utilization. The
heterogeneous nature of HPC systems forces scientists to focus on low-level
implementation details rather than their core research.

To address these challenges, we present `itwinai`, a Python library designed to
bridge the gap between domain science, AI, and HPC. The library provides a
standardized interface enabling seamless scaling from laptops to supercomputers.
Through its modular architecture, `itwinai` significantly lowers the barrier to
entry for domain specialists while ensuring optimal HPC resource utilization.

# Statement of Need

Efficient GPU utilization is crucial for modern AI workflows, yet integrating AI
with HPC infrastructure typically demands substantial effort from scientists.
`itwinai` is a Python library that streamlines this process by bridging AI, HPC,
and domain sciences through ready-to-use components. The library addresses three
core scientific ML needs: distributed training with uniform interfaces across
PyTorch DDP [@torch_ddp], DeepSpeed [@deepspeed], and Horovod [@horovod]
backends; scalable HPO via Ray Tune [@ray_tune] supporting various search
strategies; and comprehensive performance analysis including profiling, scaling
and parallel efficiency metrics, and bottleneck identification. All features
work seamlessly from laptops to supercomputers with minimal changes.

Built with a modular architecture, `itwinai` enables component integration and
plugin-based extensions for independent development of scientific use cases.
Developed as part of the interTwin [^intertwin_eu] project to support Digital
Twin applications across physics to environmental sciences, the library has
proven versatile for general AI applications. By consolidating diverse
functionalities into a single framework, `itwinai` significantly lowers the
barrier to HPC adoption and empowers the scientific community to scale AI
workloads efficiently.

[^intertwin_eu]: interTwin project: [intertwin.eu](https://www.intertwin.eu/)
(Accessed on 2025-08-14).

# Package Features

**Configuration for reproducible AI workloads**: `itwinai` employs a
declarative, YAML-based configuration system that separates experimental
parameters from implementation code. Configurations are hierarchical,
composable, and Command Line Interface (CLI)-overrideable. This design ensures
reproducible and portable executions across environments (workstation, cloud,
HPC) with minimal code change.

**Distributed training and inference**: `itwinai` supports multiple distributed
ML training frameworks, including PyTorch-DDP, DeepSpeed, Horovod and
Ray [@ray:2018]. The underlying implementation of these frameworks is hidden
from the user and they can simply change the parallelization strategy through a
flag, to investigate potential performance gains.

<!-- ![Distributed Data Parallel](img/distributed_data_parallel.pdf) -->

**HPO** systematically improves model performance by searching the
hyperparameter space. Leveraging Ray, itwinai scales HPO across HPC systems by
(i) assigning multiple workers to a single trial (e.g., data-parallel training)
and (ii) running many trials concurrently to exploit massive parallelism. It
also provides robust scheduling and fine-grained resource management for large,
distributed studies (\autoref{fig:hpo}).

![Conceptual representation of an HPO workflow in itwinai.\label{fig:hpo}](img/hpo_itwinai_figure.pdf)

**Profilers**: `itwinai` integrates with multiple profilers, such as the
`py-spy` profiler [@py_spy] and the PyTorch Profiler, and also logs metrics
about training time, GPU utilization and GPU power consumption.

**ML logs tracking**:  `itwinai` integrates with existing ML logging frameworks,
such as TensorBoard [@tensorboard], Mlflow [@mlflow], Weights&Biases [@wandb],
and yProvML [@yprovml] logger, and provides a unified interface across all of
them through a thin abstraction layer.

**Offloading to HPC and Cloud**: To benefit from both cloud and HPC, interLink
[@interlink] is used, a lightweight component to enable seamless offloading of
compute-intensive jobs from cloud and HPC, performing an automatic translation
from Kubernetes pods to SLURM jobs.

**Continuous Integration and Development** `itwinai` includes extensive tests
(library and use cases). A Dagger pipeline [^dagger] builds containers on
release, runs smoke tests on GitHub Actions (Azure runners: 4 CPUs, 16 GB)
[^choosing_gh_runners], offloads distributed tests to HPC via interLink, and
publishes on success.

[^dagger]: Dagger: [dagger.io](https://dagger.io/) (Accessed on 2025-08-14).

[^choosing_gh_runners]: GitHub hosted runners define the type of machine that will process a job
in your workflow.
[Find more here](https://docs-internal.github.com/en/actions/how-tos/write-workflows/choose-where-workflows-run/choose-the-runner-for-a-job?utm_source=chatgpt.com)
(Accessed on 2025-08-14).

# Use-case integrations

There are a wide range of scientific use-cases
currently integrated with `itwinai` via plug-in architecture. Earth-observation
plugins cover hydrological forecasting, drought prediction, and
climate/remote-sensing pipelines; physics plugins include high-energy physics,
radio astronomy, lattice QCD, and gravitational-wave/glitch analysis. Packaging
these as `itwinai` plugins enables reproducible, shareable workflows that run
consistently from laptops to HPC. The full list of `itwinai` plugins can be
found at [this
link](https://itwinai.readthedocs.io/latest/getting-started/plugins-list.html).

# Performance

`itwinai` provides tools to assess scalability and diagnose bottlenecks,
enabling efficient and accountable use of HPC resources. Two complementary
components are provided: scalability report generation and profiling.

## Scalability Report

For data-parallel training, throughput improves with worker count while
all-reduce communication costs grow, until communication overhead dominates and
scaling plateaus. The report characterizes this trade-off across GPUs/nodes and
backends, reporting wall-clock epoch time, relative speedup
(\autoref{fig:speedup}), GPU utilization (0–100%), energy (Wh), and
compute-versus-other time (\autoref{fig:compvsother}). Considered jointly, these
metrics identify the most efficient configuration and distribution strategy,
rather than relying on a single indicator.

Figure (\autoref{fig:speedup}) and (\autoref{fig:compvsother}) show the
scalability of the physics use case from INFN [^infn] targeting
gravitational-wave analysis at the Virgo interferometer.

[^infn]: Istituto Nazionale di Fisica Nucleare [infn.it](https://www.infn.it/en/)
(Accessed on 2025-08-14).

![Relative speedup of average epoch time vs. number of workers for the Virgo use
case.\label{fig:speedup}](img/virgo_relative_epoch_time_speedup.svg){ width=70% }

![Proportion of time spent on computation versus other operations, such as
collective communication, in the Virgo use case, broken down by number of
workers and distributed framework
\label{fig:compvsother}](img/virgo_computation_vs_other_plot.svg){ width=110% }

## Addressing bottlenecks

To explain why performance degrades, `itwinai` integrates low-overhead,
sample-based profiling (e.g., py-spy [@py_spy]) and summarizes flame-graph data
into actionable hotspots (e.g., data loading and I/O, kernel execution, host–device
transfer, communication). These summaries guide targeted remedies such as
adjusting batch size, data-loader parallelism, gradient accumulation, or
backend/collective settings.

# Outlook and future developments

`itwinai` provides ready-to-use ML tools that are applicable across a wide range
of scientific applications. The development of the library is continued through
other follow-up projects. The future developments will include integration of
new scientific use cases, exploring additional parallelism approaches,
integration of advanced user interfaces, addition of other EuroHPC systems and
performance optimization.

# Acknowledgements

This work has been funded by the European Commission in the context of the
interTwin project, with Grant Agreement Number 101058386. In the context of the
interTwin project, `itwinai` has been actively developed on HPC systems from JSC
such as the HDF-ML and JUWELS Booster system and EuroHPC resources such as Vega
HPC. `itwinai` is an open-source Python library primarily developed by CERN, in
collaboration with Forschungszentrum Jülich (FZJ).

# References
