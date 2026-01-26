---
title: "itwinai: A Python Toolkit for Scalable Scientific Machine Learning on HPC Systems"
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
    affiliation: 1
    orcid: 0009-0009-4804-4188
  - given-names: Kalliopi
    surname: Tsolaki
    email: kalliopi.tsolaki@cern.ch
    affiliation: 1
    orcid: 0000-0002-3192-4260
  - given-names: Killian
    surname: Verder
    email: killian.verder@cern.ch
    affiliation: 1
    orcid: 0009-0006-4819-3229
  - given-names: Henry
    surname: Mutegeki
    email: henry.mutegeki@cern.ch
    affiliation: 1
    orcid: 0009-0001-9940-1167
  - given-names: Roman
    surname: Machacek
    email: roman.machacek@cern.ch
    affiliation: 1
    orcid: 0009-0007-9976-4420
  - given-names: Maria
    surname: Girone
    affiliation: 1
    orcid: 0000-0003-0261-8392
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
expanded significantly over the past decade, driven by the availability of
large-scale datasets and Graphics Processing Units (GPUs),
in particular at High Performance Computing (HPC) sites.

However, many researchers face significant barriers
when deploying AI workflows on HPC systems, as their heterogeneous nature forces
scientists to focus on low-level implementation details rather than on their core
research. At the same time, the researchers often lack specialized HPC/AI knowledge
to implement their workflows efficiently.

To address this, we present `itwinai`, a Python library that simplifies scalable
AI on HPC. Its modular architecture and standard interface allow users to scale
workloads efficiently from laptops to supercomputers, reducing implementation
overhead and improving resource usage.

# Statement of need

Integrating machine learning into scientific workflows on HPC systems remains
complex. Researchers must often invest substantial effort to configure
distributed training, manage hyperparameter optimization, and analyze
performance, while adapting to varied system architectures.

`itwinai` is a Python library that streamlines this process by providing a
unified framework for scalable AI workflows. It offers consistent interfaces for
distributed training, supports large-scale hyperparameter optimization, and
includes tools for profiling and code scalability analysis.

Developed within the [interTwin](https://www.intertwin.eu/) project
[@manzi_intertwin_2025] to support Digital Twin applications in physics and
environmental sciences, `itwinai` is designed to be extensible and reusable. By
consolidating core functionality into a single framework, it lowers the
technical barrier to HPC adoption and enables researchers to focus on scientific
objectives.

# State of the field

Scalable AI workflows on HPC systems are typically assembled from multiple
specialized tools. PyTorch-DDP [@torch_ddp], DeepSpeed [@deepspeed], Horovod
[@horovod], and Ray [@ray:2018] provide distributed training backends; Ray Tune
[@ray_tune] and Optuna [@optuna_2019] offer HPO frameworks; and
TensorBoard [@tensorboard], MLflow [@mlflow], and Weights \& Biases [@wandb]
support experiment tracking and visualization. HeAT [@heat],
AI4HPC [@ai4hpc], and perun [@perun] address, respectively, distributed tensor
operations on HPC systems, Computational Fluid Dynamics (CFD)-oriented AI and HPO
workflows on HPC systems, and performance or energy profiling, while
interLink [@interlink] enables cloud to HPC offloading of containerized workloads.

`itwinai` combines these capabilities into a single, configurable library: it
offers a plugin system for domain use cases, uniform configuration and logging
interfaces, support for multiple distributed training and HPO backends, built-in
profiling and scalability reporting, and portable deployment across laptops,
cloud clusters, and SLURM-based HPC systems via cloud to HPC offloading. This
makes it particularly suitable when users need to prototype and deploy AI
workflows that should run with minimal changes on heterogeneous HPC
infrastructures.

# Package features

The main features offered by the `itwinai` library are:

**Configuration for reproducible AI workloads**:
a declarative, hierarchical, composable, and CLI-overrideable
YAML-based configuration system that separates experimental
parameters from implementation code.

**Distributed training and inference**:
PyTorch-DDP [@torch_ddp], DeepSpeed [@deepspeed], Horovod [@horovod], and Ray [@ray:2018]
distributed ML training frameworks are supported.

**Hyperparameter optimization (HPO)**:
model performance can be improved by automatically traversing the hyperparameter space.

Ray integration provides two HPO strategies: (i) assigning multiple workers to
a single trial or (ii) running many trials
concurrently (\autoref{fig:hpo}).

![Conceptual representation of an HPO workflow in
itwinai.\label{fig:hpo}](img/hpo_itwinai_figure.pdf)

**Profilers**: `itwinai` integrates with multiple profilers, such as the
`py-spy` profiler [@py_spy] and the PyTorch Profiler, and also logs metrics
about training time, GPU utilization, and GPU power consumption.

**ML logs tracking**:  `itwinai` integrates with existing ML logging frameworks,
such as TensorBoard [@tensorboard], MLflow [@mlflow], Weights & Biases [@wandb],
and yProvML [@yprovml] logger, and provides a unified interface across all of
them through a thin abstraction layer.

**Offloading to HPC systems and cloud**: to benefit from both cloud and HPC, interLink
[@interlink] is used, which is a lightweight component to enable seamless offloading
of compute-intensive jobs from cloud to HPC, performing an automatic translation
from Kubernetes pods to SLURM jobs.

**Continuous integration and deployment**: `itwinai` includes extensive tests
(library and use cases). A [Dagger](https://dagger.io/) pipeline builds containers on
release, runs smoke tests on GitHub Actions (Azure runners: 4 CPUs, 16
GB)[^choosing_gh_runners], offloads distributed tests to HPC systems via interLink, and
publishes on success.

[^choosing_gh_runners]: GitHub hosted runners define the type of machine that
will process a job in your workflow.  [Find more
here](https://docs-internal.github.com/en/actions/how-tos/write-workflows/choose-where-workflows-run/choose-the-runner-for-a-job?utm_source=chatgpt.com)
(Accessed on 2025-08-14).

# Use-case integrations

There is a wide range of scientific use cases currently integrated with
`itwinai` via its plug-in architecture. Earth-observation plugins cover
hydrological forecasting, drought prediction, and climate/remote-sensing
pipelines; physics plugins include high-energy physics, radio astronomy, lattice
quantum chromodynamics (QCD), and gravitational-wave/glitch analysis. Packaging
these as `itwinai` plugins enables reproducible, shareable workflows that run
consistently on hardware ranging from personal computers to HPC systems. The
full list of `itwinai` plugins can be found at [this
link](https://itwinai.readthedocs.io/latest/getting-started/plugins-list.html).

# Reproducibility

The scalability results in this paper were obtained on the JUWELS Booster GPU
partition at the Jülich Supercomputing Centre [@JUWELS], a SLURM-managed system.
For the Virgo scalability study (\autoref{fig:speedup},
\autoref{fig:compvsother}), `itwinai` was driven by YAML configurations that
specify dataset, model, optimizer, batch size, and number of epochs. The exact
configuration files, SLURM scripts, and commands used to generate these figures
are stored in the `itwinai` repository inside
[`joss/reproducibility`](https://github.com/interTwin-eu/itwinai/tree/joss/joss/reproducibility).
The scalability-report and profiling workflow used to produce the plots is
described in the documentation [@itwinai_docs_scalability;
@itwinai_docs_profiling].

The same scalability-report machinery applies to single- and multi-node
training: on SLURM clusters, the configurations are launched via SLURM, while on
a single node (CPU or GPU) they can be run with a local backend and reduced
problem size. This allows readers with access to a comparable SLURM cluster to
reproduce the multi-node results and readers without SLURM to run a smaller,
single-node variant that produces reports and plots with the same structure as
those shown in this paper.

# Performance

`itwinai` provides tools to assess scalability and diagnose bottlenecks,
enabling efficient and accountable use of HPC resources. Two complementary
components are provided: scalability report generation and profiling.

## Scalability report

For data-parallel training, adding more workers improves throughput, but
as all-reduce communication costs grow, communication overhead eventually dominates,
causing scaling to level-off or even decline. The report characterizes this trade-off
across GPUs/nodes and backends, reporting wall-clock epoch time, relative speedup
(\autoref{fig:speedup}), GPU utilization (0–100%), energy (Wh), and
compute-versus-other time, including collective communication and memory
operations (\autoref{fig:compvsother}). Considered jointly, these metrics
identify the most efficient configuration and distribution strategy, rather than
relying on a single indicator. \autoref{fig:speedup} and
\autoref{fig:compvsother} show the scalability of the physics use case from
INFN[^infn] targeting gravitational-wave analysis at the Virgo[^virgo]
interferometer [@tsolaki_2025_15120028; @saether_scalabiliy_2025].

[^infn]: Istituto Nazionale di Fisica Nucleare
[infn.it](https://www.infn.it/en/) (Accessed on 2025-08-14).

[^virgo]: Virgo Collaboration [www.virgo-gw.eu](https://www.virgo-gw.eu/)
(Accessed on 2025-08-14).

![Relative speedup of average epoch time vs. number of workers for the Virgo use
case.\label{fig:speedup}](img/virgo_relative_epoch_time_speedup.svg){ width=70% }

![Proportion of time spent on computation versus other operations, such as
collective communication, in the Virgo use case, broken down by number of
workers and distributed framework.
\label{fig:compvsother}](img/virgo_computation_vs_other_plot.svg){ width=110% }

## Addressing bottlenecks via profiling

To explain why performance degrades, `itwinai` integrates low-overhead,
sample-based profiling (e.g., py-spy [@py_spy]) and summarizes flame-graph data
into actionable hotspots (e.g., data loading and I/O, kernel execution, host–device
transfer, communication). These summaries guide targeted remedies such as
adjusting batch size, data-loader parallelism, gradient accumulation, or
backend/collective settings.

# Outlook and future developments

`itwinai` provides ready-to-use ML tools that are applicable across a wide range
of scientific applications. The development of the library is continued through
projects ODISSEE[^odissee] and RI-SCALE[^ri-scale]. The
future developments include the integration of new scientific use cases,
exploring additional parallelism approaches, integrating advanced user
interfaces, and adding other EuroHPC systems and performance optimization
features.

[^odissee]: Online Data Intensive Solutions for Science in the Exabytes Era
(ODISSEE): [odissee-project.eu](https://www.odissee-project.eu/) (Accessed on 2025-08-14).

[^ri-scale]: RI-SCALE project: [riscale.eu](https://www.riscale.eu/) (Accessed on 2025-08-14).

# Acknowledgements

This work has been funded by the European Commission in the context of the
interTwin project, with Grant Agreement Number 101058386. In interTwin,
`itwinai` has been actively developed on the HPC systems at JSC, such as on the
HDF-ML and JUWELS Booster systems, and using EuroHPC resources, such as on the
Vega HPC system. `itwinai` is an open-source Python library primarily developed
by CERN, in collaboration with Forschungszentrum Jülich (FZJ).

# References
