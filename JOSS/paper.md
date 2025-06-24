---
title: 'itwinai: A Python toolkit to accelerate AI and Machine Learning for scientists'
tags:
  - Python
  - Digital Twins
  - Distributed Training
  - Hyperparameter Optimization
  - High Performance Computing
authors:
  - name: Matteo Bunino
    orcid: 0009-0008-5100-9300
#    equal-contrib: true
    affiliation: 1 # (Multiple affiliations must be quoted)
  - given-names: Jarl
    dropping-particle: Sondre
    surname: Sæther
    orcid: 0009-0002-7971-2213
    affiliation: 1
  - given-names: Linus
    surname: Eickhoff
    affiliation: 1
  - given-names: Alex
    surname: Krochak
    affiliation: 2
  - given-names: Rakesh
    surname: Sarma
    orcid: 0000-0002-7069-4082
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 2
  - given-names: Mario
    surname: Girone
    affiliation: 1
  - given-names: Andreas
    surname: Lintermann
    orcid: 0000-0003-3321-6599
    affiliation: 2
affiliations:
 - name: CERN openlab, Switzerland
   index: 1
 - name: Forschungszentrum Jülich, Jülich Supercomputing Center, Germany
   index: 2
date: 17 June 2025
bibliography: paper.bib

---

# Summary

The use of AI in scientific applications has increased tremendously in the
last decade. The availability of large-scale datasets from both simulations
and experiments has allowed scientific applications to incorporate AI into their
workflows. Significantly, rapid development of hardware in the form of
accelerators, which are ideally suited to train AI and Machine Learning (ML)
models, has further contributed to the development and release of many AI models.
These rapid developments could potentially contribute to new scientific insights
through exploitation of datasets and also increase the computational efficiency
of the scientific models by inclusion of hybrid modelling approaches.

However, many of these developments could benefit from standardized software
development enabling better exploitation of computing resources. In this context,
the `itwinai` library is designed to help scientific researchers to develop their
AI and ML workflows by easing their development effort. `itwinai` offers a wide-array
of ML sub-routines which can be exploited by scientists to seamlessly scale their
workflows on state-of-the-art High Performance Computing (HPC) resources. With
`itwinai`, researchers are able to focus solely on model development, as
workflow deployment, HPC optimizations and distributed training strategies, is
handled by `itwinai`. A wide array of features is provided by the library,
which help the applications to increase their performance and accuracy.

# Statement of need

The efficient use of AI-centric accelerators, in particular Graphical Processing
Units (GPUs), is essential to maximize the potential of latest hardware development
and also increase the efficiency of the AI workflows. `itwinai` is a Python library
that provides advanced AI workflows to bridge the gap between AI, HPC and domain
science. These developments typically require significant effort from scientists;
however, `itwinai` simplifies this process by providing ready-to-use components
and tools.

`itwinai` is developed in a modular architecture, which allows users to easily import
and/or replace components in their ML pipelines. Furthermore, the application developers
can exploit `itwinai` plugins to continue their developments independently, while
exploiting features offered by the library. A comprehensive set of tutorials is provided
to allow new applications to easily adopt `itwinai`. Importantly, `itwinai` brings
together a multitude of functionalities (described in the next section) under one library,
significantly advancing the scientific applications that have already been integrated.

`itwinai` has been developed as part of the `interTwin` project to support various
Digital Twin (DT) applications, in scaling their AI workflows. These applications range
from high-energy-physics and radio astronomy to environmental sciences. The package is
part of a Digital Twin Engine (DTE), which forms the core engine to run various DT
applications on HPC and cloud resources. Although developed in the context of DTs, the
library is versatile to support any generic AI application. By lowering the barrier to
HPC and AI, `itwinai` empowers the scientific community to scale their developments
efficiently.

# Package Features

**Distributed training and inference**: `itwinai` supports multiple distributed training
frameworks, including PyTorch-DDP, DeepSpeed, Horovod and Ray. The performance of each of
these could vary, depending on the infrastructure or the use-case. `itiwnai` allows users
to benchmarks them for their applications and compare their performance.

**HyperParameter Optimization (HPO)**: 
HPO is the process of improving machine learning model performance on a given task by
systematically searching for the best hyperparameter values. Powered by Ray [@ray],
`itwinai` provides efficient HPO tools that scale the number of parallel trials and support
assigning multiple workers to individual trials for faster training. Furthermore, it
enables complex scheduling and resource management for large-scale, distributed
experiments.

**Profilers**:

**Loggers**:

# Performance

# Use-case integrations

Drought prediction: The `Hython` plugin for itwinai integrates distributed hydrological
modeling with our platform. The hydrological model WflowSBM predicts hydrological
variables over time but is computationally intensive. To address this, `Hython` [@hython] uses
LSTM models trained for sequence prediction, with custom data loading and data preparation.
Our itwinai plugin integrates `Hython` with `itwinai`, to enable distributed model training
with any of our supported distributed machine learning frameworks, including seamless
integration of Ray's HPO features with minimal changes required to the training loop and
extensive configuration options.

# Acknowledgements

# References

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }
