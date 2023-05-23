---
layout: default
title: Concepts
nav_order: 4
---

# Concepts
<!-- {: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---
-->

Here we presents the key concepts on which `itwinai` is based.

## Workflow

We define a workflow as a directed acyclic graph (DAG) of data processing
operations, in which each step can have multiple inputs and outputs, and
each input or output is a dataset.

![image](img/Workflow%20DAG%20concept.png)

In the picture above, the yellow boxes with numbers represent the steps
in the example workflow, whereas the blue cylinders represent data
(e.g., dataset, configuration file).

Each step runs in *isolation* from the others, and data is the *interface*.
Isolation can be guaranteed by executing each step in a Docker container or
in a separate Python virtual environment.
