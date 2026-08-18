Integrating a Use Case with Claude Code
=======================================

itwinai ships a `Claude Code <https://claude.com/claude-code>`__ skill that guides the
integration of existing scientific training code into an itwinai plugin. It covers the whole
path: scaffolding the plugin repository, porting the training loop, and then enabling
distributed training, logging, hyperparameter optimization, profiling and scalability reports.

The skill exists to lower the entry barrier. Everything it does can be done by hand by following
the rest of this documentation; it simply encodes the order of operations, the extension points,
and the mistakes that are easy to make and hard to diagnose.

.. note::

   The skill is optional. It is a convenience for users of Claude Code and is not required to
   use itwinai.

Installation
------------

The itwinai repository doubles as a Claude Code plugin marketplace. Install the plugin once and
it is available in **any** directory, including the plugin repository you are about to create —
you do not need to clone itwinai.

.. code-block:: text

   /plugin marketplace add interTwin-eu/itwinai
   /plugin install itwinai

Then start an integration with:

.. code-block:: text

   /itwinai:integrating-a-use-case

The skill also activates on its own when you ask Claude Code to integrate a training script into
itwinai, or to add HPO, logging or profiling to an existing plugin.

.. important::

   Installed plugins are **pinned snapshots**. They do not update automatically when this
   repository changes. Refresh the plugin from the ``/plugin`` menu periodically, especially
   after upgrading itwinai. The skill checks the installed itwinai version against its own and
   warns you when they diverge.

What it does
------------

The skill follows a fixed sequence, and each phase ends with a check that must pass before the
next begins.

.. list-table::
   :header-rows: 1
   :widths: 12 45 43

   * - Phase
     - Does
     - Check
   * - 0. Assess
     - Reads your training code, records where the model, optimizer, loop, data and
       hyperparameters live, and decides whether a custom trainer is needed at all
     - Inventory complete, approach chosen
   * - 1. Scaffold
     - Creates the plugin repository from
       `itwinai-plugin-template <https://github.com/interTwin-eu/itwinai-plugin-template>`__
       and configures packaging
     - Plugin installs and imports
   * - 2. Port
     - Moves the science under ``itwinai.plugins.<name>``, as either pipeline components or a
       :class:`~itwinai.torch.trainer.TorchTrainer` subclass
     - Import still succeeds, behaviour preserved
   * - 3. Wire
     - Writes ``config.yaml`` describing the pipeline
     - One-epoch non-distributed run, loss decreases
   * - 4. Capabilities
     - Adds logging, distributed training, profiling and HPO, **in that order**
     - Single-node multi-GPU SLURM job, metrics in MLflow
   * - 5. Scale
     - Runs a scaling test and generates a scalability report — only on explicit request
     - Report contains data

Two design points are worth knowing even if you never use the skill.

**The order in phase 4 is not arbitrary.** ``itwinai generate-scalability-report`` reads its
input back out of MLflow, so the logger must exist before profiling is enabled, and both must be
active during the scaling test. Enabling them out of order produces an empty report and no error
message. See :doc:`../how-it-works/scalability-report/scalability_report`.

**Most use cases do not need a custom trainer.** Of the plugins published so far, several supply
only a ``DataGetter`` and use the stock trainer. The skill defaults to that and only subclasses
:class:`~itwinai.torch.trainer.TorchTrainer` when the loss, optimizer or training step genuinely
requires it.

Scope
-----

In scope: PyTorch training code, the plugin layout, and the four capabilities above.

Out of scope: TensorFlow, inference pipelines, container builds, and Kubernetes deployment.
These are covered by the rest of this documentation.

The skill never submits a multi-node job without asking, and stops at a single-node job by
default.

Next steps
----------

- :doc:`../tutorials/claude-skill/integrate-a-new-use-case` — a complete worked example,
  integrating a neural-operator surrogate from scratch
- :doc:`plugins` — the plugin mechanism the skill targets
- :doc:`complete-workflow-example` — the configuration format, by hand
