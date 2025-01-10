# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import pytest

pytest.PIPE_LIST_YAML = """
my-list-pipeline:
  _target_: itwinai.pipeline.Pipeline
  steps:
    - _target_: itwinai.tests.dummy_components.FakePreproc
      max_items: 32
      name: my-preproc

    - _target_: itwinai.tests.dummy_components.FakeTrainer
      lr: 0.001
      batch_size: 32
      name: my-trainer

    - _target_: itwinai.tests.dummy_components.FakeSaver
      save_path: ./some/path
      name: my-saver
"""

pytest.PIPE_DICT_YAML = """
my-dict-pipeline:
  _target_: itwinai.pipeline.Pipeline
  steps:
    preproc-step:
      _target_: itwinai.tests.dummy_components.FakePreproc
      max_items: 33
      name: my-preproc

    train-step:
      _target_: itwinai.tests.dummy_components.FakeTrainer
      lr: 0.001
      batch_size: 32
      name: my-trainer

    save-step:
      _target_: itwinai.tests.dummy_components.FakeSaver
      save_path: ./some/path
      name: my-saver
"""

pytest.NESTED_PIPELINE = """
some:
  field:
    my-nested-pipeline:
      _target_: itwinai.pipeline.Pipeline
      steps:
        - _target_: itwinai.tests.FakePreproc
          max_items: 32
          name: my-preproc

        - _target_: itwinai.tests.FakeTrainer
          lr: 0.001
          batch_size: 32
          name: my-trainer

        - _target_: itwinai.tests.FakeSaver
          save_path: ./some/path
          name: my-saver
"""

pytest.INTERPOLATED_VALUES_PIPELINE = """
max_items: 33
name: my-trainer
my-interpolation-pipeline:
  _target_: itwinai.pipeline.Pipeline
  steps:
    - _target_: itwinai.tests.dummy_components.FakePreproc
      max_items: ${max_items}
      name: my-preproc

    - _target_: itwinai.tests.dummy_components.FakeTrainer
      lr: 0.001
      batch_size: 32
      name: ${name}
"""
