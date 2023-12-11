import pytest

pytest.PIPE_LIST_YAML = """
my-list-pipeline:
  class_path: itwinai.pipeline.Pipeline
  init_args:
    steps:
      - class_path: itwinai.tests.dummy_components.FakePreproc
        init_args:
          max_items: 32
          name: my-preproc

      - class_path: itwinai.tests.dummy_components.FakeTrainer
        init_args:
          lr: 0.001
          batch_size: 32
          name: my-trainer

      - class_path: itwinai.tests.dummy_components.FakeSaver
        init_args:
          save_path: ./some/path
          name: my-saver
"""

pytest.PIPE_DICT_YAML = """
my-dict-pipeline:
  class_path: itwinai.pipeline.Pipeline
  init_args:
    steps:
      preproc-step:
        class_path: itwinai.tests.dummy_components.FakePreproc
        init_args:
          max_items: 32
          name: my-preproc

      train-step:
        class_path: itwinai.tests.dummy_components.FakeTrainer
        init_args:
          lr: 0.001
          batch_size: 32
          name: my-trainer

      save-step:
        class_path: itwinai.tests.dummy_components.FakeSaver
        init_args:
          save_path: ./some/path
          name: my-saver
"""

pytest.NESTED_PIPELINE = """
some:
  field:
    nst-pipeline:
      class_path: itwinai.pipeline.Pipeline
      init_args:
        steps:
          - class_path: itwinai.tests.FakePreproc
            init_args:
              max_items: 32
              name: my-preproc

          - class_path: itwinai.tests.FakeTrainer
            init_args:
              lr: 0.001
              batch_size: 32
              name: my-trainer

          - class_path: itwinai.tests.FakeSaver
            init_args:
              save_path: ./some/path
              name: my-saver
"""
