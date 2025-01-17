# Test with `pytest`

Do this only if you are a developer wanting to test your code with pytest.

First, you need to create virtual environments both for torch and tensorflow,
following the instructions above, depending on the system that you are using
(e.g., JSC).

To select the name of the torch and tf environments in which the tests will be
executed you can set the following environment variables.
If these env variables are not set, the testing suite will assume that the
PyTorch environment is under
`.venv-pytorch` and the TensorFlow environment is under `.venv-tf`.

```bash
export TORCH_ENV="my_torch_env"
export TF_ENV="my_tf_env"
```

Functional tests (marked with `pytest.mark.functional`) will be executed under
`/tmp/pytest` location to guarantee isolation among tests.

To run functional tests use:

```bash
pytest -v tests/ -m "functional"
```

> [!NOTE]
> Depending on the system that you are using, we implemented a tailored Makefile
> target to run the test suite on it. Read these instructions until the end!

We provide some Makefile targets to run the whole test suite including unit, integration,
and functional tests. Choose the right target depending on the system that you are using:

Makefile targets:

- Juelich Supercomputer (JSC): `test-jsc`
- In any other case: `test`

For instance, to run the test suite on your laptop user:

```bash
make test
```
