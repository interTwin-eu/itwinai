# interTwin use cases integrated into itwinai

Show how `itwinai` can be used to support scientific use cases. Each use case folder contains:

- A YAML configuration file describing the ML workflows for that use case.
- A SLURM job script, used to execute the ML workflows on a SLURM-based cluster.
- `requirements.txt`: (optional) use case-specific requirements. can be installed with:
  
  ```bash
  cd use/case/folder
  # After activating the correct environment...
  pip install -r requirements.txt
  ```

## How to run a use case

First, create the use case's Python environment (i.e., PyTorch or TensorFlow)
as described [in the main README](../README.md#environment-setup), and activate it.
Then, install use case-specific dependencies, if any:

```bash
pip install -r /use/case/path/requirements.txt
```

Alternatively, you can use the use case Docker image, if available.

Then, go to the use case's directory:

```bash
cd use/case/path
```

From there you can run the use case following the instruction provided in the use case's folder.
