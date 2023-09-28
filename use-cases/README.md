# itwinai use cases

Show how `itwinai` can be used. Each use case folder contains:

- `pipeline.yaml`: textual description of the ML workflow for that use case 
- `train.py`: entry point of training workflow.
- `startscript`: file to execute the training workflow on a SLURM-based cluster.
- `requirements.txt`: (optional) use case-specific requirements. can be installed with:
  
  ```bash
  cd use/case/folder
  # After activating the correct environment...
  pip install -r requirements.txt
  ```

## How to run a use case

First, create the use case's Python environment (i.e., PyTorch or TensorFlow)
as described [in the main README](../README), and activate it. Then, install use case-specific
dependencies, if any:

```bash
pip install -r /use/case/path/requirements.txt
```

Alternatively, you can use the use case Docker image, if available.

Then, go to the use case's directory:

```bash
cd /use/case/path
```

From here you can run the use case (having activated the correct Python env):

```bash
# Locally
python train.py [OPTIONS...]

# With SLURM: stdout and stderr will be saved to job.out and job.err files
sbatch startscript
```
