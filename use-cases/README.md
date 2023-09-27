# itwinai use cases

Show how `itwinai` can be used. Each use case folder contains:

- `train.py`: entry point of training workflow
- `startscript`: file to execute the training workflow on a SLURM-based cluster
- `requirements.txt`: (optional) use case-specific requirements. can be installed with
  
  ```bash
  cd use/case/folder
  # After activating the correct environment...
  pip install -r requirements.txt
  ```
