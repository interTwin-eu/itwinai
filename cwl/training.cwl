cwlVersion: v1.2
class: CommandLineTool

# Define the command that will be executed when the tool is run
baseCommand: [conda, run, -p, /afs/cern.ch/work/a/azoechba/T6.5-AI-and-ML/ai/.venv-training, itwinai, train]

# Define requirements for the tool
# In this case, set an environment variable to try to solve a permission error
requirements:
  EnvVarRequirement:
    envDef:
      FILE_READ_BUFFER_SIZE: "10"

# Define inputs for the tool
inputs:
  configPath:
    type: string
    inputBinding:
      position: 1
      prefix: --config

  preprocesseDatasetPath:
    type: string
    inputBinding:
      position: 2
      prefix: --input

  preprocessingFlag:
    type: File

# Define outputs for the tool
# In this case, there are no outputs
outputs: []
