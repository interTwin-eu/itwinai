cwlVersion: v1.2
class: CommandLineTool

# Define the command that will be executed when the tool is run
baseCommand: [conda, run]

# Define requirements for the tool
# In this case, set an environment variable to try to solve a permission error
requirements:
  EnvVarRequirement:
    envDef:
      FILE_READ_BUFFER_SIZE: "10"

# Define inputs for the tool
inputs:

  trainingEnvironment:
    type: string
    inputBinding:
      position: 1
      prefix: -p

  trainingCommand:
    type: string
    inputBinding:
      position: 2
      prefix: itwinai

  trainingConfig:
    type: string
    inputBinding:
      position: 3
      prefix: --config

  preprocesseDatasetPath:
    type: string
    inputBinding:
      position: 4
      prefix: --input

  preprocessingFlag:
    type: File

# Define outputs for the tool
# In this case, there are no outputs
outputs: []
