cwlVersion: v1.2
class: CommandLineTool

baseCommand: [conda, run, -p, /afs/cern.ch/work/a/azoechba/T6.5-AI-and-ML/ai/.venv-training, itwinai, train]



#doesn't solve the permission error
requirements:
  EnvVarRequirement:
    envDef:
      FILE_READ_BUFFER_SIZE: "10"

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

outputs: []


