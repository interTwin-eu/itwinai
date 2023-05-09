cwlVersion: v1.2
class: CommandLineTool

baseCommand: [conda, run, -p, /afs/cern.ch/work/a/azoechba/T6.5-AI-and-ML/use-cases/mnist/.venv-preproc, python, /afs/cern.ch/work/a/azoechba/T6.5-AI-and-ML/use-cases/mnist/mnist-preproc.py]



#doesn't solve the permission error
requirements:
  EnvVarRequirement:
    envDef:
      FILE_READ_BUFFER_SIZE: "10"

inputs:
  rawDatasetPath:
    type: string
    inputBinding:
      position: 1
      prefix: --input

  preprocesseDatasetPath:
    type: string
    inputBinding:
      position: 2
      prefix: --output

outputs:
  preprocessingStdout:
    type: stdout



