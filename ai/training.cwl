cwlVersion: v1.2        # Specifies the version of the Common Workflow Language (CWL) being used
class: CommandLineTool

baseCommand: [conda, run]
# The command to be executed by the tool. It runs the 'conda' command with 'run' subcommand.

requirements:
  EnvVarRequirement:
    envDef:
      FILE_READ_BUFFER_SIZE: "10"
# The following requirement sets the environment variable 'FILE_READ_BUFFER_SIZE' to a value of 10.
# This requirement defines an environment variable requirement, specifying the value of the environment
# variable to be set.

inputs:
  trainingEnvironment:
    type: Directory
    inputBinding:
      position: 1
      prefix: -p
      # Specifies that the 'trainingEnvironment' input is a directory.
      # The 'inputBinding' section provides additional information on how this input should be passed
      # to the command line tool. 'position' specifies the position of the argument in the command line,
      # and 'prefix' specifies the prefix to be used for the argument.

  trainingCommand:
    type: string
    inputBinding:
      position: 2
      prefix: itwinai
      # Specifies that the 'trainingCommand' input is a string.
      # 'position' and 'prefix' are used to pass this input to the command line tool.

  trainingConfig:
    type: File
    inputBinding:
      position: 3
      prefix: --config
      # Specifies that the 'trainingConfig' input is a file.
      # 'position' and 'prefix' are used to pass this input to the command line tool.

  preprocessedDatasetPath:
    type: Directory
    inputBinding:
      position: 4
      prefix: --train-dataset
      # Specifies that the 'preprocessedDatasetPath' input is a directory.
      # 'position' and 'prefix' are used to pass this input to the command line tool.

  preprocessingFlag:
    type: File
    # Specifies that the 'preprocessingFlag' input is a file.

outputs:
    outputCheckpoint:
     type: Directory
     outputBinding:
      glob: checkpoints
      # Specifies that the 'outputCheckpoint' output is a directory.
      # 'glob' specifies the glob pattern to find the output directory.

    mlLogs:
     type: Directory
     outputBinding:
      glob: "ml-logs"
      # Specifies that the 'mlLogs' output is a directory.
      # 'glob' specifies the glob pattern to find the output directory.

