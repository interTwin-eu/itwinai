cwlVersion: v1.2              # Specifies the version of the Common Workflow Language (CWL) being used
class: CommandLineTool       

baseCommand: [conda, run]
# The command to be executed by the tool. It runs the 'conda' command with 'run' subcommand, 
# then sets the path to the virtual environment to be used, and finally runs the 'mnist-preproc.py' 
# script using the 'python' interpreter.

requirements:
  EnvVarRequirement:
    envDef:
      FILE_READ_BUFFER_SIZE: "10"
# The following requirement sets the environment variable 'FILE_READ_BUFFER_SIZE' to a value of 10.
# This requirement defines an environment variable requirement, specifying the value of the environment
# variable to be set.

stdout: ./logs/mnist-preproc-stdout.txt
# Specifies that the standard output of the command will be redirected to the file 'mnist-preproc-stdout.txt'
# located in the 'logs' directory.

inputs:
  preprocessEnvironment:
    type: Directory
    inputBinding:
      position: 1
      prefix: -p
      # Specifies that the 'preprocessEnvironment' input is a directory.
      # The 'inputBinding' section provides additional information on how this input should be passed
      # to the command line tool. 'position' specifies the position of the argument in the command line,
      # and 'prefix' specifies the prefix to be used for the argument.

  preprocessScript:
    type: File
    inputBinding:
      position: 2
      prefix: python
      # Specifies that the 'preprocessScript' input is a file.
      # 'position' and 'prefix' are used to pass this input to the command line tool.

  rawDatasetPath:
    type: Directory?       
    inputBinding:          
      position: 3          
      prefix: --input      
      # Specifies that the 'rawDatasetPath' input is an optional directory.
      # 'position' and 'prefix' are used to pass this input to the command line tool.

  preprocessOutput:
    type: string?           
    inputBinding:          
      position: 4          
      prefix: --output     
      # Specifies that the 'preprocessOutput' input is an optional string.
      # 'position' and 'prefix' are used to pass this input to the command line tool.

outputs:
  preprocessingStdout:
    type: stdout           
    # Specifies that the 'preprocessingStdout' output is the standard output of the command.

  preprocessedDatasetPath:
    type: Directory
    outputBinding:
      glob: "use-cases/mnist/data/preproc-images"
      # Specifies that the 'preprocessedDatasetPath' output is a directory.
      # 'glob' specifies the glob pattern to find the output directory.


      
