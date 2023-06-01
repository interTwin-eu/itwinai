cwlVersion: v1.2
class: Workflow



inputs:
    preprocessEnvironment:
        type: Directory
    preprocessScript:
        type: File
    preprocessInput:    
        type: Directory?
    preprocessOutput:    
        type: Directory

    trainingConfig:    
        type: File
    trainingEnvironment:    
        type: Directory
    trainingCommand:    
        type: string

outputs:
  preprocessingStdout:
    type: File
    outputSource: preprocess/preprocessingStdout

  outputCheckpoint:
    type: Directory
    outputSource: training/outputCheckpoint
#   mlLogs:
#     type: File
#     outputSource: training/mlLogs

  mlLogs:
    type: Directory
    outputSource: training/mlLogs

steps:
  preprocess:
    run: mnist-preproc.cwl
    in:
        preprocessEnvironment: preprocessEnvironment
        preprocessScript: preprocessScript
        rawDatasetPath: preprocessInput
        preprocesseDatasetPath: preprocessOutput
    out: [preprocessingStdout]

  training:
    run: ../../ai/training.cwl
    in:
        preprocessedDatasetPath: preprocessOutput
        trainingConfig: trainingConfig
        trainingEnvironment: trainingEnvironment
        trainingCommand: trainingCommand
        preprocessingFlag: preprocess/preprocessingStdout
    out: [outputCheckpoint, mlLogs]