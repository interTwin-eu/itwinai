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
        type: string?

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

  preprocessedDatasetPath:
    type: Directory
    outputSource: preprocess/preprocessedDatasetPath

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
        preprocessOutput: preprocessOutput
    out: [preprocessingStdout, preprocessedDatasetPath]

  training:
    run: ../../ai/training.cwl
    in:
        preprocessedDatasetPath: preprocess/preprocessedDatasetPath
        trainingConfig: trainingConfig
        trainingEnvironment: trainingEnvironment
        trainingCommand: trainingCommand
        preprocessingFlag: preprocess/preprocessingStdout
    out: [outputCheckpoint, mlLogs]