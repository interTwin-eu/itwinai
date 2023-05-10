cwlVersion: v1.2
class: Workflow



inputs:
    preprocessEnvironment:
        type: string
    preprocessScript:
        type: string
    preprocessInput:    
        type: string
    preprocessOutput:    
        type: string

    trainingConfig:    
        type: string
    trainingEnvironment:    
        type: string
    trainingCommand:    
        type: string

outputs:
  preprocessingStdout:
    type: File
    outputSource: preprocess/preprocessingStdout

steps:
  preprocess:
    run: use-cases/mnist/mnist-preproc.cwl
    in:
        preprocessEnvironment: preprocessEnvironment
        preprocessScript: preprocessScript
        rawDatasetPath: preprocessInput
        preprocesseDatasetPath: preprocessOutput
    out: [preprocessingStdout]

  training:
    run: ai/src/training.cwl
    in:
        preprocesseDatasetPath: preprocessOutput
        trainingConfig: trainingConfig
        trainingEnvironment: trainingEnvironment
        trainingCommand: trainingCommand
        preprocessingFlag: preprocess/preprocessingStdout
    out: []