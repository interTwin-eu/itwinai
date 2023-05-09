cwlVersion: v1.2
class: Workflow

inputs:
    preprocessInput:    
        type: string
    preprocessOutput:    
        type: string
    config:    
        type: string

outputs:
  preprocessingStdout:
    type: File
    outputSource: preprocess/preprocessingStdout

steps:
  preprocess:
    run: cwl/preprocessing.cwl
    in:
        rawDatasetPath: preprocessInput
        preprocesseDatasetPath: preprocessOutput
    out: [preprocessingStdout]

  training:
    run: cwl/training.cwl
    in:
        preprocesseDatasetPath: preprocessOutput
        configPath: config
        preprocessingFlag: preprocess/preprocessingStdout
    out: []