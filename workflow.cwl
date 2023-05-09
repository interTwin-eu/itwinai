cwlVersion: v1.2
class: Workflow

requirements:
  InlineJavascriptRequirement: {}

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
    run: preprocessing.cwl
    in:
        rawDatasetPath: preprocessInput
        preprocesseDatasetPath: preprocessOutput
    out: [preprocessingStdout]

  training:
    run: training.cwl
    in:
        preprocesseDatasetPath: preprocessOutput
        configPath: config
        preprocessingFlag: preprocess/preprocessingStdout
    out: []