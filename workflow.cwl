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


steps:

  preprocess:
    run: preprocessing.cwl
    in:
        rawDatasetPath: preprocessInput
        preprocesseDatasetPath: preprocessOutput
    out: []

  training:
    run: training.cwl
    in:
        preprocesseDatasetPath: preprocessOutput
        configPath: config
    out: []




outputs: []
  # stored_flag:
  #   type: string
  #   outputSource: preprocess/flag





