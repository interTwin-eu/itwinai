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
  output_parameter:
    type: File
    outputSource: preprocess/output_parameter

steps:
  preprocess:
    run: preprocessing.cwl
    in:
        rawDatasetPath: preprocessInput
        preprocesseDatasetPath: preprocessOutput
    out: [output_parameter]

  training:
    run: training.cwl
    in:
        preprocesseDatasetPath: preprocessOutput
        configPath: config
        #preprocessingFlag: preprocess/output_parameter
    out: []




#outputs: []
  # stored_flag:
  #   type: string
  #   outputSource: preprocess/flag





