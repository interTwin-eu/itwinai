cwlVersion: v1.2
class: Workflow

requirements:
  InlineJavascriptRequirement: {}

inputs:
    preprocessDir:    
        type: File
    AIDir:    
        type: File
        # A default value that can be overridden, e.g. --message "Hola mundo"
        #default: "/afs/cern.ch/work/a/azoechba/T6.5-AI-and-ML/ai/ai-training.py"
    flag:
      type: boolean
      default: true

outputs:
  output_file:
    type: File
    outputBinding:
      glob: output_file.txt


steps:
  preprocess:
    run: preprocess.cwl
    in:
        filepath: preprocessDir
    out:
      datasetPath: preprocesseDataset.txt

  AI:
    run: python.cwl
    in:
        filepath: AIDir
        datasetPath: preprocess/datasetPath
    out: []



outputs: []
  # stored_flag:
  #   type: string
  #   outputSource: preprocess/flag





