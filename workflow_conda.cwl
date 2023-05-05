cwlVersion: v1.2
class: Workflow

requirements:
  InlineJavascriptRequirement: {}

inputs:
    initCommand:    
        type: string
    initArgument:    
        type: string

    activateCommand:    
        type: string
    activateArgument:    
        type: string
        # A default value that can be overridden, e.g. --message "Hola mundo"
        #default: "/afs/cern.ch/work/a/azoechba/T6.5-AI-and-ML/ai/ai-training.py"


steps:

  activateConda:
    run: conda.cwl
    in:
        commandName: activateCommand
        argumentName: activateArgument
    out: []

  initConda:
    run: conda.cwl
    in:
        commandName: initCommand
        argumentName: initArgument
    out: []







outputs: []
  # stored_flag:
  #   type: string
  #   outputSource: preprocess/flag





