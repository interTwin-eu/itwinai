cwlVersion: v1.2
class: Workflow

requirements:
  InlineJavascriptRequirement: {}

inputs:
    preprocessDir:    
        type: File
        # A default value that can be overridden, e.g. --message "Hola mundo"
        #default: "/afs/cern.ch/work/a/azoechba/intertwin-t6.5/use-cases/mnist/mnist-preproc.py"
    AIDir:    
        type: File
        # A default value that can be overridden, e.g. --message "Hola mundo"
        #default: "/afs/cern.ch/work/a/azoechba/intertwin-t6.5/ai/ai-training.py"


steps:

  preprocess:
    run: python.cwl
    in:
        filepath: preprocessDir
    #out: [flag]
    out: []

  AI:
    run: python.cwl
    in:
        filepath: AIDir
    out: []



outputs: []
  # stored_flag:
  #   type: string
  #   outputSource: preprocess/flag





