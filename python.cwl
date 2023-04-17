cwlVersion: v1.2
class: CommandLineTool

baseCommand: python


inputs:
  message:
    type: string
    # A default value that can be overridden, e.g. --message "Hola mundo"
    # Bind this message value as an argument to "echo".
    default: "/afs/cern.ch/work/a/azoechba/intertwin-t6.5/ai/ai-training.py"
    inputBinding:
      position: 1
outputs: []