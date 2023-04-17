cwlVersion: v1.2
class: CommandLineTool

baseCommand: python


inputs:
  message:
    type: string
    # A default value that can be overridden, e.g. --message "/afs /eos"
    # Bind this message value as an argument to "cd".
    default: "/afs/cern.ch/work/a/azoechba/intertwin-t6.5/ai/ai-training.py"
    inputBinding:
      position: 1
outputs: []