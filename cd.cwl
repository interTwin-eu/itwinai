cwlVersion: v1.2
class: CommandLineTool

baseCommand: cd


inputs:
  message:
    type: string
    # A default value that can be overridden, e.g. --message "Hola mundo"
    default: "/afs/cern.ch/work/a/azoechba/intertwin-t6.5"
    # Bind this message value as an argument to "echo".
    inputBinding:
      position: 1
outputs: []