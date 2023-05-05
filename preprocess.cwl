cwlVersion: v1.2
class: CommandLineTool

baseCommand: python

inputs:
  filepath:
    type: File
    inputBinding:
      position: 1

outputs:
  output_file:
    type: File
    outputBinding:
      glob: preprocesseDataset.txt