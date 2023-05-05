cwlVersion: v1.2
class: CommandLineTool

baseCommand: python

inputs:
  filepath:
    type: File
    inputBinding:
      position: 1
  
  preprocesseDatasetPath:
    type: File
    inputBinding:
      position: 2
      prefix: --input

outputs:
  output_file:
    type: File
    outputBinding:
      glob: output_file.txt