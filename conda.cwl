cwlVersion: v1.2
class: CommandLineTool

baseCommand: conda
#mamba activate


inputs:
  # filepath:
  #     type: File
  #     label: Aligned sequences in BAM format
  #     #format: edam:format_2572 fix an exact format later
  #     #path: /tmp/test.txt
  #     default: File(ai/ai-training.py)
  #     inputBinding:
  #       position: 1


  commandName:
    type: string
    inputBinding:
      position: 1    

  argumentName:
    type: string
    inputBinding:
      position: 2


outputs: []
#   flag:    
#     type: boolean
#     # A default value that can be overridden, e.g. --message "Hola mundo"
#     #outputBinding: true
#     value: true
#   flag:
#     type: string
#     outputBinding:
#       loadContents: abcd


  
