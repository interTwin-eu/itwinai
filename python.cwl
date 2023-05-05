cwlVersion: v1.2
class: CommandLineTool

baseCommand: python


inputs:
  # filepath:
  #     type: File
  #     label: Aligned sequences in BAM format
  #     #format: edam:format_2572 fix an exact format later
  #     #path: /tmp/test.txt
  #     default: File(ai/ai-training.py)
  #     inputBinding:
  #       position: 1


  filepath:
    type: File
    inputBinding:
      position: 1    


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


  
