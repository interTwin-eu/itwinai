cwlVersion: v1.2

# What type of CWL process we have in this document.
class: CommandLineTool
# This CommandLineTool executes the linux "echo" command-line tool.

inputs:

  script:
    type: File
    inputBinding:
      position: 1
    default:
      class: File
      path: ai/ai-training.py


outputs: []


requirements:
  InitialWorkDirRequirement:
    listing:
      - entry: $(inputs.script)
        entryname: ai/ai-training.py