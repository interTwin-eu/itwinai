cwlVersion: v1.2
class: Workflow

requirements:
  InlineJavascriptRequirement: {}

inputs: []
outputs: []


steps:
  uppercase:
    run: /afs/cern.ch/work/a/azoechba/intertwin-t6.5/ai/ai-training.py
    in: []
    out: []