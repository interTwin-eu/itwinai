cwlVersion: v1.2
class: Workflow

inputs:
    preprocessEnvironment:   # Input directory for the preprocess environment
        type: Directory
    preprocessScript:   # Input file for the preprocess script
        type: File
    preprocessInput:   # Optional input directory for preprocess
        type: Directory?
    preprocessOutput:   # Optional output string for preprocess
        type: string?

    trainingConfig:   # Input file for the training configuration
        type: File
    trainingEnvironment:   # Input directory for the training environment
        type: Directory
    trainingCommand:   # Input string for the training command
        type: string

outputs:
  preprocessingStdout:   # Output file for the preprocessing stdout
    type: File
    outputSource: preprocess/preprocessingStdout

  outputCheckpoint:   # Output directory for the trained model checkpoint
    type: Directory
    outputSource: training/outputCheckpoint

  preprocessedDatasetPath:   # Output directory for the preprocessed dataset
    type: Directory
    outputSource: preprocess/preprocessedDatasetPath

  mlLogs:   # Output directory for the machine learning logs
    type: Directory
    outputSource: training/mlLogs

steps:
  preprocess:   # Step for preprocessing
    run: mnist-preproc.cwl
    in:
        preprocessEnvironment: preprocessEnvironment
        preprocessScript: preprocessScript
        rawDatasetPath: preprocessInput
        preprocessOutput: preprocessOutput
    out: [preprocessingStdout, preprocessedDatasetPath]

  training:   # Step for training
    run: ../../ai/training.cwl
    in:
        preprocessedDatasetPath: preprocess/preprocessedDatasetPath
        trainingConfig: trainingConfig
        trainingEnvironment: trainingEnvironment
        trainingCommand: trainingCommand
        preprocessingFlag: preprocess/preprocessingStdout
    out: [outputCheckpoint, mlLogs]
