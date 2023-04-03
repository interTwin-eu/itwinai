# interTwin: an end2end prototype

The goal is to simulate the whole interTwin software stack,
starting from a minimal prototype, with a special focus on
AI/ML workflows (T6.5).

## Workflow

1. Preprocessing: inside `./preproc`
2. AI/ML workflows (T6.5): inside `./ai`
3. SQAaaS: inside `./sqaaas`

## Use cases

In order of complexity;

1. MNIST CV tasks (e.g., generation, classification)
2. (Simplified) CERN use case (i.e., fastSim)
3. EO ML task: pre-processing openEO. An example is cloud removal.
Ask Ilaria for ideas.
4. Streaming: online ML. Collaborate with Virgo (?).

## Tasks

- Containers (backbone) and link with the infrastructure
(e.g., Kubernetes, TOSCA)
- AI workflows logic
  - Training/validation
  - Integration with `mlflow` logging
  - Define a standard configuration for the task (e.g., NN arch,
  learning rate), which is loaded from JSON files.

- Use case-specific pre-processing.