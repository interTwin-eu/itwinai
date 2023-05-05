print(
"""
*************************
* Called AI/ML training *
*************************

- Split training dataset into: train, val, test
- Train on train split using given hyperparams
- Store trainig metrics
- Store validation metrics
- Save best model

"""
)

import time

input_text = '\nAI training completed'

# Append the output to a new file
with open('output_file.txt', 'a') as output_file:
    output_file.write(time.ctime(time.time()))
    output_file.write(input_text)

