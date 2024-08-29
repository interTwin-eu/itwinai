import os
import pandas as pd
import matplotlib.pyplot as plt

# Path to the main directory
main_dir = '/p/home/jusers/<username>/hdfml/ray_results/<specific run folder name>'

# List to store loss histories for each run
loss_histories = []
labels = []

# Iterate over all subdirectories in the main directory
for subdir in os.listdir(main_dir):
    if subdir:
        print("i am here")
        csv_file_path = os.path.join(main_dir, subdir, 'progress.csv')
        if os.path.exists(csv_file_path):
            df = pd.read_csv(csv_file_path)
            if 'loss' in df.columns:
                # loss_histories.append(df['training_iteration'])
                loss_histories.append(df['loss'])
                labels.append(f'Run {len(labels)+1}')
            else:
                print(f"'loss' column not found in {csv_file_path}")
        else:
            print(f"Fail {csv_file_path} does not exist")
print(loss_histories)

# Plot loss histories and save it
plt.figure(figsize=(10, 6))
for i, (loss) in enumerate(loss_histories):
    plt.plot(loss, label=labels[i])

plt.xlabel('Training Epoch')
plt.ylabel('Loss')
plt.title('HPO | Loss History for Each Run ')
plt.legend(labels)
plt.grid(True)
plt.show()

output_file = 'loss_history_plot.png'
plt.savefig(output_file)