#%%
import json
import matplotlib.pyplot as plt

experiment_folder = '/home/yalamaku/Documents/Thesis/Dataset_files/ZED_camera_dataset/Roboflow_annotated_data/CustomDataset/Model_weights_2/train_9'

def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

experiment_metrics = load_json_arr(experiment_folder + '/metrics.json')

plt.plot(
    [x['iteration'] for x in experiment_metrics if 'total_loss' in x], 
    [x['total_loss'] for x in experiment_metrics if 'total_loss' in x])
plt.plot(
    [x['iteration'] for x in experiment_metrics if 'Validation_loss' in x], 
    [x['Validation_loss'] for x in experiment_metrics if 'Validation_loss' in x])
plt.legend(['total_loss', 'validation_loss'], loc='upper left')
plt.show()
# %%
