from src.data_processing.DataPipelines import _create_training_feature_vectors
import pandas as pd


test_data = {
        "hi":[1, 2, 3, 4, 5, 6, 7, 9],
        "bye":[2, 3, 5, 4, 2, 9, 9, 9],
        "label":[0, 1, 2, 1, 2, 1, 0, 1]
    }

data = pd.DataFrame(test_data)
print(data)
data = _create_training_feature_vectors(data,backward_window=2, forward_window=1)
print(data)