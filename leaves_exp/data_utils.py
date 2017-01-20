import numpy as np
import pandas as pd

# sklearn - fn to convert categorical strings to categorical ID's
def number_encode(labels):
    key = list(sorted(set(labels)))
    
    values = []
    print(len(labels))
    for each in labels:
        values.append(key.index(each))
    return np.array(values)

def load_data(filename):
    data_file = pd.read_csv(filename)
    print("Data File Columns:")
    print(data_file.columns)
    
    labels = np.array(data_file['species'])
    labels = number_encode(labels)
    features = data_file.drop(['id', 'species'], axis=1)
    #del features['id']
    #del features['species']
    
    print("Features Columns:")
    print(features.columns)

    dataset = {
        'labels' : labels,
        'features' : features.as_matrix(),
    } 
    return dataset
