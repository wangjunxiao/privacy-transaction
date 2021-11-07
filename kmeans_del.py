import numpy as np

def deletion(num_dels, raw_labels, raw_data):
    deleteindices = np.random.choice(len(raw_data), size=num_dels, replace=False) 
    c = 1
    for indice in deleteindices:
    #    print(f'processing request # {c}..., to delete {indice+1}/{len(raw_data)}')
        c += 1
    data = np.delete(raw_data, deleteindices, 0)
    labels = np.delete(raw_labels, deleteindices, 0)
    return labels, data