import pickle
import pandas as pd

def read_csv(csv_fn, group_key):
    print 'Reading', csv_fn
    data = pd.read_csv(csv_fn)
    data.sort([group_key, 'T'], ascending = [True, True], inplace = True)
    return data

def write_pkl(data, pkl_fn, group_key):
    print 'Getting groups'
    g = data.groupby(group_key)

    print 'Grouping'
    by_group = dict()
    for k in g.groups.keys():
        by_group[k] = g.get_group(k).as_matrix()

    print 'Saving', pkl_fn
    pickle.dump(by_group, open(pkl_fn, 'wb'), -1)

    return by_group

def write_h5(data, h5_fn, group_key):
    print 'Getting groups'
    g = data.groupby(group_key)

    store = pd.HDFStore(h5_fn)

    print 'Grouping'
    by_group = dict()
    for k in g.groups.keys():
        store[str(k)] = g.get_group(k)

    store.close()

data = read_csv('train.csv',  'Device')
write_h5(data, 'train_by_dev.h5', 'Device')
#write_pkl(data, 'train_by_dev.pkl', 'Device')
data = None # free some memory

data = read_csv('test.csv',  'SequenceId')
write_h5(data, 'test_by_seq.h5', 'SequenceId')
#write_pkl(data, 'test_by_seq.pkl', 'SequenceId')
