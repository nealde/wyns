
def data_to_h5(directory, filename, splits=5):
    '''This function turns our twitter data from a csv into an h5 file that is split into train/test sets with 5-fold cross validation
    for the training set.'''
    from sklearn.model_selection import KFold, train_test_split
    import numpy as np
    import pandas as pd
    import h5py 
    data = pd.read_csv(directory+filename,encoding = "ISO-8859-1")
    # get unique labels
    dt = np.array(data['existence'])
    unique = []
    for i in dt:
        if i not in unique:
            unique.append(i)
    ohe = np.zeros((dt.shape[0],3))
    for i, j in enumerate(dt):
        if j == 'Yes' or j == 'Y':
            ohe[i,0] = 1
        elif j == 'No' or j=='N':
            ohe[i,1] = 1
        else:
            ohe[i,2] = 1
    x = np.array(data['tweet'])
    y = ohe
    xx, xt, yy, yt = train_test_split(x, y, test_size=.2)
    kf = KFold(n_splits=splits, shuffle=True)
    a = []
    b = []
    for train_index, test_index in kf.split(xx):
        a.append([xx[train_index],xx[test_index]])
        b.append([yy[train_index], yy[test_index]])
    with h5py.File("twitter_data.hdf5", "w") as f:
        for i in range(len(a)):
            dset = f.create_dataset("x%i" % i, data=[aa.encode('utf8') for aa in a[i][0]])
            dset = f.create_dataset("xt%i" % i, data=[aa.encode('utf8') for aa in a[i][1]])
            dset = f.create_dataset("y%i" % i, data=b[i][0])
            dset = f.create_dataset("yt%i" % i, data=b[i][1])
        dset = f.create_dataset("ytest", data=yt)
        dset = f.create_dataset("xtest", data=[aa.encode('utf8') for aa in xt])
    print('done!')
    return
data_to_h5('../../core/data/','tweet_global_warming.csv')