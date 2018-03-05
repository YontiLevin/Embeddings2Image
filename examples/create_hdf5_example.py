import h5py
import numpy as np

name_list = ['name1', 'name2']
image_names_array = np.asarray(name_list)
data_vectors = np.empty((2, 3000))

# write an hdf5 file
hf = h5py.File('data.hdf5', 'w')
hf.create_dataset('urls', data=image_names_array)
hf.create_dataset('vectors', data=data_vectors)
hf.close()

# read file
hf = h5py.File('data.hdf5', 'r')
url_list = hf['.']['urls']
vectors = hf['.']['vectors']
hf.close()
