import os
from tqdm import tqdm
from e2i import EmbeddingsProjector
import numpy as np
import h5py

# paths
imgs_dir = 'imgs'
data_path = 'data.hdf5'
output_path = 'output_plot'

# get image embeddings and path to each image
embeddings = np.random.randn(10, 37)
name_list = []
for i in tqdm(range(10)):
    path = os.path.join(imgs_dir, '%d.png' % i)
    name_list.append(path)

# write an hdf5 file
with h5py.File(data_path, 'w') as hf:
    hf.create_dataset('urls', data=np.asarray(name_list).astype("S"))
    hf.create_dataset('vectors', data=embeddings)
    hf.close()

# compute embeddings and create output plot
image = EmbeddingsProjector()
image.path2data = data_path
image.background_color = 'white'
image.load_data()
image.each_img_size = 100
image.calculate_projection()
image.output_img_name = output_path
image.output_img_type = 'scatter'
image.create_image()

print('done!')
