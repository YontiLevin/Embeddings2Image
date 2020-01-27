from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE as SKLEARN_TSNE
from e2i.Maatens_tsne import TSNE as MATTENS_TSNE
from e2i.img_tools import get_image
from tqdm import tqdm
import cv2
from math import ceil
import h5py
import numpy as np
from umap import UMAP as UMAP_PROJECTION
from e2i.consts import *


class EmbeddingsProjector(object):
    def __init__(self):
        self._path2data = None
        self._output_img_name = 'Projection'
        self._output_img_type = SCATTER
        self._output_img_size = 2500
        self._each_img_size = 50
        self._background_color = BLACK
        self._shuffle = True
        self._method = UMAP
        self._batch_size = 0
        self._svd = True
        self._data_vectors = None
        self._projection_vectors = None
        self._image_list = None

    @property
    def path2data(self):
        return self._path2data

    @path2data.setter
    def path2data(self, data):
        assert data.endswith('.hdf5'), 'bad data file type. hdf5 expected!'
        self._path2data = data

    @property
    def output_img_name(self):
        return self._output_img_name

    @output_img_name.setter
    def output_img_name(self, name):
        self._output_img_name = name

    @property
    def output_img_type(self):
        return self._output_img_type

    @output_img_type.setter
    def output_img_type(self, img_type):
        self._output_img_type = img_type

    @property
    def output_img_size(self):
        return self._output_img_size

    @output_img_size.setter
    def output_img_size(self, img_size):
        self._output_img_size = img_size

    @property
    def image_list(self):
        return self._image_list

    @image_list.setter
    def image_list(self, img_list):
        self._image_list = img_list

    @property
    def each_img_size(self):
        return self._each_img_size

    @each_img_size.setter
    def each_img_size(self, img_size):
        self._each_img_size = img_size

    @property
    def args(self):
        args = {'path2data': self.path2data,
                'output_img_name': self.output_img_name,
                'output_img_type': self.output_img_type,
                'output_img_size': self.output_img_size,
                'each_img_size': self.each_img_size,
                'background_color': self._background_color,
                'shuffle': self._shuffle,
                'sklearn': True if self._method is SKLEARN else False,
                'batch_size': self._batch_size,
                'svd': self._svd}
        return args

    @args.setter
    def args(self, args):
        self.path2data = args.path2data
        self.output_img_name = args.output_name
        self.output_img_type = args.output_type
        self.output_img_name = args.output_size
        self.each_img_size = args.each_img_size
        self._background_color = args.bg_color
        self._shuffle = args.shuffle
        self._method = SKLEARN if args.sklearn else MAATEN
        self._svd = args.svd

    @property
    def data_vectors(self):
        return self._data_vectors

    @data_vectors.setter
    def data_vectors(self, vecs):
        self._data_vectors = vecs

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, size):
        self._batch_size = size
        if size > self.data_vectors.shape[0]:
            self.load_data()
        else:
            self._crop()

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, method):
        if method != UMAP:
            if len(self.data_vectors) > 10000:
                print("""
                WARNING! NOTICE THAT TSNE RUNS VERY SLOW AS THE NUMBER OF VECTORS GROW.\n
                IT\'S RECOMMENDED TO SET A BATCH SIZE SMALLER THAN 10K OR USE UMAP PROJECTION""")
        self._method = method

    @property
    def projection_vectors(self):
        return self._projection_vectors

    @projection_vectors.setter
    def projection_vectors(self, vectors):
        self._projection_vectors = vectors

    @property
    def background_color(self):
        return 0 if self._background_color is BLACK else 255

    @background_color.setter
    def background_color(self, color):
        if color.lower() not in [BLACK, WHITE]:
            print('color can only accept black or white')
        else:
            self._background_color = color.lower()

    def load_data(self, data_filename=None):
        self.path2data = data_filename or self._path2data
        with h5py.File(self.path2data, 'r') as hf:
            image_names = hf.get('urls')[()]
            data_vecs = np.array(hf.get('vectors')[()])
            assert len(image_names) == data_vecs.shape[0], 'img list length doen\'t match vector count'
            hf.close()
        self.data_vectors = data_vecs.astype(type('float_', (float,), {}))
        self.image_list = [im.decode("utf-8")  for im in image_names]
        self._crop()

    def shuffle(self):
        if self._shuffle and self._data_vectors is not None:
            perm = np.random.permutation(self._data_vectors.shape[0])
            self._data_vectors = self._data_vectors[perm]
            self._image_list = self._image_list[perm]

    def _crop(self):
        if self.batch_size > 0:
            self.shuffle()
            self._data_vectors = self._data_vectors[:self.batch_size]
            self._image_list = self._image_list[:self.batch_size]

    def _perform_svd(self):
        if self._svd and self.data_vectors.shape[1] > 50:
            print('dimension reduction using svd')
            print('dimension before: {}'.format(str(self.data_vectors.shape[1])))
            self.data_vectors = TruncatedSVD(n_components=50, random_state=0).fit_transform(self.data_vectors)
            print('dimension after: {}'.format(str(self.data_vectors.shape[1])))

    def calculate_projection(self):
        self._perform_svd()
        if self.method == SKLEARN:
            projection_vectors = SKLEARN_TSNE(n_components=2, perplexity=40, verbose=2).fit_transform(self.data_vectors)
        elif self.method == MAATEN:
            projection_vectors = MATTENS_TSNE(self.data_vectors, no_dims=2, initial_dims=self.data_vectors.shape[1],
                                        perplexity=40.0)
        else:
            projection_vectors = UMAP_PROJECTION(n_neighbors=5, min_dist=0.3).fit_transform(self.data_vectors)
        projection_vectors -= projection_vectors.min(axis=0)
        projection_vectors /= projection_vectors.max(axis=0)
        self.projection_vectors = projection_vectors

    def _shift(self):
        minx_idx, miny_idx = self.projection_vectors.argmin(axis=0)
        minx, _ = self.projection_vectors[minx_idx]
        _, miny = self.projection_vectors[miny_idx]
        self.projection_vectors -= [minx, miny]

    def create_image(self):
        self._shift()
        if self._output_img_type == GRID:
            constructed_image = self._grid()
        else:
            constructed_image = self._scatter()
        filename = '{}_{}.jpg'.format(self.output_img_name, self._output_img_type)
        cv2.imwrite(filename, constructed_image)

    def _grid(self):
        ratio = int(self.output_img_size / self.each_img_size)
        tsne_norm = self.projection_vectors[:, ] # / ratio
        used_imgs = np.equal(self.projection_vectors[:, 0], None)
        image = np.ones((self.output_img_size, self.output_img_size, 3)) * self.background_color
        for x in tqdm(range(ratio)):
            x0 = x * self.each_img_size
            x05 = (x + 0.5) * self.each_img_size
            y = 0
            while y < ratio:
                y0 = y * self.each_img_size
                y05 = (y + 0.5) * self.each_img_size
                tmp_tsne = tsne_norm - [x05, y05]
                tmp_tsne[used_imgs] = 99999  # don't use the same img twice
                tsne_dist = np.hypot(tmp_tsne[:, 0], tmp_tsne[:, 1])
                min_index = np.argmin(tsne_dist)
                used_imgs[min_index] = True
                img_path = self.image_list[min_index]
                small_img, x1, y1, dx, dy = get_image(img_path, self.each_img_size)
                if small_img is None:
                    continue
                if x < 1 and all(side < self.each_img_size for side in [x1, y1]):
                    self.each_img_size = min(x1, y1)
                    dx = int(ceil(x1 / 2))
                    dy = int(ceil(y1 / 2))
                image[y0 + dy:y0 + dy + y1,x0 + dx:x0 + dx + x1] = small_img
                y += 1

        return image

    def _scatter(self):
        image_num = len(self.image_list) #self.ratio ** 2
        maxx_idx, maxy_idx = self.projection_vectors.argmax(axis=0)
        tmp_vectors = self.projection_vectors * self.output_img_size  # self.each_img_size * 2
        maxx, _ = tmp_vectors[maxx_idx]
        _, maxy = tmp_vectors[maxy_idx]
        image = np.ones((self.output_img_size + self.each_img_size, self.output_img_size + self.each_img_size, 3)) * self.background_color
            # np.zeros((int(ceil(maxx)) + self.each_img_size,
            #               int(ceil(maxy)) + self.each_img_size, 3)) * self.background_color
        for i in tqdm(range(image_num)):
            img_path = self.image_list[i]
            x0, y0 = map(int, tmp_vectors[i])
            small_img, x1, y1, dx, dy = get_image(img_path, self.each_img_size)
            if small_img is None:
                continue
            if i < 1 and all(side < self.each_img_size for side in [x1,y1]):
                self.each_img_size = min(x1, y1)
                dx = int(ceil(x1 / 2))
                dy = int(ceil(y1 / 2))
            # test if there is an image there already
            if np.mean(image[y0 + dy:y0 + dy + y1, x0 + dx:x0 + dx + x1]) != self.background_color:
                continue
            image[y0 + dy:y0 + dy + y1, x0 + dx:x0 + dx + x1] = small_img

        return image


if __name__ == '__main__':
    from time import time
    start = time()
    image = EmbeddingsProjector()
    image.path2data = 'data/mnist_images.hdf5'
    image.load_data()
    image.each_img_size = 25
    # image.method = SKLEARN
    image.calculate_projection()
    image.output_img_name = 'data/mnist_nov_11_2019'
    image.output_img_type= 'scatter'
    image.create_image()
    stop = time()
    duration = stop - start
    print(f'process took {duration} seconds')


