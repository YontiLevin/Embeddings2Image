import argparse
from e2i.modules import EmbeddingsProjector


def user_inputs():

    desc = 'Creating 2d images out of your images embeddings vectors'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-d', '--path2data', type=str, default=None, dest='path2data',
                        help='Path to the hdf5 file', required=True)
    parser.add_argument('-n', '--output_name', type=str, default='tsne', dest='output_name',
                        help='output image name. Default is tsne_scatter/grid.jpg')
    parser.add_argument('-t', '--output_type', type=str, default='scatter', dest='output_type',
                        help='the type of the output images (scatter/grid)')
    parser.add_argument('-s', '--output_size', type=int, default=2500, dest='output_size',
                        help='output image size (default=2500)')
    parser.add_argument('-i', '--img_size', type=int, default=50, dest='each_img_size',
                        help='each image size (default=50)')
    parser.add_argument('-c', '--background', type=str, default='black', dest='bg_color',
                        help='choose output background color (black/white)')
    parser.add_argument('--no-shuffle', dest='shuffle', default=True, action='store_false',
                        help='use this flag if you don\'t want to shuffle')
    parser.add_argument('--no-sklearn', dest='sklearn', default=True, action='store_false',
                        help='use this flag if you don\'t want to use sklearn implementation of tsne '
                             'and you prepare the local option')
    parser.add_argument('--no-svd', dest='svd', default=True, action='store_false',
                        help='it is better to reduce the dimension of long dense vectors to a size of 50 or smaller'
                             'before computing the tsne.'
                             'use this flag if you don\'t want to do so')
    parser.add_argument('-b', '--batch_size', type=int, default=0, dest='batch_size',
                        help='for speed/memory size errors consider using just a portion of your data (default=all)')
    return parser.parse_args()


if __name__ == "__main__":

    args = user_inputs()
    ep = EmbeddingsProjector()
    ep.args = args
    ep.load_data()
    ep.calculate_projection()
    ep.create_image()

    print('Done!')


