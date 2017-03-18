# visualize-tsne
This small project is for creating 2d images out of your tsne vectors.   
The project was inspired by [Andrej Karpathy's blog post](http://cs.stanford.edu/people/karpathy/cnnembed/) on the visualization of CNNs using t-sne.  
(this guy is pretty sharp :wink: - you should definitely follow him! ) 


## some examples
<p align='center'>
<img src="/examples/mnist2d.jpg" alt="Image of mnist 2d image" width="300" height="300"/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="/examples/mnistscatter.jpg" alt="Image of mnist scatter image" width="300" height="300"/>
<br/>
mnist 2d image example
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
mnist scatter image example
</p>

## usage

### import into program
```python
from modules import TsneImage   
 
image = TsneImage()
image.path2data = 'data.hdf5'
image.load_data()
image.calculate_tsne()
image.create_image()
```
#### important! the module expects an hdf5 file with 2 datasets:   
 * urls - datasets which contain the path/url of each image    
 * vectors - dataset which contains the corresponding vector for each image.           
             make sure that they are both ordered alike
 * checkout this [hdf5 example](examples/create_hdf5_example.py)

#### another option is to load the data and urls explicitly:     
 * urls - create a np.asarray out of a url list and load to image.image_list    
 * vectors - create a np.ndarray of the vectors and load to image.data_vectors   
 
### from cmd
```
root@yonti:~/github/visualize-tsne$ python cmd.py -h
usage: cmd.py [-h] -d PATH2DATA [-n OUTPUT_NAME] [-t OUTPUT_TYPE]
              [-s OUTPUT_SIZE] [-i EACH_IMG_SIZE] [-c BG_COLOR] [--no-shuffle]
              [--no-sklearn] [--no-svd] [-b BATCH_SIZE]

t-SNE visualization using images

optional arguments:
  -h, --help            show this help message and exit
  -d PATH2DATA, --path2data PATH2DATA
                        Path to the hdf5 file
  -n OUTPUT_NAME, --output_name OUTPUT_NAME
                        output image name. Default is tsne_scatter/grid.jpg
  -t OUTPUT_TYPE, --output_type OUTPUT_TYPE
                        the type of the output images (scatter/grid)
  -s OUTPUT_SIZE, --output_size OUTPUT_SIZE
                        output image size (default=2500)
  -i EACH_IMG_SIZE, --img_size EACH_IMG_SIZE
                        each image size (default=50)
  -c BG_COLOR, --background BG_COLOR
                        choose output background color (black/white)
  --no-shuffle          use this flag if you don't want to shuffle
  --no-sklearn          use this flag if you don't want to use sklearn
                        implementation of tsne and you prepare the local
                        option
  --no-svd              it is better to reduce the dimension of long dense
                        vectors to a size of 50 or smallerbefore computing the
                        tsne.use this flag if you don't want to do so
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        for speed/memory size errors consider using just a
                        portion of your data (default=all)

root@yonti:~/github/visualize-tsne$ python cmd.py -d /home/data/data.hdf5 -i 50 -s 4000 -n test
```

### full usage options

```python
# the folowing have both getter and setter
image.path2doc # getter 
image.path2doc = '/home/data/data.hdf5' # setter -> expects string and correct path to an hdf5 file

image.output_img_name  #  getter
image.output_img_name = 'be_creative'  # expects string. default is 'tsne'
                                       # don't add the file type - jpg is set automatically
                                       # also the image type(scatter/grid) is added automatically
image.output_img_type  #  getter
image.output_img_type = 'grid' # expects string. default is 'scatter'. set grid to this way.

image.output_img_size  #  getter
image.output_img_size =  2500  # expects int. default is 2500. 
                               # all images are squared so it means 2500x2500 img.
                               # also the image type(scatter/grid) is added automatically

image.each_img_size    #  getter
image.each_img_size =  50      # expects int. default is 50. 
                               # the output looks better when constructed with squared images
                               # but can also handle rects
                               
image.image_list       #  getter
image.image_list = img_list    # expects numpy array of strings. 
                               # this is filled up automatically when load_data is called.
                               # set this explicitly only if you dont load your data from 
                               # an hdf5 file

image.data_vector      #  getter
image.data_vector = data       # expects numpy ndarray of dense vectors. 
                               # this is filled up automatically when load_data is called.
                               # set this explicitly only if you dont load your data from 
                               # an hdf5 file

image.batch_size       #  getter
image.batch_size =  5000       # expects int. default is 0 which means that all images are taken
                               # use this when you have memory issues. 
                               # it will shuffle your data and take only a subset in order to 
                               # compute the tsne. 

image.method       #  getter
image.method =  'maaten'       # expects string. default is 'sklearn'. 
                               # the other option is 'maaten'
                               # this sets the tsne method to sklearn.tsne vs python version
                               # of Maaten's tsne.
                               # i guess they both do the same but didn't fully check it 
                               # so i left it as an option

image.background_color         #  getter
image.background_color =  'white'  # expects string. default is 'black'. the other option is 'white'
                                        
image.tsne_vector      #  getter
image.tsne_vector = data       # expects numpy ndarray of dense 2d vectors. 
                               # this is filled up automatically when 
                               # image.calaculate_tsne is called.
                               # set this explicitly only if you have already the tsne vectors

# the followings are methods
image.load_data()  #  opens the file which path2file point to
                   #  fills image.data_vectors and image.image_list  
                   
image.calculate_tsne()  #  straight forward

image.create_image()  #  straight forward

#

 ```
## TODO list
- [x] upload my code
- [ ] upload more examples
  - [ ] cifar 10
  - [ ] cifar 100
  - [ ] imagenet
- [x] better documentation 
  - [x] add usage examples to readme
- [ ] \(optional) create server
