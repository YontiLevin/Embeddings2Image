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

## TODO list
- [x] upload my code
- [ ] upload more examples
  - [ ] cifar 10
  - [ ] cifar 100
  - [ ] imagenet
- [ ] better documentation 
  - [ ] add usage examples to readme
- [ ] \(optional) create server
