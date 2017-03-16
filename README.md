# visualize-tsne
This small project is for creating 2d images of your tsne.   
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
from visualize-tsne import VisuTsne   
 
tsne_image = VisuTsne()
tsne_image.load_data(data_filename='data.hdf5')
tsne_image.output_filename('tsne.jpg')
tsne_image.create_image()
tsne_image.close()
```

### from cmd
```
root@yonti:~/github/visualize-tsne$ python yonti_tsne.py -h
usage: visualize_tsne.py [-h] [-n NDARRAY] [-u IMAGES_LIST] [-f HDF5]
                         [-o OUTPUT_FILENAME] [-s EMBEDDING_SIZE] [-i IMAGE_SIZE]
                         [-t METHOD]

root@yonti:~/github/visualize-tsne$ python visualize_tsne.py -f data2.hdf5 -i 50 -s 4000 -o data2.jpg 
```

## TODO list
- [ ] upload my code
- [ ] upload more examples
  - [ ] cifar 10
  - [ ] cifar 100
  - [ ] imagenet
- [ ] better documentation 
  - [ ] add usage examples to readme
- [ ] \(optional) create server
