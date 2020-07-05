
# Colorizer

Based on [DeOldify](https://github.com/jantic/DeOldify)

**Quick Start**: The easiest way to colorize images using DeOldify (for free!) is here: [DeOldify Image Colorization on DeepAI](https://deepai.org/machine-learning-model/colorizer)

The **most advanced** version of DeOldify image colorization is available here, exclusively.  Try a few images for free! [MyHeritiage In Color](https://www.myheritage.com/incolor)

----------------------------

Image (artistic) [<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://colab.research.google.com/github/jantic/DeOldify/blob/master/ImageColorizerColab.ipynb) |
Video [<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://colab.research.google.com/github/jantic/DeOldify/blob/master/VideoColorizerColab.ipynb)

## Table of Contents
- [About DeOldify](#about-deoldify)
- [Getting Started Yourself](#getting-started-yourself)
    - [Easiest Approach](#easiest-approach)
    - [Your Own Machine](#your-own-machine-not-as-easy)
- [Docker](#docker)
- [Pretrained Weights](#pretrained-weights)

## About DeOldify

Simply put, the mission of this project is to colorize and restore old images and film footage.
We'll get into the details in a bit, but first let's see some pretty pictures and videos! 


## Getting Started Yourself

### Easy Install

You should now be able to do a simple install with Anaconda. Here are the steps:

Open the command line and navigate to the root folder you wish to install.  Then type the following commands 

```console
conda env create -f environment.yml
```

Then start running with these commands:

```console
source activate deoldify
python app.py
```


Calling the API for image processing
```console
curl -X POST "http://MY_SUPER_API_IP:5000/process" -H "accept: image/png" -H "Content-Type: application/json" -d "{\"source_url\":\"http://www.afrikanheritage.com/wp-content/uploads/2015/08/slave-family-P.jpeg\", \"render_factor\":35}" --output colorized_image.png
```


### Installation Details

This project is built around the wonderful Fast.AI library.  Prereqs, in summary:

- **Fast.AI 1.0.51** (and its dependencies).  If you use any higher version you'll see grid artifacts in rendering and tensorboard will malfunction. So yeah...don't do that.
- **PyTorch 1.0.1** Not the latest version of PyTorch- that will not play nicely with the version of FastAI above.  Note however that the conda install of FastAI 1.0.51 grabs the latest PyTorch, which doesn't work.  This is patched over by our own conda install but fyi.
- **Tensorboard** (i.e. install Tensorflow) and **TensorboardX** (https://github.com/lanpa/tensorboardX).  I guess you don't *have* to but man, life is so much better with it.  FastAI now comes with built in support for this- you just  need to install the prereqs: `conda install -c anaconda tensorflow-gpu` and `pip install tensorboardX`
- **ImageNet** – Only if you're training, of course. It has proven to be a great dataset for my purposes.  http://www.image-net.org/download-images

## Pretrained Weights

To start right away on your own machine with your own images or videos without training the models yourself, you'll need to download the "Completed Generator Weights" listed below and drop them in the /models/ folder.

### Completed Generator Weights

- [Artistic](https://www.dropbox.com/s/zkehq1uwahhbc2o/ColorizeArtistic_gen.pth?dl=0)
- [Stable](https://www.dropbox.com/s/mwjep3vyqk5mkjc/ColorizeStable_gen.pth?dl=0)
- [Video](https://www.dropbox.com/s/336vn9y4qwyg9yz/ColorizeVideo_gen.pth?dl=0)

### Completed Critic Weights

- [Artistic](https://www.dropbox.com/s/8g5txfzt2fw8mf5/ColorizeArtistic_crit.pth?dl=0)
- [Stable](https://www.dropbox.com/s/7a8u20e7xdu1dtd/ColorizeStable_crit.pth?dl=0)
- [Video](https://www.dropbox.com/s/0401djgo1dfxdzt/ColorizeVideo_crit.pth?dl=0)

### Pretrain Only Generator Weights

- [Artistic](https://www.dropbox.com/s/9zexurvrve141n9/ColorizeArtistic_PretrainOnly_gen.pth?dl=0)
- [Stable](https://www.dropbox.com/s/mdnuo1563bb8nh4/ColorizeStable_PretrainOnly_gen.pth?dl=0)
- [Video](https://www.dropbox.com/s/avzixh1ujf86e8x/ColorizeVideo_PretrainOnly_gen.pth?dl=0)

### Pretrain Only Critic Weights

- [Artistic](https://www.dropbox.com/s/lakxe8akzjgjnmh/ColorizeArtistic_PretrainOnly_crit.pth?dl=0)
- [Stable](https://www.dropbox.com/s/b3wka56iyv1fvdc/ColorizeStable_PretrainOnly_crit.pth?dl=0)
- [Video](https://www.dropbox.com/s/j7og84cbhpa94gs/ColorizeVideo_PretrainOnly_crit.pth?dl=0)

