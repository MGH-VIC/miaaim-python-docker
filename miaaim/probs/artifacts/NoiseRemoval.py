#Function for removing artifacts from ilastik probability images
#Joshua Hess
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import skimage.exposure
import skimage.feature
import skimage.filters
import skimage.morphology
import skimage.io
import skimage.segmentation
import skimage.util

from pathlib import Path
import h5py
import os



def RemoveArtifacts(input,output,sigma=None,correction=None):
    """Function for removing noise from 4 channel probability image trained in ilastik.
    This function will first apply gaussian filter to the noise channel (4th) of an
    ilastik probability image, then use a global threshold (with correction if not None)
    to create a mask. The mask is then used to zero pixels in the other three channels,
    and the resulting three channel image is exported.
    output: path to output directory"""

    #Get the image path
    im_name = Path(input)

    #Check to see if the image is tiff
    if im_name.suffix == '.tiff' or im_name.suffix == '.tif':
        #Read the image
        probs = skimage.io.imread(im_name,plugin='tifffile')
        #Get the image type
        dtype = probs.dtype
        #Convert to float
        probs = skimage.img_as_float(probs)
    #Check to see if image is hdf5
    elif im_name.suffix == '.h5' or im_name.suffix == '.hdf5':
        #Read the image
        f = h5py.File(im_name,'r+')
        #Get the dataset name from the h5 file
        dat_name = list(f.keys())[0]
        #Get the image data
        probs = np.array(f[dat_name])
        #Remove the first axis (ilastik convention)
        probs = probs.reshape((probs.shape[1],probs.shape[2],probs.shape[3]))
        #Get the image type
        dtype = probs.dtype
        #Convert to float
        probs = skimage.img_as_float(probs)

    #Get the noise channel
    noise = probs[:,:,3]
    #Blur the image
    if sigma is not None:
        noise = skimage.util.apply_parallel(
            skimage.filters.gaussian,
            noise,
            extra_keywords={"sigma": int(sigma)})
    #Threshold the noise image
    threshold = skimage.filters.threshold_otsu(noise)
    #Check for threshold correction
    if correction is not None:
        #Correct threshold
        threshold = threshold*float(correction)
    #Use the threshold to mask the noise image
    mask = np.zeros_like(noise, np.bool)
    mask[noise > threshold] = True
    #Use the mask on each of the channels in the original image
    probs[:,:,:][mask] = 0
    #Convert back to original data type
    #probs = probs.astype(dtype)
    #Export the mask to the same location as the input image and add suffix 'NoiseRemove'
    if im_name.suffix == '.h5' or im_name.suffix == '.hdf5':
        #Create temporary name
        tmp_name = im_name.stem+"_RemoveNoise.hdf5"
        #Export the image to hdf5
        print("Writing "+tmp_name+"...")
        h5 = h5py.File(os.path.join(output,tmp_name), "w")
        #Create dataset only with the three channel image (no noise channel)
        h5.create_dataset(str(dat_name), data=probs[:,:,0:2],chunks=True)
        h5.close()
        print('Finished exporting '+tmp_name)
    #Export the mask to tiff
    elif im_name.suffix == '.tiff' or im_name.suffix == '.tif':
        #Create temporary name
        tmp_name = im_name.stem+"_RemoveNoise.tif"
        #Export the image to tiff
        print("Writing "+tmp_name+"...")
        skimage.io.imsave(os.path.join(output,tmp_name),probs[:,:,0:3],plugin='tifffile')
        print('Finished exporting '+tmp_name)


def MultiRemoveArtifacts(input,output,sigma=None,correction=None):
    """Function for iterating over a list of files and output locations to
    export artifact removed images"""
    #Iterate over each image in the list if only a single output
    if len(output) < 2:
        #Iterate through the images and export to the same location
        for im_name in input:
            #Run the IlastikPrepOME function for this image
            RemoveArtifacts(im_name,output[0],sigma,correction)
    #Alternatively, iterate over output directories
    else:
        #Check to make sure the output directories and image paths are equal in length
        if len(output) != len(input):
            raise(ValueError("Detected more than one output but not as many directories as images"))
        else:
            #Iterate through images and output directories
            for i in range(len(input)):
                #Run the IlastikPrepOME function for this image and output directory
                RemoveArtifacts(input[i],output[i],sigma,correction)

















#
