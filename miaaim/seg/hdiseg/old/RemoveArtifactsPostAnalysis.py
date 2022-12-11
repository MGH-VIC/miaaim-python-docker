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



def RemoveArtifactsExportMask(input,output,sigma=None,correction=None):
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
    #Remove probs from memory
    probs = None
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
    mask = np.zeros_like(noise, np.uint8)
    mask[noise > threshold] = 255
    #Write a noise mask
    skimage.io.imsave(os.path.join(output,im_name.stem+"_mask.tif"),mask,plugin='tifffile')

    #Return the noise mask
    return mask


#-------------------Post hoc noise mask export------------
#Loop through the images and export a noise mask
out_dir = "D:/Josh_Hess/Triplets-Melanoma-Ilastik/NoiseMasks_posthoc"
all_nms = ["25546ON",'25546POST',"25546PRE",\
    "26531ON","26531POST","26531PRE",\
    "27960ON","27960POST","27960PRE",\
    "33466ON","33466POST","33466PRE",\
    "33680ON","33680POST","33680PRE",\
    "36844ON","36844POST","36844PRE",\
    "37648ON","37648POST","37648PRE"]
input_files = [os.path.join("D:/Josh_Hess/Triplets-Melanoma-Ilastik",nm+"_Probabilities_noise.tif") for nm in all_nms]
check_files = [os.path.join("D:/Josh_Hess/Triplets-Melanoma-Ilastik",nm+"_Probabilities_noise_RemoveNoise.tif") for nm in all_nms]
#Loop through files
for i in range(len(input_files)):
    #Run the noise removal
    mask = RemoveArtifactsExportMask(input=input_files[i],output=out_dir,sigma=2,correction=1.2)
    #Convert to boolean
    mask = mask.astype(np.bool)
    #Read the check file
    check_im = skimage.io.imread(check_files[i],plugin='tifffile')
    #Get the zeros in the check image
    black_pixels_mask = np.all(check_im == [0, 0, 0], axis=-1)
    #Remove check image from memory
    check_im = None
    #CHeck for equality
    if not np.array_equal(mask, black_pixels_mask):
        raise ValueError("Masks not equal!")
    #Clear memory of both masks -- arrays are the same
    mask, black_pixels_mask = None,None




#
