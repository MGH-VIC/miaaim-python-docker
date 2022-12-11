#Watershed segmentation functions
#Joshua Hess

#Import modules
import mahotas
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




def BinaryErosion(mask,radius):
    """Function for performing binary erosion using a spherical structuring element"""

    #Perform the erosion
    eroded = skimage.util.apply_parallel(
        skimage.morphology.binary_erosion,
        mask,
        extra_keywords={
        "selem":skimage.morphology.disk(int(radius))
        }
    )
    #Return the eroded mask
    return eroded



def OtsuThresh(nuc_channel,sigma=None,correction=None):
    """Function for thresholding nuclei channel from probability image"""

    #Apply filter to nuclei channel if chosen
    if sigma is not None:
        #Apply parallel filtering with sigma value
        nuc = skimage.util.apply_parallel(
            skimage.filters.gaussian,
            nuc_channel,
            extra_keywords={
                "sigma": int(sigma)
            }
        )
    else:
        #Just make a copy of the nuc_channel so the original isnt mutated
        nuc=nuc_channel

    #Apply the global threshold for the nuclei channel
    threshold = skimage.filters.threshold_otsu(nuc)
    #Check for threshold correction
    if correction is not None:
        #Use the correction factor
        threshold = threshold*float(correction)

    #Generate an integer valued mask image based off the threshold
    int_mask = np.zeros_like(nuc, np.uint64)
    #Set values larger than threshold value to be maximum for 64bit
    int_mask[nuc > threshold] = 2^63-1
    #Get a binary mask
    bin_mask = np.zeros_like(nuc, np.bool)
    #Get binary mask counterpart to the int_mask
    bin_mask[int_mask>0]=True
    #Get a labeled mask for the binary mask
    label_mask = skimage.measure.label(bin_mask)
    #Return the mask and threshold value
    return int_mask, bin_mask, label_mask


def FilterHoles(mask,area_threshold=16):
    """Function for removing small holes from an image"""

    #Remove small holes in the image
    mask = skimage.util.apply_parallel(
        skimage.morphology.remove_small_holes,
        mask,
        extra_keywords={
            #Remove holes smaller than the area
            "area_threshold":int(area_threshold),\
            #Type of pixel connectivity
            "connectivity":int(1),\
            #Optional here to mutate the array if true
            "in_place":False
        }
    )
    #Create labeled counterpart for this mask
    labeled_mask = skimage.measure.label(mask)
    #Return the mask
    return mask, labeled_mask



def FilterDebris(mask,min_size=16):
    """Function for filtering debris and small cells in mask"""

    #Remove the small objects
    mask = skimage.util.apply_parallel(
        skimage.morphology.remove_small_objects,
        mask,
        extra_keywords={
        #Remove cells smaller than the area indicated
        "min_size":int(min_size),\
        #Type of pixel connectivity
        "connectivity":int(1),\
        #Optional here to mutate the array if true
        "in_place":False
        }
    )
    #Label the image, assuming this is first pass filtering before watershed
    labeled_image = skimage.measure.label(mask)
    #Return the mask
    return mask, labeled_image



def WatershedSegmentation(labeled_image, mask, footprint=np.ones((3,3))):
    """Function for applying watershed segmentation to a labled mask and corresponding
    binary image mask"""

    #Apply distance transform to the
    distance = skimage.util.apply_parallel(
        scipy.ndimage.distance_transform_edt,
        mask
    )
    #***Adapted from cellprofiler unclumping by shape module***
    watershed_image = 1-distance
    watershed_image = watershed_image-np.min(watershed_image)
    #Get the local max from the distance transform image
    markers = skimage.util.apply_parallel(
        skimage.feature.peak_local_max,
        distance,
        extra_keywords={
            "footprint": footprint,
            "indices": False
        }
    )
    #Get the marker labels
    markers = skimage.measure.label(markers)
    #Get the watershed watershed_boundaries
    watershed_boundaries = skimage.morphology.watershed(connectivity=np.ones((3, 3), bool),\
        image=watershed_image,\
        markers=markers,\
        mask=labeled_image != 0)
    #Create binary mask counterpart of watershed labels
    bin_mask = np.zeros_like(watershed_boundaries,dtype=np.bool)
    #Create boolean array
    bin_mask[watershed_boundaries>0]=True
    #Return the new watershed mask and boolean mask
    return watershed_boundaries, bin_mask


def TripletPipeline(input,output,nuc_channel=0,sigma=2,correction=1.3,area_threshold=64,min_size=30,footprint=np.ones((5,5))):
    """Function for running triplet melanoma cell segmenation pipeline for a single image"""

    #Get the image path
    im = Path(input)
    #Create a name for the output of masking
    new_name = Path(os.path.join(output,(im.stem+"_mask.tif")))
    #Read the image
    image = skimage.io.imread(im,plugin='tifffile')
    #Apply threshold
    int_mask, bin_mask, label_mask = OtsuThresh(image[:,:,nuc_channel],sigma=sigma,correction=correction)
    #Filter small holes
    fil_hol,labeled_fil_hol = FilterHoles(int_mask,area_threshold=area_threshold)
    #Filter debris
    fil_debris,labeled_fil_debris = FilterDebris(fil_hol,min_size = min_size)
    #Apply watershed
    watershed_mask, bin_mask = WatershedSegmentation(labeled_fil_debris,fil_debris,footprint=footprint)
    #Export the mask
    skimage.io.imsave(str(new_name),watershed_mask,plugin='tifffile')
    #Print an update
    print('Finished segmentation for '+str(im))



def MultiTripletPipeline(input,output,nuc_channel=0,sigma=2,correction=1.3,area_threshold=64,min_size=30,footprint=np.ones((5,5))):
    """Function for iterating over a list of images and output locations to segment and export cell masks"""
    #Iterate over each image in the list if only a single output
    if len(output) < 2:
        #Iterate through the images and export to the same location
        for im_name in input:
            #Run the IlastikPrepOME function for this image
            TripletPipeline(im_name,output[0],nuc_channel,sigma,correction,area_threshold,min_size,footprint)
    #Alternatively, iterate over output directories
    else:
        #Check to make sure the output directories and image paths are equal in length
        if len(output) != len(input):
            raise(ValueError("Detected more than one output but not as many directories as images"))
        else:
            #Iterate through images and output directories
            for i in range(len(input)):
                #Run the IlastikPrepOME function for this image and output directory
                TripletPipeline(input[i],output[i],nuc_channel,sigma,correction,area_threshold,min_size,footprint)






#
