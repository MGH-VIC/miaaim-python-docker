#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:19:49 2022

@author: joshuahess
"""

# Watershed based cell segmentation
# Developer: Joshua M. Hess, BSc
# Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

# import modules
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
import centrosome
from centrosome import cpmorphology
import centrosome.cpmorphology
import centrosome.propagate
import numpy
import scipy.ndimage
import scipy.sparse
import math

import matplotlib.pyplot as plt


def BinaryErosion(mask,radius,parallel=False):
    """Function for performing binary erosion using a spherical structuring element"""

    if parallel:
        #Perform the erosion
        eroded = skimage.util.apply_parallel(
            skimage.morphology.binary_erosion,
            mask,
            extra_keywords={
            "selem":skimage.morphology.disk(int(radius))
            }
        )
    else:
        eroded = skimage.morphology.binary_erosion(mask,selem=skimage.morphology.disk(int(radius)))
    #Return the eroded mask
    return eroded

def Adjacent(labels):
    """Return a binary mask of all pixels which are adjacent to a pixel of
       a different label.

    """
    high = labels.max() + 1
    if high > np.iinfo(labels.dtype).max:
        labels = labels.astype(np.int32)
    image_with_high_background = labels.copy()
    image_with_high_background[labels == 0] = high
    min_label = scipy.ndimage.minimum_filter(
        image_with_high_background,
        footprint=np.ones((3, 3), bool),
        mode="constant",
        cval=high,
    )
    max_label = scipy.ndimage.maximum_filter(
        labels, footprint=np.ones((3, 3), bool), mode="constant", cval=0
    )
    return (min_label != max_label) & (labels > 0)

def SobelFilter(nuc_channel,parallel=False):

    if parallel:
        #Apply parallel filtering with sigma value
        edges = skimage.util.apply_parallel(
            skimage.filters.sobel,
            nuc_channel,
            extra_keywords={
                "mode": 'reflect',
                'cval': 0.0
            }
        )
    else:
        edges = skimage.filters.sobel(nuc_channel,mode='reflect',cval=0.0)
    return edges

def GaussianFilter(nuc_channel,sigma=2,parallel=False):

    if parallel:
        #Apply parallel filtering with sigma value
        nuc = skimage.util.apply_parallel(
            skimage.filters.gaussian,
            nuc_channel,
            extra_keywords={
                "sigma": int(sigma)
            }
        )
    else:
        nuc = skimage.filters.gaussian(nuc_channel,sigma=int(sigma))
    return nuc

def OtsuThresh(nuc,log_transform=False,correction=None):
    """Function for thresholding nuclei channel from probability image"""

    if log_transform:
        #Apply the global threshold for the nuclei channel
        threshold = np.exp(skimage.filters.threshold_otsu(np.log(nuc)))
    else:
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
    bin_mask = np.zeros_like(nuc, bool)
    #Get binary mask counterpart to the int_mask
    bin_mask[int_mask>0]=True
    #Get a labeled mask for the binary mask
    label_mask = skimage.measure.label(bin_mask)
    #Return the mask and threshold value
    return int_mask, bin_mask, label_mask

def RemoveSmallHoles(mask,area_threshold=16,parallel=False):
    """Function for removing small holes from an image"""

    if parallel:
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
    else:
        mask = skimage.morphology.remove_small_holes(mask,
                                                    area_threshold = int(area_threshold),
                                                    connectivity = int(1),
                                                    in_place = False)
    #Create labeled counterpart for this mask
    labeled_mask = skimage.measure.label(mask)
    #Return the mask
    return mask, labeled_mask

def RemoveDebris(mask,min_size=16,parallel=False):
    """Function for filtering debris and small cells in mask"""

    if parallel:
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
    else:
        mask = skimage.morphology.remove_small_objects(mask,
                                                    min_size = int(min_size),
                                                    connectivity = int(1),
                                                    in_place = False)
    #Label the image, assuming this is first pass filtering before watershed
    labeled_mask = skimage.measure.label(mask)
    #Return the mask
    return mask, labeled_mask


def ClearBorder(mask=None):
    """Remove labels which are on the border of a provided mask
    """

    cleared_labels = skimage.segmentation.clear_border(labels,
                                                buffer_size=0,
                                                bgval=0,
                                                in_place=False,
                                                mask=mask)

    return cleared_labels



def FillLabeledHoles(y_data, mask=None, size_fn=None):
    """Fill labeled mask holes
    """
    filled_labels = cpmorphology.fill_labeled_holes(y_data, mask=mask, size_fn=size_fn)
    return filled_labels


# def WatershedSegmentation(labeled_image, mask, footprint=np.ones((3,3)),parallel=False):
#     """Function for applying watershed segmentation to a labled mask and corresponding
#     binary image mask"""
#
#     if parallel:
#         #Apply distance transform to the
#         distance = skimage.util.apply_parallel(
#             scipy.ndimage.distance_transform_edt,
#             mask
#         )
#     else:
#         distance = scipy.ndimage.distance_transform_edt(mask)
#
#     #***Adapted from cellprofiler unclumping by shape module***
#     watershed_image = 1-distance
#     watershed_image = watershed_image-np.min(watershed_image)
#
#     if parallel:
#         #Get the local max from the distance transform image
#         markers = skimage.util.apply_parallel(
#             skimage.feature.peak_local_max,
#             distance,
#             extra_keywords={
#                 "footprint": footprint,
#                 "indices": False
#             }
#         )
#     else:
#         markers = skimage.feature.peak_local_max(distance,
#                                                 footprint=footprint,
#                                                 indices=False)
#     #Get the marker labels
#     markers = skimage.measure.label(markers)
#     #Get the watershed watershed_boundaries
#     watershed_boundaries = skimage.segmentation.watershed(connectivity=np.ones((1, 1), bool),
#         image=watershed_image,
#         markers=markers,
#         mask=labeled_image != 0,
#         watershed_line=True)
#     #Create binary mask counterpart of watershed labels
#     bin_mask = np.zeros_like(watershed_boundaries,dtype=np.bool)
#     #Create boolean array
#     bin_mask[watershed_boundaries>0]=True
#     #Return the new watershed mask and boolean mask
#     return watershed_boundaries, bin_mask


def WatershedSegmentation(mask, footprint=np.ones((3,3)),parallel=False):
    """Function for applying watershed segmentation to a labled mask and corresponding
    binary image mask"""

    if parallel:
        #Apply distance transform to the
        distance = skimage.util.apply_parallel(
            scipy.ndimage.distance_transform_edt,
            mask
        )
    else:
        distance = scipy.ndimage.distance_transform_edt(mask)

    #***Adapted from cellprofiler unclumping by shape module***
    # watershed_image = 1-distance
    # watershed_image = watershed_image-np.min(watershed_image)

    if parallel:
        #Get the local max from the distance transform image
        coords = skimage.util.apply_parallel(
            skimage.feature.peak_local_max,
            distance,
            extra_keywords={
                "footprint": footprint,
                "indices": False
            }
        )
    else:
        coords = skimage.feature.peak_local_max(distance,
                                                footprint=footprint,
                                                labels=mask)

    mask = np.zeros(distance.shape,dtype='bool')
    mask[tuple(coords.T)] = True
    markers= skimage.measure.label(mask)
    labels = skimage.segmentation.watershed(image = -distance,
                                            markers=markers,
                                            watershed_line=True,
                                            mask=mask)


    # #Get the marker labels
    # markers = skimage.measure.label(markers)
    # #Get the watershed watershed_boundaries
    # watershed_boundaries = skimage.segmentation.watershed(connectivity=np.ones((1, 1), bool),
    #     image=watershed_image,
    #     markers=markers,
    #     mask=labeled_image != 0,
    #     watershed_line=True)
    # #Create binary mask counterpart of watershed labels
    # bin_mask = np.zeros_like(watershed_boundaries,dtype=np.bool)
    # #Create boolean array
    # bin_mask[watershed_boundaries>0]=True
    #Return the new watershed mask and boolean mask
    return labels


def AreaFilter(maxima,min_area,max_area):
    if min_area is not None and max_area is not None:
        maxima = skimage.measure.label(maxima, connectivity=1).astype(np.int32)
        areas = np.bincount(maxima.ravel())
        size_passed = np.arange(areas.size)[
            np.logical_and(areas > min_area, areas < max_area)
        ]
        maxima *= np.isin(maxima, size_passed)
        return np.greater(maxima, 0, out=maxima)



def ExpandLabels(labels, distance, parallel=False):
    """
    Expand labels by a specified distance.
    """

    if parallel:
        out_labels = skimage.util.apply_parallel(
            skimage.segmentation.expand_labels,
            labels,
            extra_keywords={
                "distance": distance
            }
        )
    else:
        out_labels = skimage.segmentation.expand_labels(labels,distance=distance)
    return out_labels


def CellProfilerDeclump(watershed_mask,image,sigma=2,min_distance=0,threshold_rel=0,selem=2):
    """Declumping method based on intensity taken from CellProfiler.
    """

     ##### DECLUMP on intensity from cellprofiler
    x_data = watershed_mask.copy()
    # Get the segmentation distance transform
    peak_image = scipy.ndimage.distance_transform_edt(x_data > 0)
    # plt.imshow(peak_image)

    reference_data = image[:,:,0].copy()

    # Set the image as a float and rescale to full bit depth
    watershed_image = skimage.img_as_float(reference_data, force_copy=True)
    watershed_image -= watershed_image.min()
    watershed_image = 1 - watershed_image
    # plt.imshow(watershed_image)

    # Smooth the image
    watershed_image = skimage.filters.gaussian(watershed_image, sigma=sigma)

    # Generate local peaks
    seeds = skimage.feature.peak_local_max(peak_image,
                                            min_distance=min_distance, # Minimum number of pixels separating peaks in a region of `2 * min_distance + 1 `(i.e. peaks are separated by at least min_distance)
                                            threshold_rel=0,     # Minimum absolute intensity threshold for seed generation. Since this threshold is applied to the distance transformed image, this defines a minimum object "size". Objects smaller than this size will not contain seeds.
                                            exclude_border=False,
                                            num_peaks=np.inf,
                                            indices=False)
    # plt.imshow(seeds)
    # Dilate seeds based on settings
    seeds = skimage.morphology.binary_dilation(seeds, skimage.morphology.disk(int(selem)))
    seeds_dtype = (np.int32)

    # Create a marker array where the unlabeled image has a label of
    # -(nobjects+1)
    # and every local maximum has a unique label which will become
    # the object's label. The labels are negative because that
    # makes the watershed algorithm use FIFO for the pixels which
    # yields fair boundaries when markers compete for pixels.
    #
    seeds = scipy.ndimage.label(seeds)[0]

    markers = np.zeros_like(seeds, dtype=seeds_dtype)
    markers[seeds > 0] = -seeds[seeds > 0]

    # Perform the watershed
    watershed_boundaries = skimage.segmentation.watershed(
        connectivity=1,
        image=watershed_image,
        markers=markers,
        mask=x_data != 0
    )


    y_data = watershed_boundaries.copy()
    # Copy the location of the "background"
    zeros = np.where(y_data == 0)
    # Re-shift all of the labels into the positive realm
    y_data += np.abs(np.min(y_data)) + 1
    # Re-apply the background
    y_data[zeros] = 0

    declumped_labels = cpmorphology.fill_labeled_holes(y_data, mask=None, size_fn=None)

    return declumped_labels



def MarkBoundaries(mask):
    """Return image with boundaries of mask marked
    """
    return skimage.segmentation.find_boundaries(mask)


def DiameterFilter(labeled_mask,diameter_min=8,diameter_max=15):
    """Remove connected components of mask falling outside diameter range
    """

    closed = skimage.morphology.diameter_closing(label_mask, diameter_min, connectivity=1)
    opened = skimage.morphology.diameter_closing(label_mask, diameter_max, connectivity=1)

    # relabel mask
    labeled_mask = skimage.measure.label(opened)

    return labeled_mask



import matplotlib.pyplot as plt
im = Path("/Users/joshuahess/Desktop/ROI028_PROSTATETMA020_Probabilities.tiff")
#Read the image
image = skimage.io.imread(im,plugin='tifffile')
# filter
nuc = GaussianFilter(image[:,:,0])

plt.imshow(image[:,:,0])

#Apply threshold
int_mask, bin_mask, label_mask = OtsuThresh(image[:,:,0],log_transform=False,correction=1.8)
plt.imshow(bin_mask)

#Apply watershed
watershed_mask = WatershedSegmentation(bin_mask)
plt.imshow(watershed_mask)

diameter_filtered = DiameterFilter(watershed_mask,diameter_min=8,diameter_max=15)
plt.imshow(diameter_filtered)

labeled_expand = ExpandLabels(diameter_filtered,distance=3)
plt.imshow(labeled_expand)



test = skimage.segmentation.clear_border(labeled_expand)
plt.imshow(test)

bounds = skimage.segmentation.find_boundaries(test)
plt.imshow(bounds)
new = skimage.segmentation.mark_boundaries(image[:,:,0],bounds)
plt.imshow(new)
skimage.io.imsave("/Users/joshuahess/Desktop/test.tiff",bounds)


final_out = np.stack([image[:,:,0],image[:,:,0],bounds])
final_out.shape

skimage.io.imsave("/Users/joshuahess/Desktop/test3.tiff",final_out)


skimage.io.imsave("/Users/joshuahess/Desktop/test2.tiff",test)









# Filter debris
fil_debris,labeled_fil_debris = RemoveDebris(label_mask,min_size = 5)
plt.imshow(labeled_fil_hol)

#Apply watershed
# watershed_mask, bin_mask = WatershedSegmentation(label_mask,labeled_fil_debris,footprint=np.ones((3,3)))
# plt.imshow(watershed_mask)

labeled_expand = ExpandLabels(diameter_filtered,distance=3)
plt.imshow(labeled_expand)


declumped_labels = CellProfilerDeclump(labeled_expand,image)

plt.imshow(declumped_labels)

filled_labels = FillLabeledHoles(declumped_labels

)
plt.imshow(filled_labels)


diameter_filtered = DiameterFilter(filled_labels,diameter_min=8,diameter_max=15)

plt.imshow(diameter_filtered)


labeled_expand = ExpandLabels(diameter_filtered,distance=3)
plt.imshow(labeled_expand)

filled_labels = FillLabeledHoles(declumped_labels

)



test = skimage.segmentation.clear_border(filled_labels)
plt.imshow(test)

bounds = skimage.segmentation.find_boundaries(test)
plt.imshow(bounds)
new = skimage.segmentation.mark_boundaries(image[:,:,0],bounds)
plt.imshow(new)
skimage.io.imsave("test.tiff",bounds)


final_out = np.stack([image[:,:,0],image[:,:,0],bounds])
final_out.shape

skimage.io.imsave("test3.tiff",final_out)


skimage.io.imsave("test2.tiff",test)






t = Path("/Users/joshuahess/Desktop/ROI028_PROSTATETMA020_Probabilities_mask.tiff")
#Read the image
t = skimage.io.imread(t,plugin='tifffile')
boundst = skimage.segmentation.find_boundaries(t)


final_outt = np.stack([image[:,:,0],image[:,:,0],boundst])
final_outt.shape

skimage.io.imsave("test3.tiff",final_outt)









import matplotlib.pyplot as plt
im = Path("/Users/joshuahess/Desktop/ROI028_PROSTATETMA020_Probabilities.tiff")
#Read the image
image = skimage.io.imread(im,plugin='tifffile')
# filter
nuc = GaussianFilter(image[:,:,0])

plt.imshow(image[:,:,0])

#Apply threshold
int_mask, bin_mask, label_mask = OtsuThresh(image[:,:,0],correction=1.4)
plt.imshow(label_mask)


filled_labels = FillLabeledHoles(label_mask)
plt.imshow(filled_labels)

#Filter small holes
fil_hol,labeled_fil_hol = RemoveSmallHoles(filled_labels,area_threshold=8)

plt.imshow(labeled_fil_hol)

#Filter debris
fil_debris,labeled_fil_debris = RemoveDebris(fil_hol,min_size = 5)
plt.imshow(labeled_fil_hol)

#Apply watershed
watershed_mask, bin_mask = WatershedSegmentation(edges,labeled_fil_debris,footprint=np.ones((3,3)))
plt.imshow(watershed_mask)

labeled_expand = ExpandLabels(watershed_mask,distance=1)
plt.imshow(labeled_expand)


declumped_labels = CellProfilerDeclump(labeled_expand,image)

test = skimage.segmentation.clear_border(declumped_labels)
plt.imshow(test)

bounds = skimage.segmentation.find_boundaries(test)
plt.imshow(bounds)
new = skimage.segmentation.mark_boundaries(image[:,:,0],watershed_mask)
plt.imshow(new)
skimage.io.imsave("test.tiff",new)








