# Nuclear and membrane based cell segmentation
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
import os
import centrosome
import centrosome.cpmorphology
import centrosome.propagate
import numpy
import scipy.sparse
import math
import yaml
import logging
import sys

# import custom modules
import miaaim
from miaaim.cli.seg.hdiseg import _parse


# set constants according to CellProfiler convention
UN_INTENSITY = "Intensity"
UN_SHAPE = "Shape"
UN_LOG = "Laplacian of Gaussian"
UN_NONE = "None"

WA_INTENSITY = "Intensity"
WA_SHAPE = "Shape"
WA_PROPAGATE = "Propagate"
WA_NONE = "None"

M_PROPAGATION = "Propagation"
M_WATERSHED_G = "Watershed - Gradient"
M_WATERSHED_I = "Watershed - Image"
M_DISTANCE_N = "Distance - N"
M_DISTANCE_B = "Distance - B"

# =============================================================================
# Nuecleus segmentation functions
# =============================================================================

def SearchDir(ending = ".txt",dir=None):
    """Search only in given directory for files that end with
    the specified suffix.

    Parameters
    ----------
    ending: string (Default: ".txt")
        Ending to search for in the given directory

    dir: string (Default: None, will search in current working directory)
        Directory to search for files in.

    Returns
    -------
    full_list: list
        List of pathlib objects for each file found with the given suffix.
    """

    #If directory is not specified, use the working directory
    if dir is None:
        tmp = Path('..')
        dir = tmp.cwd()
    #Search the directory only for files
    full_list = []
    for file in os.listdir(dir):
        if file.endswith(ending):
            full_list.append(Path(os.path.join(dir,file)))
    #Return the list
    return full_list



def GaussianFilter(channel,sigma=2,parallel=False):
    """
    Gaussian filter image channel.

    Parameters
    ----------
    channel : TYPE
        DESCRIPTION.
    sigma : TYPE, optional
        DESCRIPTION. The default is 2.
    parallel : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    smoothed : TYPE
        DESCRIPTION.

    """

    if parallel:
        #Apply parallel filtering with sigma value
        smoothed = skimage.util.apply_parallel(
            skimage.filters.gaussian,
            channel,
            extra_keywords={
                "sigma": int(sigma)
            }
        )
    else:
        smoothed = skimage.filters.gaussian(channel,sigma=int(sigma))
    return smoothed

def OtsuThresh(image,correction=None):
    """
    Function for thresholding single grayscale image channel.

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    correction : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    int_mask : TYPE
        DESCRIPTION.
    bin_mask : TYPE
        DESCRIPTION.
    label_mask : TYPE
        DESCRIPTION.

    """

    # apply the global threshold for the nuclei channel
    threshold = skimage.filters.threshold_otsu(image)
    #Check for threshold correction
    if correction is not None:
        #Use the correction factor
        threshold = threshold*float(correction)

    #Generate an integer valued mask image based off the threshold
    int_mask = np.zeros_like(image, np.uint64)
    #Set values larger than threshold value to be maximum for 64bit
    int_mask[image > threshold] = 2^63-1
    #Get a binary mask
    bin_mask = np.zeros_like(image, bool)
    #Get binary mask counterpart to the int_mask
    bin_mask[int_mask>0]=True
    #Get a labeled mask for the binary mask
    label_mask = skimage.measure.label(bin_mask)
    #Return the mask and threshold value
    return int_mask, bin_mask, label_mask



def smooth_image(
        image,
        filter_size,
        mask=None,
        use_cellprofiler=False,
        parallel=False
        ):
    """
    Smooth image channel. All credit to cellprofiler developers.

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    filter_size : TYPE
        DESCRIPTION.
    mask : TYPE, optional
        DESCRIPTION. The default is None.
    use_cellprofiler : TYPE, optional
        DESCRIPTION. The default is False.
    parallel : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # check for cellprofiler smoothing
    # this condition copied from cellprofiler github
    if use_cellprofiler:

        if filter_size == 0:
            return image
        sigma = filter_size / 2.35
        #
        # We not only want to smooth using a Gaussian, but we want to limit
        # the spread of the smoothing to 2 SD, partly to make things happen
        # locally, partly to make things run faster, partly to try to match
        # the Matlab behavior.
        #
        filter_size = max(int(float(filter_size) / 2.0), 1)
        f = (
            1
            / np.sqrt(2.0 * np.pi)
            / sigma
            * np.exp(
                -0.5 * np.arange(-filter_size, filter_size + 1) ** 2 / sigma ** 2
            )
        )

        def fgaussian(image):
            output = scipy.ndimage.convolve1d(image, f, axis=0, mode="constant")
            return scipy.ndimage.convolve1d(output, f, axis=1, mode="constant")

        #
        # Use the trick where you similarly convolve an array of ones to find
        # out the edge effects, then divide to correct the edge effects
        #
        edge_array = fgaussian(mask.astype(float))
        masked_image = image.copy()
        masked_image[~mask] = 0
        smoothed_image = fgaussian(masked_image)
        masked_image[mask] = smoothed_image[mask] / edge_array[mask]
        # get final smoothed image
        smoothed = masked_image.copy()
    else:
        # simply smooth using the gaussian smoothing skimage function
        smoothed = GaussianFilter(channel=image,
                                  sigma=filter_size,
                                  parallel=parallel)
    return smoothed




def separate_neighboring_objects(labeled_image,
                                 object_count,
                                 image,
                                 mask=None,
                                 force_resmooth=True,
                                 smoothing_filter_size=2,
                                 use_cellprofiler_smoothing=False,
                                 parallel=False,
                                 unclump_method=UN_INTENSITY,
                                 watershed_method=WA_INTENSITY,
                                 automatic_suppression=True,
                                 advanced=False,
                                 basic=True,
                                 size_range_min=8,
                                 size_range_max=16,
                                 maxima_suppression_size_value=5):
    """
    Separate objects based on local maxima or distance transform
    labeled_image - image labeled by scipy.ndimage.label
    object_count  - # of objects in image
    returns revised labeled_image, object count, maxima_suppression_size,
    LoG threshold and filter diameter.

    Parameters
    ----------
    labeled_image : TYPE
        DESCRIPTION.
    object_count : TYPE
        DESCRIPTION.
    image : TYPE
        DESCRIPTION.
    mask : TYPE, optional
        DESCRIPTION. The default is None.
    force_resmooth : TYPE, optional
        DESCRIPTION. The default is False.
    smoothing_filter_size : TYPE, optional
        DESCRIPTION. The default is 2.
    use_cellprofiler_smoothing : TYPE, optional
        DESCRIPTION. The default is False.
    parallel : TYPE, optional
        DESCRIPTION. The default is False.
    unclump_method : TYPE, optional
        DESCRIPTION. The default is UN_INTENSITY.
    watershed_method : TYPE, optional
        DESCRIPTION. The default is WA_INTENSITY.
    automatic_suppression : TYPE, optional
        DESCRIPTION. The default is True.
    advanced : TYPE, optional
        DESCRIPTION. The default is False.
    basic : TYPE, optional
        DESCRIPTION. The default is True.
    size_range_min : TYPE, optional
        DESCRIPTION. The default is 8.
    size_range_max : TYPE, optional
        DESCRIPTION. The default is 16.

    Returns
    -------
    watershed_boundaries : TYPE
        DESCRIPTION.
    object_count : TYPE
        DESCRIPTION.
    reported_maxima_suppression_size : TYPE
        DESCRIPTION.

    """

    # check for smoothing
    if force_resmooth:
        blurred_image = smooth_image(image=image,
                                     filter_size=smoothing_filter_size,
                                     mask=mask,
                                     use_cellprofiler=use_cellprofiler_smoothing,
                                     parallel=parallel)
    else:
        blurred_image = image.copy()

    image_resize_factor = 1.0
    if basic or automatic_suppression:
        maxima_suppression_size = size_range_min / 1.5
    else:
        maxima_suppression_size = maxima_suppression_size_value
    reported_maxima_suppression_size = maxima_suppression_size

    maxima_mask = centrosome.cpmorphology.strel_disk(
        max(1, maxima_suppression_size - 0.5)
    )
    distance_transformed_image = None


    if basic or unclump_method == UN_INTENSITY:
        # Remove dim maxima
        maxima_image = get_maxima(
            blurred_image, labeled_image, maxima_mask, image_resize_factor
        )

    # Create the image for watershed
    if basic or watershed_method == WA_INTENSITY:
        # use the reverse of the image to get valleys at peaks
        watershed_image = 1 - image
    elif watershed_method == WA_SHAPE:
        if distance_transformed_image is None:
            distance_transformed_image = scipy.ndimage.distance_transform_edt(
                labeled_image > 0
            )
        watershed_image = -distance_transformed_image
        watershed_image = watershed_image - np.min(watershed_image)

    #
    # Create a marker array where the unlabeled image has a label of
    # -(nobjects+1)
    # and every local maximum has a unique label which will become
    # the object's label. The labels are negative because that
    # makes the watershed algorithm use FIFO for the pixels which
    # yields fair boundaries when markers compete for pixels.
    #
    labeled_maxima, object_count = scipy.ndimage.label(
        maxima_image, numpy.ones((3, 3), bool)
    )
    if advanced and watershed_method == WA_PROPAGATE:
        watershed_boundaries, distance = centrosome.propagate.propagate(
            numpy.zeros(labeled_maxima.shape),
            labeled_maxima,
            labeled_image != 0,
            1.0,
        )
    else:
        markers_dtype = (
            numpy.int16
            if object_count < numpy.iinfo(numpy.int16).max
            else numpy.int32
        )
        markers = np.zeros(watershed_image.shape, markers_dtype)
        markers[labeled_maxima > 0] = -labeled_maxima[
            labeled_maxima > 0
        ]

        #
        # Some labels have only one maker in them, some have multiple and
        # will be split up.
        #

        watershed_boundaries = skimage.segmentation.watershed(
            connectivity=np.ones((3, 3), bool),
            image=watershed_image,
            markers=markers,
            mask=labeled_image != 0,
        )

        watershed_boundaries = -watershed_boundaries

    return watershed_boundaries, object_count, reported_maxima_suppression_size



def get_maxima(image, labeled_image, maxima_mask, image_resize_factor):
    """


    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    labeled_image : TYPE
        DESCRIPTION.
    maxima_mask : TYPE
        DESCRIPTION.
    image_resize_factor : TYPE
        DESCRIPTION.

    Returns
    -------
    shrunk_image : TYPE
        DESCRIPTION.

    """

    if image_resize_factor < 1.0:
        shape = numpy.array(image.shape) * image_resize_factor
        i_j = (
            numpy.mgrid[0 : shape[0], 0 : shape[1]].astype(float)
            / image_resize_factor
        )
        resized_image = scipy.ndimage.map_coordinates(image, i_j)
        resized_labels = scipy.ndimage.map_coordinates(
            labeled_image, i_j, order=0
        ).astype(labeled_image.dtype)

    else:
        resized_image = image
        resized_labels = labeled_image
    #
    # find local maxima
    #
    if maxima_mask is not None:
        binary_maxima_image = centrosome.cpmorphology.is_local_maximum(
            resized_image, resized_labels, maxima_mask
        )
        binary_maxima_image[resized_image <= 0] = 0
    else:
        binary_maxima_image = (resized_image > 0) & (labeled_image > 0)
    if image_resize_factor < 1.0:
        inverse_resize_factor = float(image.shape[0]) / float(
            binary_maxima_image.shape[0]
        )
        i_j = (
            numpy.mgrid[0 : image.shape[0], 0 : image.shape[1]].astype(float)
            / inverse_resize_factor
        )
        binary_maxima_image = (
            scipy.ndimage.map_coordinates(binary_maxima_image.astype(float), i_j)
            > 0.5
        )
        assert binary_maxima_image.shape[0] == image.shape[0]
        assert binary_maxima_image.shape[1] == image.shape[1]

    # Erode blobs of touching maxima to a single point

    shrunk_image = centrosome.cpmorphology.binary_shrink(binary_maxima_image)
    return shrunk_image


def filter_on_size(labeled_image, object_count, size_range_min=8, size_range_max=16):
    """
    Filter the labeled image based on the size range
    labeled_image - pixel image labels
    object_count - # of objects in the labeled image
    returns the labeled image, and the labeled image with the
    small objects removed.

    Parameters
    ----------
    labeled_image : TYPE
        DESCRIPTION.
    object_count : TYPE
        DESCRIPTION.
    size_range_min : TYPE, optional
        DESCRIPTION. The default is 8.
    size_range_max : TYPE, optional
        DESCRIPTION. The default is 16.

    Returns
    -------
    labeled_image : TYPE
        DESCRIPTION.
    small_removed_labels : TYPE
        DESCRIPTION.

    """
    if object_count > 0:
        areas = scipy.ndimage.measurements.sum(
            np.ones(labeled_image.shape),
            labeled_image,
            np.array(list(range(0, object_count + 1)), dtype=np.int32),
        )
        areas = np.array(areas, dtype=int)
        min_allowed_area = (
            np.pi * (size_range_min * size_range_min) / 4
        )
        max_allowed_area = (
            np.pi * (size_range_max * size_range_max) / 4
        )
        # area_image has the area of the object at every pixel within the object
        area_image = areas[labeled_image]
        labeled_image[area_image < min_allowed_area] = 0
        small_removed_labels = labeled_image.copy()
        labeled_image[area_image > max_allowed_area] = 0
    else:
        small_removed_labels = labeled_image.copy()
    return labeled_image, small_removed_labels

def filter_on_border(image, labeled_image, image_mask=False, mask=None, exclude_border_objects=True):
    """
    Filter out objects touching the border
    In addition, if the image has a mask, filter out objects
    touching the border of the mask.

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    labeled_image : TYPE
        DESCRIPTION.
    image_mask : TYPE, optional
        DESCRIPTION. The default is False.
    mask : TYPE, optional
        DESCRIPTION. The default is None.
    exclude_border_objects : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    labeled_image : TYPE
        DESCRIPTION.

    """
    if exclude_border_objects:
        border_labels = list(labeled_image[0, :])
        border_labels.extend(labeled_image[:, 0])
        border_labels.extend(labeled_image[labeled_image.shape[0] - 1, :])
        border_labels.extend(labeled_image[:, labeled_image.shape[1] - 1])
        border_labels = np.array(border_labels)
        #
        # the following histogram has a value > 0 for any object
        # with a border pixel
        #
        histogram = scipy.sparse.coo_matrix(
            (
                np.ones(border_labels.shape),
                (border_labels, np.zeros(border_labels.shape)),
            ),
            shape=(np.max(labeled_image) + 1, 1),
        ).todense()
        histogram = np.array(histogram).flatten()
        if any(histogram[1:] > 0):
            histogram_image = histogram[labeled_image]
            labeled_image[histogram_image > 0] = 0
        elif image_mask:
            # The assumption here is that, if nothing touches the border,
            # the mask is a large, elliptical mask that tells you where the
            # well is. That's the way the old Matlab code works and it's duplicated here
            #
            # The operation below gets the mask pixels that are on the border of the mask
            # The erosion turns all pixels touching an edge to zero. The not of this
            # is the border + formerly masked-out pixels.
            mask_border = np.logical_not(
                scipy.ndimage.binary_erosion(mask)
            )
            mask_border = np.logical_and(mask_border, mask)
            border_labels = labeled_image[mask_border]
            border_labels = border_labels.flatten()
            histogram = scipy.sparse.coo_matrix(
                (
                    np.ones(border_labels.shape),
                    (border_labels, np.zeros(border_labels.shape)),
                ),
                shape=(np.max(labeled_image) + 1, 1),
            ).todense()
            histogram = np.array(histogram).flatten()
            if any(histogram[1:] > 0):
                histogram_image = histogram[labeled_image]
                labeled_image[histogram_image > 0] = 0
    return labeled_image

def set_image(image, convert=True):
    """
    Taken directly from cellprofiler.

    Convert the image to a numpy array of dtype = np.float64.
    Rescale according to Matlab's rules for im2double:
    * single/double values: keep the same
    * uint8/16/32/64: scale 0 to max to 0 to 1
    * int8/16/32/64: scale min to max to 0 to 1
    * logical: save as is (and get if must_be_binary)

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    convert : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    img : TYPE
        DESCRIPTION.

    """
    img = np.asanyarray(image)
    mval = 0.0
    scale = 1.0
    fix_range = False
    if issubclass(img.dtype.type, np.floating):
        pass
    elif img.dtype.type is np.uint8:
        scale = math.pow(2.0, 8.0) - 1
    elif img.dtype.type is np.uint16:
        scale = math.pow(2.0, 16.0) - 1
    elif img.dtype.type is np.uint32:
        scale = math.pow(2.0, 32.0) - 1
    elif img.dtype.type is np.uint64:
        scale = math.pow(2.0, 64.0) - 1
    elif img.dtype.type is np.int8:
        scale = math.pow(2.0, 8.0)
        mval = -scale / 2.0
        scale -= 1
        fix_range = True
    elif img.dtype.type is np.int16:
        scale = math.pow(2.0, 16.0)
        mval = -scale / 2.0
        scale -= 1
        fix_range = True
    elif img.dtype.type is np.int32:
        scale = math.pow(2.0, 32.0)
        mval = -scale / 2.0
        scale -= 1
        fix_range = True
    elif img.dtype.type is np.int64:
        scale = math.pow(2.0, 64.0)
        mval = -scale / 2.0
        scale -= 1
        fix_range = True
    # Avoid temporaries by doing the shift/scale in place.
    img = img.astype(np.float32)
    img -= mval
    img /= scale
    if fix_range:
        # These types will always have ranges between 0 and 1. Make it so.
        np.clip(img, 0, 1, out=img)
    return img

# =============================================================================
# Membrane segmentation functions
# =============================================================================

def filter_labels(
        labels_out,
        objects,
        image,
        segmented_labels,
        mask=None,
        wants_discard_edge=False
        ):
    """Filter labels out of the output
    Filter labels that are not in the segmented input labels. Optionally
    filter labels that are touching the edge.
    labels_out - the unfiltered output labels
    objects    - the objects thing, containing both segmented and
    small_removed labels
    """

    max_out = np.max(labels_out)
    if max_out > 0:
        segmented_labels, m1 = size_similarly(labels_out, segmented_labels)
        segmented_labels[~m1] = 0
        lookup = scipy.ndimage.maximum(
            segmented_labels, labels_out, list(range(max_out + 1))
        )
        lookup = np.array(lookup, int)
        lookup[0] = 0
        segmented_labels_out = lookup[labels_out]
    else:
        segmented_labels_out = labels_out.copy()
    if wants_discard_edge:
        if mask is not None:
            mask_border = mask & ~scipy.ndimage.binary_erosion(mask)
            edge_labels = segmented_labels_out[mask_border]
        else:
            edge_labels = np.hstack(
                (
                    segmented_labels_out[0, :],
                    segmented_labels_out[-1, :],
                    segmented_labels_out[:, 0],
                    segmented_labels_out[:, -1],
                )
            )
        edge_labels = np.unique(edge_labels)
        #
        # Make a lookup table that translates edge labels to zero
        # but translates everything else to itself
        #
        lookup = np.arange(max(max_out, np.max(segmented_labels)) + 1)
        lookup[edge_labels] = 0
        #
        # Run the segmented labels through this to filter out edge
        # labels
        segmented_labels_out = lookup[segmented_labels_out]

    return segmented_labels_out



def size_similarly(labels, secondary):
    """Size the secondary matrix similarly to the labels matrix
    labels - labels matrix
    secondary - a secondary image or labels matrix which might be of
                different size.
    Return the resized secondary matrix and a mask indicating what portion
    of the secondary matrix is bogus (manufactured values).
    Either the mask is all ones or the result is a copy, so you can
    modify the output within the unmasked region w/o destroying the original.
    """
    if labels.shape[:2] == secondary.shape[:2]:
        return secondary, np.ones(secondary.shape, bool)
    if labels.shape[0] <= secondary.shape[0] and labels.shape[1] <= secondary.shape[1]:
        if secondary.ndim == 2:
            return (
                secondary[: labels.shape[0], : labels.shape[1]],
                np.ones(labels.shape, bool),
            )
        else:
            return (
                secondary[: labels.shape[0], : labels.shape[1], :],
                np.ones(labels.shape, bool),
            )

    #
    # Some portion of the secondary matrix does not cover the labels
    #
    result = np.zeros(
        list(labels.shape) + list(secondary.shape[2:]), secondary.dtype
    )
    i_max = min(secondary.shape[0], labels.shape[0])
    j_max = min(secondary.shape[1], labels.shape[1])
    if secondary.ndim == 2:
        result[:i_max, :j_max] = secondary[:i_max, :j_max]
    else:
        result[:i_max, :j_max, :] = secondary[:i_max, :j_max, :]
    mask = np.zeros(labels.shape, bool)
    mask[:i_max, :j_max] = 1
    return result, mask




# =============================================================================
# Segmentation implementations (nucleus and membrane)
# =============================================================================

def NuclearSegmentation(
        nuclear_image,
        mask=None,
        threshold_smoothing_filter_size=2,
        force_separation_smoothing=False,
        separation_smoothing_filter_size=2,
        use_cellprofiler_smoothing=False,
        parallel=False,
        threshold_correction=1.2,
        size_range_min=8,
        size_range_max=16,
        exlcude_outside_size_range=False,
        exclude_border_labels=True,
        unclump_method=UN_INTENSITY,
        watershed_method=WA_INTENSITY,
        automatic_suppression=True,
        advanced=False,
        basic=True
        ):
    """
    Single-nucleus segmentation pipeline taken / adapted from CellProfiler
    identifyPrimaryObjects module.

    Parameters
    ----------
    nuclear_image : TYPE
        DESCRIPTION.
    mask : TYPE, optional
        DESCRIPTION. The default is None.
    smoothing_filter_size : TYPE, optional
        DESCRIPTION. The default is 2.
    use_cellprofiler_smoothing : TYPE, optional
        DESCRIPTION. The default is False.
    parallel : TYPE, optional
        DESCRIPTION. The default is False.
    threshold_correction : TYPE, optional
        DESCRIPTION. The default is 1.3.
    size_range_min : TYPE, optional
        DESCRIPTION. The default is 8.
    size_range_max : TYPE, optional
        DESCRIPTION. The default is 16.
    unclump_method : TYPE, optional
        DESCRIPTION. The default is UN_INTENSITY.
    watershed_method : TYPE, optional
        DESCRIPTION. The default is WA_INTENSITY.
    automatic_suppression : TYPE, optional
        DESCRIPTION. The default is True.
    advanced : TYPE, optional
        DESCRIPTION. The default is False.
    basic : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    # smooth nuclear prediction probability image
    smoothed_nuclear_image = smooth_image(image=nuclear_image,
                                          filter_size=threshold_smoothing_filter_size,
                                          mask=mask,
                                          use_cellprofiler=use_cellprofiler_smoothing,
                                          parallel=parallel)

    # binary thresholding of smoothed image
    int_mask, binary_image, label_mask = OtsuThresh(image=smoothed_nuclear_image,
                                                    correction=threshold_correction)

    #
    # Fill background holes inside foreground objects
    #
    def size_fn(size, is_foreground):
        return size < size_range_max * size_range_max

    # fill holes in binary image
    binary_image = centrosome.cpmorphology.fill_labeled_holes(
        binary_image, size_fn=size_fn
    )

    # get labeled image and object counts
    labeled_image, object_count = scipy.ndimage.label(
        binary_image, numpy.ones((3, 3), bool)
    )

    # perform unclumping / separating objects
    (labeled_image,
     object_count,
     maxima_suppression_size,
    ) = separate_neighboring_objects(labeled_image=labeled_image,
                                 object_count=object_count,
                                 image=smoothed_nuclear_image,
                                 mask=mask,
                                 force_resmooth=force_separation_smoothing,
                                 smoothing_filter_size=separation_smoothing_filter_size,
                                 unclump_method=unclump_method,
                                 watershed_method=watershed_method,
                                 automatic_suppression=automatic_suppression,
                                 advanced=advanced,
                                 basic=basic,
                                 size_range_min=size_range_min,
                                 size_range_max=size_range_max)

    # get unedited labels
    unedited_labels = labeled_image.copy()

    if exclude_border_labels:
        # Filter out objects touching the border or mask
        border_excluded_labeled_image = labeled_image.copy()
        labeled_image = filter_on_border(nuclear_image, labeled_image)
        border_excluded_labeled_image[labeled_image > 0] = 0
        # create outline of labeled image of border excluded labels
        outline_border_excluded_image = centrosome.outline.outline(
            border_excluded_labeled_image
        )
    else:
        border_excluded_labeled_image = None
        outline_border_excluded_image = None

    # check for excluding outside of size range
    if exlcude_outside_size_range:
        # Filter out small and large objects
        size_excluded_labeled_image = labeled_image.copy()
        labeled_image, small_removed_labels = filter_on_size(
            labeled_image, object_count
        )
        size_excluded_labeled_image[labeled_image > 0] = 0
        # create outline of labels of size excluded labels
        outline_size_excluded_image = centrosome.outline.outline(
            size_excluded_labeled_image
        )
    else:
        size_excluded_labeled_image = None
        outline_size_excluded_image = None


    #
    # Fill holes again after watershed
    #
    labeled_image = centrosome.cpmorphology.fill_labeled_holes(labeled_image)

    # Relabel the image
    labeled_image, object_count = centrosome.cpmorphology.relabel(labeled_image)

    # Make an outline image
    outline_image = centrosome.outline.outline(labeled_image)

    # return information
    return (labeled_image,
            object_count,
            outline_image,
            border_excluded_labeled_image,
            outline_border_excluded_image,
            size_excluded_labeled_image,
            outline_size_excluded_image,
            unedited_labels)



def MembraneSegmentation(
        membrane_image,
        mask,
        objects,
        labels_in,
        distance_to_dilate=3,
        threshold_correction=1.8,
        fill_holes=True,
        method=M_PROPAGATION,
        discard_edge=True,
        regularization_factor=1.8
        ):
    """


    Parameters
    ----------
    membrane_image : TYPE
        DESCRIPTION.
    mask : TYPE
        DESCRIPTION.
    objects : TYPE
        DESCRIPTION.
    labels_in : TYPE
        DESCRIPTION.
    distance_to_dilate : TYPE, optional
        DESCRIPTION. The default is 3.
    threshold_correction : TYPE, optional
        DESCRIPTION. The default is 1.8.
    fill_holes : TYPE, optional
        DESCRIPTION. The default is True.
    method : TYPE, optional
        DESCRIPTION. The default is M_PROPAGATION.
    discard_edge : TYPE, optional
        DESCRIPTION. The default is True.
    regularization_factor : TYPE, optional
        DESCRIPTION. The default is 1.8.

    Returns
    -------
    segmented_out : TYPE
        DESCRIPTION.

    """

    # get pixel data
    img=set_image(membrane_image, convert=True)

    # check for whether to threshold or not based on method of segmentation
    if method == M_DISTANCE_N:
        has_threshold = False
    else:
        thresholded_image,_,_ = OtsuThresh(image=membrane_image,
                                           correction=threshold_correction)
        has_threshold = True

    #
    # Get the following labels:
    # * all edited labels
    # * labels touching the edge, including small removed
    #
    labels_in = labels_in.copy()
    labels_touching_edge = np.hstack(
        (labels_in[0, :], labels_in[-1, :], labels_in[:, 0], labels_in[:, -1])
    )
    labels_touching_edge = np.unique(labels_touching_edge)
    is_touching = np.zeros(numpy.max(labels_in) + 1, bool)
    is_touching[labels_touching_edge] = True
    is_touching = is_touching[labels_in]

    labels_in[(~is_touching) & (objects == 0)] = 0

    # check for methods of secondary object detection
    if method in (M_DISTANCE_B, M_DISTANCE_N):
        if method == M_DISTANCE_N:
            distances, (i, j) = scipy.ndimage.distance_transform_edt(
                labels_in == 0, return_indices=True
            )
            labels_out = np.zeros(labels_in.shape, int)
            dilate_mask = distances <= distance_to_dilate
            labels_out[dilate_mask] = labels_in[i[dilate_mask], j[dilate_mask]]
        else:
            labels_out, distances = centrosome.propagate.propagate(
                img, labels_in, thresholded_image, 1.0
            )
            labels_out[distances > distance_to_dilate] = 0
            labels_out[labels_in > 0] = labels_in[labels_in > 0]
        if fill_holes:
            label_mask = labels_out == 0
            small_removed_segmented_out = centrosome.cpmorphology.fill_labeled_holes(
                labels_out, mask=label_mask
            )
        else:
            small_removed_segmented_out = labels_out
        #
        # Create the final output labels by removing labels in the
        # output matrix that are missing from the segmented image
        #

        # filter
        segmented_out = filter_labels(
            small_removed_segmented_out,
            objects,
            image=membrane_image,
            mask=mask,
            segmented_labels=labels_in,
            wants_discard_edge=discard_edge
        )
    # check for propagation type methods
    elif method == M_PROPAGATION:
        labels_out, distance = centrosome.propagate.propagate(
            img, labels_in, thresholded_image, regularization_factor
        )
        if fill_holes:
            label_mask = labels_out == 0
            small_removed_segmented_out = centrosome.cpmorphology.fill_labeled_holes(
                labels_out, mask=label_mask
            )
        else:
            small_removed_segmented_out = labels_out.copy()
        # filter as in the other options
        segmented_out = filter_labels(
            small_removed_segmented_out,
            objects,
            image=membrane_image,
            mask=mask,
            segmented_labels=labels_in,
            wants_discard_edge=discard_edge
        )
    # check for watershed gradient based method
    elif method == M_WATERSHED_G:
        #
        # First, apply the sobel filter to the image (both horizontal
        # and vertical). The filter measures gradient.
        #
        sobel_image = numpy.abs(scipy.ndimage.sobel(img))
        #
        # Combine the image mask and threshold to mask the watershed
        #
        watershed_mask = numpy.logical_or(thresholded_image, labels_in > 0)
        watershed_mask = numpy.logical_and(watershed_mask, mask)

        #
        # Perform the first watershed
        #

        labels_out = skimage.segmentation.watershed(
            connectivity=numpy.ones((3, 3), bool),
            image=sobel_image,
            markers=labels_in,
            mask=watershed_mask,
        )

        if fill_holes:
            label_mask = labels_out == 0
            small_removed_segmented_out = centrosome.cpmorphology.fill_labeled_holes(
                labels_out, mask=label_mask
            )
        else:
            small_removed_segmented_out = labels_out.copy()
        # filter as in the other options
        segmented_out = filter_labels(
            small_removed_segmented_out,
            objects,
            image=membrane_image,
            mask=mask,
            segmented_labels=labels_in,
            wants_discard_edge=discard_edge
        )
    # intensity watershed method
    elif method == M_WATERSHED_I:
        #
        # invert the image so that the maxima are filled first
        # and the cells compete over what's close to the threshold
        #
        inverted_img = 1 - img
        #
        # Same as above, but perform the watershed on the original image
        #
        watershed_mask = numpy.logical_or(thresholded_image, labels_in > 0)
        watershed_mask = numpy.logical_and(watershed_mask, mask)
        #
        # Perform the watershed
        #

        labels_out = skimage.segmentation.watershed(
            connectivity=numpy.ones((3, 3), bool),
            image=inverted_img,
            markers=labels_in,
            mask=watershed_mask,
        )

        if fill_holes:
            label_mask = labels_out == 0
            small_removed_segmented_out = centrosome.cpmorphology.fill_labeled_holes(
                labels_out, mask=label_mask
            )
        else:
            small_removed_segmented_out = labels_out
        # filter as in the other options
        segmented_out = filter_labels(
            small_removed_segmented_out,
            objects,
            image=membrane_image,
            mask=mask,
            segmented_labels=labels_in,
            wants_discard_edge=discard_edge
        )

    # return segmentation
    return segmented_out








# =============================================================================
# =============================================================================
# =============================================================================
# TESTING
# =============================================================================
# =============================================================================
# =============================================================================



# =============================================================================
# test nuclear segmentation pipeline
# =============================================================================


# =============================================================================
# # set image path
# im = Path("/Users/joshuahess/Desktop/ROI028_PROSTATETMA020_Probabilities.tiff")
# #Read the image
# image = skimage.io.imread(im,plugin='tifffile')
# # get nuclear channel from probability image
# nuclear_image = image[:,:,0].copy()
#
# # run nuclear segmentation
# (labeled_image,
#  object_count,
#  outline_image,
#  border_excluded_labeled_image,
#  outline_border_excluded_image,
#  size_excluded_labeled_image,
#  outline_size_excluded_image,
#  unedited_labels) = NuclearSegmentation(
#                                      nuclear_image,
#                                      mask=None,
#                                      threshold_smoothing_filter_size=1.8,
#                                      force_separation_smoothing=False,
#                                      separation_smoothing_filter_size=3,
#                                      use_cellprofiler_smoothing=False,
#                                      parallel=False,
#                                      threshold_correction=1.2,
#                                      size_range_min=8,
#                                      size_range_max=15,
#                                      exlcude_outside_size_range=False,
#                                      exclude_border_labels=True,
#                                      unclump_method=UN_INTENSITY,
#                                      watershed_method=WA_INTENSITY,
#                                      automatic_suppression=True,
#                                      advanced=False,
#                                      basic=True
#                                      )
#
#
# # get boundaries
# boundst = skimage.segmentation.find_boundaries(labeled_image)
# final_outt = np.stack([image[:,:,0],image[:,:,1], boundst])
# skimage.io.imsave("/Users/joshuahess/Desktop/testNuclear.tiff",final_outt)
#
# =============================================================================



# =============================================================================
# test membrane segmentation pipeline
# =============================================================================
# =============================================================================
#
# labels_in = labeled_image.copy()
#
# segmented_out = MembraneSegmentation(
#     membrane_image=image[:,:,1],
#     mask=None,
#     objects=labels_in,
#     labels_in=labels_in,
#     distance_to_dilate=5,
#     threshold_correction=1.8,
#     fill_holes=True,
#     method=M_PROPAGATION,
#     discard_edge=True,
#     regularization_factor=1.8
#     )
#
#
# # get boundaries
# boundst = skimage.segmentation.find_boundaries(segmented_out)
# final_outt = np.stack([image[:,:,0],image[:,:,1], boundst])
# skimage.io.imsave("/Users/joshuahess/Desktop/testMembrane.tiff",final_outt)
#
# =============================================================================




# =============================================================================
# import matplotlib.pyplot as plt
# import miaaim.io.imwrite._export
#
# test = Path("/Users/joshuahess/Desktop/test_new/ROI024_PROSTATE_TMA019/probabilities/imc/ilastik")
# root = Path("/Users/joshuahess/Desktop/test_new/ROI024_PROSTATE_TMA019")
#
# ABC = HDISegmentation(probabilities_dir=test,root_folder=root)
# ABC.NuclearSegmentation()
# ABC.MembraneSegmentation()
#
# plt.imshow(ABC.labeled_image)
# plt.imshow(ABC.whole_cell_segmented)
#
# tmp = ABC.whole_cell_segmented.copy()
#
# ABC.ExportSegmentationMask()
#
# skimage.io.imsave("/Users/joshuahess/Desktop/test.tiff",tmp)
#
# miaaim.io.imwrite._export.HDIexporter(tmp, "/Users/joshuahess/Desktop/test.tiff")
#
# =============================================================================
# =============================================================================
# MIAAIM segmentation class
# =============================================================================




class HDISegmentation:
    """
    MIAAIM single-cell segmentation pipeline.

    """

    def __init__(
            self,
            probabilities_dir="ilastik/imc",
            probabilities_image=None,
            image_mask=None,
            nuclear_index = 0,
            membrane_index = 1,
            background_index = 2,
            root_folder=None,
            module_name='hdiseg',
            name=None,
            qc=True
            ):
        """


        Parameters
        ----------
        probabilities_dir : TYPE
            DESCRIPTION.
        raw_image : TYPE, optional
            DESCRIPTION. The default is None.
        raw_image_mask : TYPE, optional
            DESCRIPTION. The default is None.
        nuclear_index : TYPE, optional
            DESCRIPTION. The default is 0.
        membrane_index : TYPE, optional
            DESCRIPTION. The default is 1.
        background_index : TYPE, optional
            DESCRIPTION. The default is 2.
        root_folder : TYPE, optional
            DESCRIPTION. The default is None.
        module_name : TYPE, optional
            DESCRIPTION. The default is 'hdiseg'.
        name : TYPE, optional
            DESCRIPTION. The default is None.
         : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        # create logger format
        FORMAT = '%(asctime)s | [%(pathname)s:%(lineno)s - %(funcName)s() ] | %(message)s'

        # check for root folder name
        if root_folder is not None:
            # make pathlib
            root_folder = Path(root_folder)
        else:
            # use current working directory
            root_folder = Path(os.getcwd())
        self.root_folder=root_folder

        # create names
        self.probabilities_dir = root_folder.joinpath(probabilities_dir)
        self.probabilities_method = probabilities_dir.name

        if name is None:
            self.name = self.probabilities_dir.parent.name
        else:
            self.name = name

        # create direectory specific for process
        segmentation_dir = root_folder.joinpath("segmentation")
        # check if exists already
        if not segmentation_dir.exists():
            # make it
            segmentation_dir.mkdir()
        # create direectory specific for process
        segmentation_dir = segmentation_dir.joinpath(self.name)
        # check if exists already
        if not segmentation_dir.exists():
            # make it
            segmentation_dir.mkdir()
        # create direectory specific for process
        segmentation_dir = segmentation_dir.joinpath(f"hdiseg-{self.probabilities_method}")
        # check if exists already
        if not segmentation_dir.exists():
            # make it
            segmentation_dir.mkdir()
        self.segmentation_dir = segmentation_dir

        # create a docs directory
        docs_dir = Path(root_folder).joinpath("docs")
        # check if exists already
        if not docs_dir.exists():
            # make it
            docs_dir.mkdir()
        self.docs_dir = docs_dir

        # create a parameters directory
        pars_dir = docs_dir.joinpath("parameters")
        # check if exists already
        if not pars_dir.exists():
            # make it
            pars_dir.mkdir()
        self.pars_dir = pars_dir
        # create name of shell command
        yaml_name = os.path.join(
            Path(pars_dir),
            f"miaaim-seg-{module_name}"+f'-{self.probabilities_method}-{self.name}'+".yaml"
            )
        # add to self
        self.yaml_name = yaml_name

        # create a qc directory
        qc_dir = docs_dir.joinpath("qc")
        # check if exists already
        if not qc_dir.exists():
            # make it
            qc_dir.mkdir()
        self.qc_dir = qc_dir

        # create direectory specific for process
        qc_seg_dir = qc_dir.joinpath("segmentation")
        # check if exists already
        if not qc_seg_dir.exists():
            # make it
            qc_seg_dir.mkdir()
        # create direectory specific for process
        qc_seg_dir = qc_seg_dir.joinpath(f"{self.name}")
        # check if exists already
        if not qc_seg_dir.exists():
            # make it
            qc_seg_dir.mkdir()
        self.qc_seg_dir = qc_seg_dir

        qc_seg_name_dir = qc_seg_dir.joinpath(f"hdiseg-{self.probabilities_method}")
        # check if exists already
        if not qc_seg_name_dir.exists():
            # make it
            qc_seg_name_dir.mkdir()
        self.qc_seg_name_dir = qc_seg_name_dir

        # create a qc directory
        prov_dir = docs_dir.joinpath("provenance")
        # check if exists already
        if not prov_dir.exists():
            # make it
            prov_dir.mkdir()
        self.prov_dir = prov_dir
        # create name of logger
        log_name = os.path.join(
            Path(prov_dir),
            f"miaaim-seg-{module_name}"+f'-{self.probabilities_method}-{self.name}'+".log"
            )

        # start yaml log
        self.yaml_log = {}
        # update logger with version number of miaaim
        self.yaml_log.update({"MIAAIM VERSION":miaaim.__version__})

        # check if it exists already
        if Path(log_name).exists():
            # remove it if not resuming
            Path(log_name).unlink()
        # configure log
        logging.basicConfig(filename=log_name,
                                encoding='utf-8',
                                level=logging.DEBUG,
                                format=FORMAT)

        # get logger
        logger = logging.getLogger()
        # writing to stdout
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        # handler.setFormatter(FORMAT)
        logger.addHandler(handler)

        # command to capture print functions to log
        print = logger.info
        self.log_name = None
        self.logger = None


        # create name of shell command
        sh_name = os.path.join(
            Path(prov_dir),
            f"miaaim-seg-{module_name}"+f'-{self.probabilities_method}-{self.name}'+".sh"
            )
        self.sh_name = sh_name

        # set other class attributes
        self.image_mask = skimage.io.imread(Path(image_mask)) if image_mask is not None else None
        self.nuclear_index = nuclear_index
        self.membrane_index = membrane_index
        self.background_index = background_index

        # other attributes to be filled
        self.probabilities_image = None
        self.mask = None
        self.nuclear_segmentation_qc_image = None
        self.labeled_image = None
        self.object_count = None
        self.outline_image = None
        self.border_excluded_labeled_image = None
        self.outline_border_excluded_image = None
        self.size_excluded_labeled_image = None
        self.outline_size_excluded_image = None
        self.unedited_labels = None

        self.qc = qc

        # print first log
        logging.info("MIAAIM SEGMENTATION")
        logging.info(f'MIAAIM VERSION {miaaim.__version__}')
        logging.info(f'METHOD: hdiseg')
        logging.info(f'ROOT FOLDER: {self.root_folder}')
        logging.info(f'PROVENANCE FOLDER: {self.prov_dir}')
        logging.info(f'QC FOLDER: {self.qc_seg_name_dir} \n')

        # Get file extensions for tiff probability images
        tiff_ext = [".tif", ".tiff"]

        # check for input image
        logging.info("Parsing probabilities image...")
        if probabilities_image is None:
            # search directory for probability image
            fs = SearchDir(ending=tiff_ext[0],dir=self.probabilities_dir)
            if not len(fs) > 0:
                # try other tiff extension
                fs = SearchDir(ending=tiff_ext[1],dir=self.probabilities_dir)
            # make sure there are not more than one file in directory
            if len(fs)>1:
                raise Exception(f'More than one TIFF found in {str(self.probabilities_dir)}')
            # get image
            im = fs[0]
        else:
            im = Path(probabilities_image)
        # read and add class attributes
        self.probabilities_image = skimage.io.imread(im)
        self.image_name = im.name.replace(im.suffix,"")
        self.image_name_full = im

        # update yaml file
        self.yaml_log.update({'MODULE':"Segmentation"})
        self.yaml_log.update({'METHOD':"hdseg"})
        self.yaml_log.update({'ImportOptions':{'probabilities_dir':str(self.probabilities_dir),
                                                'probabilities_image':str(self.probabilities_image),
                                                'image_mask':self.image_mask,
                                                'nuclear_index':self.nuclear_index,
                                                'membrane_index':self.membrane_index,
                                                'background_index':self.background_index,
                                                'root_folder':str(self.root_folder),
                                                'module_name':str(module_name),
                                                'name':self.name,
                                                'qc':qc}})


        # update logger
        logging.info(f'\n')
        logging.info("PROCESSING DATA")


    def NuclearSegmentation(
            self,
            mask=False,
            threshold_smoothing_filter_size=2,
            force_separation_smoothing=False,
            separation_smoothing_filter_size=2,
            use_cellprofiler_smoothing=False,
            parallel=False,
            threshold_correction=1.2,
            size_range_min=8,
            size_range_max=15,
            exlcude_outside_size_range=False,
            exclude_border_labels=True,
            unclump_method=UN_INTENSITY,
            watershed_method=WA_INTENSITY,
            automatic_suppression=True,
            advanced=False,
            basic=True
            ):
        """
        Single-nucleus segmentation pipeline taken / adapted from CellProfiler
        identifyPrimaryObjects module.

        Parameters
        ----------
        mask : TYPE, optional
            DESCRIPTION. The default is None.
        threshold_smoothing_filter_size : TYPE, optional
            DESCRIPTION. The default is 2.
        force_separation_smoothing : TYPE, optional
            DESCRIPTION. The default is False.
        separation_smoothing_filter_size : TYPE, optional
            DESCRIPTION. The default is 2.
        use_cellprofiler_smoothing : TYPE, optional
            DESCRIPTION. The default is False.
        parallel : TYPE, optional
            DESCRIPTION. The default is False.
        threshold_correction : TYPE, optional
            DESCRIPTION. The default is 1.2.
        size_range_min : TYPE, optional
            DESCRIPTION. The default is 8.
        size_range_max : TYPE, optional
            DESCRIPTION. The default is 16.
        exlcude_outside_size_range : TYPE, optional
            DESCRIPTION. The default is False.
        exclude_border_labels : TYPE, optional
            DESCRIPTION. The default is True.
        unclump_method : TYPE, optional
            DESCRIPTION. The default is UN_INTENSITY.
        watershed_method : TYPE, optional
            DESCRIPTION. The default is WA_INTENSITY.
        automatic_suppression : TYPE, optional
            DESCRIPTION. The default is True.
        advanced : TYPE, optional
            DESCRIPTION. The default is False.
        basic : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        labeled_image : TYPE
            DESCRIPTION.
        object_count : TYPE
            DESCRIPTION.
        outline_image : TYPE
            DESCRIPTION.
        border_excluded_labeled_image : TYPE
            DESCRIPTION.
        outline_border_excluded_image : TYPE
            DESCRIPTION.
        size_excluded_labeled_image : TYPE
            DESCRIPTION.
        outline_size_excluded_image : TYPE
            DESCRIPTION.
        unedited_labels : TYPE
            DESCRIPTION.

        """
        # update logger
        self.yaml_log.update({'ProcessingSteps':[]})
        self.yaml_log['ProcessingSteps'].append({"NuclearSegmentation":{'mask':mask,
                                        'threshold_smoothing_filter_size':threshold_smoothing_filter_size,
                                        'force_separation_smoothing':force_separation_smoothing,
                                        'separation_smoothing_filter_size':separation_smoothing_filter_size,
                                        'use_cellprofiler_smoothing':use_cellprofiler_smoothing,
                                        'parallel':parallel,
                                        'threshold_correction':threshold_correction,
                                        'size_range_min':size_range_min,
                                        'size_range_max':size_range_max,
                                        'exlcude_outside_size_range':exlcude_outside_size_range,
                                        'exclude_border_labels':exclude_border_labels,
                                        'unclump_method':unclump_method,
                                        'watershed_method':watershed_method,
                                        'automatic_suppression':automatic_suppression,
                                        'advanced':advanced,
                                        'basic':basic}})


        # check for mask
        if mask:
            mask = self.mask.copy()
        else:
            mask=None

        # get nuclear channel from probability image
        nuclear_image = self.probabilities_image[:,:,self.nuclear_index].copy()

        # run nuclear segmentation
        (labeled_image,
         object_count,
         outline_image,
         border_excluded_labeled_image,
         outline_border_excluded_image,
         size_excluded_labeled_image,
         outline_size_excluded_image,
         unedited_labels) = NuclearSegmentation(
             nuclear_image=nuclear_image,
             mask=mask,
             threshold_smoothing_filter_size=threshold_smoothing_filter_size,
             force_separation_smoothing=force_separation_smoothing,
             separation_smoothing_filter_size=separation_smoothing_filter_size,
             use_cellprofiler_smoothing=use_cellprofiler_smoothing,
             parallel=parallel,
             threshold_correction=threshold_correction,
             size_range_min=size_range_min,
             size_range_max=size_range_max,
             exlcude_outside_size_range=exlcude_outside_size_range,
             exclude_border_labels=exclude_border_labels,
             unclump_method=unclump_method,
             watershed_method=watershed_method,
             automatic_suppression=automatic_suppression,
             advanced=advanced,
             basic=basic
             )

        # get boundaries
        # boundst = skimage.segmentation.find_boundaries(labeled_image)
        self.nuclear_segmentation_qc_image = np.stack(
            [self.probabilities_image[:,:,self.nuclear_index],
             self.probabilities_image[:,:,self.membrane_index],
             outline_image]
            )
        # set class attributes
        self.labeled_image = labeled_image
        self.object_count = object_count
        self.outline_image = outline_image
        self.border_excluded_labeled_image = border_excluded_labeled_image
        self.outline_border_excluded_image = outline_border_excluded_image
        self.size_excluded_labeled_image = size_excluded_labeled_image
        self.outline_size_excluded_image = outline_size_excluded_image
        self.unedited_labels = unedited_labels

    def MembraneSegmentation(
            self,
            mask=False,
            distance_to_dilate=3,
            threshold_correction=1.8,
            fill_holes=True,
            method=M_PROPAGATION,
            discard_edge=True,
            regularization_factor=1.8
            ):
        """
        Whole cell segmentation pipeline taken / adapted from CellProfiler
        identifySecondaryObjects module.

        Parameters
        ----------
        mask : TYPE, optional
            DESCRIPTION. The default is None.
        distance_to_dilate : TYPE, optional
            DESCRIPTION. The default is 5.
        threshold_correction : TYPE, optional
            DESCRIPTION. The default is 1.8.
        fill_holes : TYPE, optional
            DESCRIPTION. The default is True.
        method : TYPE, optional
            DESCRIPTION. The default is M_PROPAGATION.
        discard_edge : TYPE, optional
            DESCRIPTION. The default is True.
        regularization_factor : TYPE, optional
            DESCRIPTION. The default is 1.8.

        Returns
        -------
        segmented_out : TYPE
            DESCRIPTION.

        """
        self.yaml_log['ProcessingSteps'].append({"MembraneSegmentation":{'mask':mask,
                                        'distance_to_dilate':distance_to_dilate,
                                        'threshold_correction':threshold_correction,
                                        'fill_holes':fill_holes,
                                        'method':method,
                                        'discard_edge':discard_edge,
                                        'regularization_factor':regularization_factor}})

        # check for mask
        if mask:
            mask = self.mask.copy()
        else:
            mask=None

        # get membrane channel from probability image
        membrane_image = self.probabilities_image[:,:,self.membrane_index ].copy()
        # get nuclear segmentation labels
        labels_in = self.labeled_image.copy()

        # run membrane based segmentation using the previous nuclear segmentation
        segmented_out = MembraneSegmentation(
            membrane_image=membrane_image,
            mask=mask,
            objects=labels_in,
            labels_in=labels_in,
            distance_to_dilate=distance_to_dilate,
            threshold_correction=threshold_correction,
            fill_holes=fill_holes,
            method=method,
            discard_edge=discard_edge,
            regularization_factor=regularization_factor
            )

        # get boundaries
        outline_image = centrosome.outline.outline(segmented_out)
        # bcreate overlayed image for qc
        self.membrane_segmentation_qc_image = np.stack(
            [self.probabilities_image[:,:,self.nuclear_index],
             self.probabilities_image[:,:,self.membrane_index],
             outline_image]
            )

        # store the final segmentation
        self.whole_cell_segmented = segmented_out
        # return final whole cell segmentation
        return segmented_out

    def _exportSegmentationMask(
            self,
            out_dir,
            out_name
            ):
        """
        Helper function to export final segmentation mask.

        Parameters
        ----------
        out_dir : TYPE
            DESCRIPTION.
        out_name : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # create pathlib objects
        out_dir = Path(out_dir)
        outfile = out_dir.joinpath(out_name)
        # save image
        skimage.io.imsave(outfile, self.whole_cell_segmented.astype(np.uint16))

    def _exportSegmentationMaskQC(
            self,
            out_dir,
            out_name
            ):
        """
        Helper function to export final segmentation mask.

        Parameters
        ----------
        out_dir : TYPE
            DESCRIPTION.
        out_name : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # create pathlib objects
        out_dir = Path(out_dir)
        outfile = out_dir.joinpath(out_name)
        # save image
        skimage.io.imsave(outfile, self.membrane_segmentation_qc_image)


    def ExportSegmentationMask(
            self,
            out_dir=None,
            out_name=None
            ):
        """


        Parameters
        ----------
        out_dir : TYPE, optional
            DESCRIPTION. The default is None.
        out_name : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """

        self.yaml_log['ProcessingSteps'].append({"ExportSegmentationMask":{'out_dir':out_dir,
                                        'out_name':out_name}})

        # check for defaults
        if out_dir is None:
            # set as the output directory from initialized
            out_dir = self.segmentation_dir
        else:
            out_dir = Path(out_dir)
        if out_name is None:
            out_name = self.image_name+"_mask.tiff"
        else:
            out_name = out_name

        # write as tiff
        self._exportSegmentationMask(out_dir,out_name)


    def ExportSegmentationMaskQC(
            self,
            out_dir=None,
            out_name=None
            ):
        """


        Parameters
        ----------
        out_dir : TYPE, optional
            DESCRIPTION. The default is None.
        out_name : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        if out_dir is None:
            # set as the output directory from initialized
            out_dir = self.qc_seg_name_dir
        else:
            out_dir = Path(out_dir)
        if out_name is None:
            out_name = self.image_name+"_maskQC.tiff"
        else:
            out_name = out_name
        # write as tiff
        self._exportSegmentationMaskQC(out_dir,out_name)

    def _exportYAML(self):
        """Function to export yaml log to file for documentation
        """
        logging.info(f'Exporting {self.yaml_name}')
        # open file and export
        with open(self.yaml_name, 'w') as outfile:
            yaml.dump(self.yaml_log, outfile, default_flow_style=False,sort_keys=False)

    def _exportSH(self):
        """Function to export sh command to file for documentation
        """
        logging.info(f'Exporting {self.sh_name}')
        # get name of the python path and cli file
        proc_fname = os.path.join(Path(_parse.__file__).parent,"_cli_seg.py")
        # get path to python executable
        # create shell command script
        with open (self.sh_name, 'w') as rsh:
            rsh.write(f'''\
        #! /bin/bash
        {sys.executable} {proc_fname} --pars {self.yaml_name}
        ''')

    def QC(self):
        """Function to export QC metrics to file for documentation
        """
        # log
        logging.info("QC: extracting quality control information")
        self.yaml_log['ProcessingSteps'].append("QC")

        # export QC information
        # check for qc
        if self.qc:
            # export processed masks
            self.ExportQCMask()
            # export subsampled masks
            self.ExportSubsampleQCMask()

        # provenance
        self._exportYAML()
        self._exportSH()
        # close the logger
        self.logger.handlers.clear()





#
