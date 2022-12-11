# Module for ome.tif(f) and tif(f) imaging data parsing
# Developer: Joshua M. Hess, BSc
# Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

# Import external modules
from pathlib import Path
import skimage.io
import numpy as np


# Define class object
class TIFreader:
    """TIF cytometry data reader using scikit image."""

    def __init__(self, path_to_tif):
        """Initialize the class by using the path to the image.

        path_to_tif: Path to tif image (Ex: path/to/image.extension)
        """

        # Initialize the objects in this class
        self.image = None

        # Create a pathlib object for the path_to_tif
        path_to_tif = Path(path_to_tif)

        # Read tif(f) or ome.tif(f) data and return the parsed data
        im = skimage.io.imread(str(path_to_tif), plugin="tifffile")
        # Check to see if the number of channels is greater than one
        im_shape = im.shape
        # check to see if the image is a tiff hyperstack based on the shape
        if len(im_shape) == 4:
            # ravel along the last axis
            im = np.concatenate([im[:,:,:,i] for i in range(im.shape[3])])
            # recalculate the image shape
            im_shape = im.shape
        # Check to see if the image is considered xyc or just xy(single channel)
        # Note: skimage with tifffile plugin reads channel numbers of 1 as xy array,
        # and reads images with 3 and 4 channels in the correct order. Channel numbers
        # of 2 or >5 need axis swapping
        if len(im_shape) > 2:
            ##########This will fail if the array is a 3 or 4 channel image with 5 pixels in the x direction...shouldnt happen##########
            # Check if channel numbers are 3 or 4
            if (im_shape[2] == 3) or (im_shape[2] == 4):
                pass
            else:
                # If number of channels is less than two then swap the axes to be zyxc
                im = np.swapaxes(im, 0, 2)
                # Swap the axes to be in the order zyxc
                im = np.swapaxes(im, 0, 1)
        # Assign the data to the class
        self.image = im







#
# def getMetadata(path,commit):
#     """From s3 segmenter tiff file metadata.
#     """
#     with tifffile.TiffFile(path) as tif:
#         if not tif.ome_metadata:
#             try:
#                 x_res_tag = tif.pages[0].tags['XResolution'].value
#                 y_res_tag = tif.pages[0].tags['YResolution'].value
#                 physical_size_x = x_res_tag[0] / x_res_tag[1]
#                 physical_size_y = y_res_tag[0] / y_res_tag[1]
#             except KeyError:
#                 physical_size_x = 1
#                 physical_size_y = 1
#             metadata_args = dict(
#             pixel_sizes=(physical_size_y, physical_size_x),
#             pixel_size_units=('µm', 'µm'),
#             software= 's3segmenter v' + commit
#             )
#         else:
#             metadata=ome_types.from_xml(tif.ome_metadata)
#             metadata = metadata.images[0].pixels
#             metadata_args = dict(
#             pixel_sizes=(metadata.physical_size_y,metadata.physical_size_x),
#             pixel_size_units=('µm', 'µm'),
#             software= 'MIAAIM version ' + str(commit)
#             )
#         return metadata_args
# import tifffile
# test = "/Users/joshuahess/Desktop/test_new/ROI023_LIVER_D12/input/imc/ROI023_LIVER D12.ome.tiff"
#
# met = getMetadata(path=test,commit='1')
#
#
# tif = tifffile.TiffFile(test)
# tif.pages[0].tags['XResolution'].value
#                 x_res_tag = tif.pages[0].tags['XResolution'].value
#                 y_res_tag = tif.pages[0].tags['YResolution'].value
#                 physical_size_x = x_res_tag[0] / x_res_tag[1]
#                 physical_size_y = y_res_tag[0] / y_res_tag[1]
#
# for k in tif.pages[0].tags.keys():
#     print(tif.pages[0].tags[k])
# tif.pages[0].tags[256]
#




# import skimage.io
# import numpy as np
# import tifffile
#
#
# tif = tifffile.TiffFile(path_to_modality)
# tif.pages[0].tags
# #                 x_res_tag = tif.pages[0].tags['XResolution'].value
# #                 y_res_tag = tif.pages[0].tags['YResolution'].value
# #                 physical_size_x = x_res_tag[0] / x_res_tag[1]
# #                 physical_size_y = y_res_tag[0] / y_res_tag[1]
# #
#
# len(tif.pages)
#
# c = tif.pages[1].tags['PageName'].value
#
#
#
#  channels
#
#
# for k in tif.pages[1].tags.keys():
#     print(tif.pages[1].tags[k])
# tif.pages[0].tags[256]
#






















#
