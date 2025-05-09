# Module for cytometry data parsing
# Developer: Joshua M. Hess, BSc
# Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

# Import external modules
from pathlib import Path
import os
import h5py
import skimage.io
import numpy as np
import pandas as pd
import scipy.sparse
import random
import logging

# Import custom modules
from miaaim.io.imread._tif_reader import TIFreader
from miaaim.io.imread._h5_reader import H5reader
from miaaim.io.imread._utils import ReadMarkers, FlattenZstack


# Create a class object to store attributes and functions in
class CYTreader:
    """Class for parsing and storing cytometry data that is in the ome.tif(f),
    h(df)5, or tif(f) format.

    path_to_cyt: string indicating path to cytometry file (Ex: 'path/CYTdata.ome.tif')
    """

    def __init__(
        self, path_to_cyt, path_to_markers, flatten, subsample, mask=None, **kwargs
    ):
        """Initialize class to store data in. Ensure appropriate file format
        and return a data object with pixel table.
        """

        # Create a pathlib object for the path_to_cyt
        path_to_cyt = Path(path_to_cyt)

        # Set the file extensions that we can use with this class
        all_ext = [".ome.tif", ".ome.tiff", ".tif", ".tiff", ".h5", ".hdf5"]
        # Get file extensions for ome.tif(f) or tif(f) files
        tif_ext = [".ome.tif", ".ome.tiff", ".tif", ".tiff"]
        # Get file exntensions for h(df)5 files
        h5_ext = [".h5", ".hdf5"]

        # Check to make sure the string is a valid path
        if not os.path.exists(str(path_to_cyt)):
            logging.error("Not a valid path. Try again")
        else:
            logging.info("Valid path...")
            # Check to see if there is a valid file extension for this class
            if str(path_to_cyt).endswith(tuple(all_ext)):
                logging.info("Valid file extension...")
                logging.info(f'file name: {str(path_to_cyt)}')

                # Read the data by searching through the extensions
                if str(path_to_cyt).endswith(tuple(tif_ext)):
                    # Read the ome.tif(f) or tif(f)
                    self.data = TIFreader(path_to_cyt)

                elif str(path_to_cyt).endswith(tuple(h5_ext)):
                    # Read h(df)5 data and return the parsed data
                    self.data = H5reader(path_to_cyt)
            else:
                logging.error("Not a valid file extension")

        # Create an object for a filtered/processed working
        self.data.processed_image = None
        # create object for processed mask
        self.data.processed_mask = None
        # create object for subsampled mask
        self.data.subsampled_mask = None

        # Add the shape of the image to the class object for future use
        self.data.image_shape = self.data.image.shape
        # Get the array size for the image
        self.data.array_size = (self.data.image_shape[0], self.data.image_shape[1])
        # get the number of channels
        if len(self.data.image.shape) > 2:
            # Get the number of channels in the imaging data
            self.data.num_channels = self.data.image_shape[2]
        # Otherwise just create a single entry for single-channel image
        else:
            # single channel
            self.data.num_channels = 1

        # Check to see if the mask exists
        if mask is not None:
            # Check to see if the mask is a path (string)
            if isinstance(mask, Path):
                ##############Change in future to take arbitrary masks not just tiff??################
                mask = skimage.io.imread(str(mask), plugin="tifffile")
            # Ensure the mask is a sparse boolean array
            mask = scipy.sparse.coo_matrix(mask, dtype=np.bool_)

        # Add the mask to the class object -- even if it is none. Will not be applied to image yet
        self.data.mask = mask

        # Check for a marker list
        if path_to_markers is not None:
            # Read the channels list
            channels = ReadMarkers(path_to_markers)
        else:
            # Check to see if the image shape includes a channel (if not, it is one channel)
            if self.data.num_channels > 2:
                # Create a numbered marker list based on channel number
                channels = [str(num) for num in range(0, self.data.num_channels)]
            # Otherwise just create a single entry for single-channel image
            else:
                # Create a numbered marker list based on channel number
                channels = [str(num) for num in range(0, 1)]

            # Add the channels to the class object
            self.data.channels = channels

        # Check to see if creating a pixel table (used for dimension reduction)
        if flatten:
            # Create a pixel table and extract the full list of coordinates being used
            pix, coords, sub_mask = FlattenZstack(
                z_stack=self.data.image,
                z_stack_shape=self.data.image_shape,
                mask=self.data.mask,
                subsample=subsample,
                **kwargs
            )
            # Add the pixel table to our object
            self.data.pixel_table = pd.DataFrame(
                pix.values, columns=channels, index=pix.index
            )
            # Clear the pixel table object to save memory
            pix = None
            # Check to see if we subsampled
            if subsample is None:
                # Assign subsampled coordinates to be false
                self.data.sub_coordinates = None
            else:
                # Add pixel coordinates to the class object (similar to imzML parser) subsampled
                self.data.sub_coordinates = list(self.data.pixel_table.index)
                # add subsampled mask
                self.data.subsampled_mask = sub_mask

            # Assign full coordinates to be coords
            self.data.coordinates = coords

        else:
            # Create a pixel table as None
            self.data.pixel_table = None
            # Set the pixel coordinates as None
            self.data.coordinates = None

        # Add the filename to the data object
        self.data.filename = path_to_cyt
        # update the data type
        self.data.hdi_type = "array"

        # Print an update on the parsing of cytometry data
        logging.info("Finished parsing data")



    def Slice(self,channels):
        """Subset channels of ndarray and associated channels and class
        attributes.

        Parameters
        ----------
        channels: integer or list of integers
            Indicates which indices to extract from c axis of image
        """
        # Print an update on the parsing of cytometry data
        self.data.image = self.data.image[:,:,channels] if self.data.image is not None else None
        self.data.pixel_table = self.data.pixel_table.iloc[:,channels] if self.data.pixel_table is not None else None
        self.data.channels = [self.data.channels[i] for i in channels] if self.data.channels is not None else None
        self.data.image_shape = (self.data.image_shape[0], self.data.image_shape[1],len(channels)) if self.data.image_shape is not None else None
















#
