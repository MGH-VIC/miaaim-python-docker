# Module for creating
# Developer: Joshua M. Hess, BSc
# Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

# Import external modules
from pathlib import Path
import os
import numpy as np
import pandas as pd
import scipy.sparse
import random
import logging

# Import custom modules
from miaaim.io.imread._utils import FlattenZstack


# Create a class object to store attributes and functions in
class PrepBridge:
    """Class for parsing and storing cytometry data with all parameters
    pre-assigned.
    """

    def __init__(
        self,
        data,
        image,
        mask,
        channels,
        flatten,
        filename,
        subsample=True,
        method='default',
        **kwargs
    ):
        """Initialize class to store data in. Ensure appropriate file format
        and return a data object with pixel table.
        """

        # create storage for padding and resizing
        self.padding = None
        self.target_size = None

        # set data
        self.data = data
        # Create an object for a filtered/processed working
        self.data.processed_image = None
        # create object for processed mask
        self.data.processed_mask = None
        # Initialize subsampled mask
        self.data.subsampled_mask = None


        self.data.image = image

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
            # Ensure the mask is a sparse boolean array
            mask = scipy.sparse.coo_matrix(mask, dtype=np.bool_)

        # Add the mask to the class object -- even if it is none. Will not be applied to image yet
        self.data.mask = mask
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
        self.data.filename = filename
        # update the data type
        self.data.hdi_type = "raster"

        # Print an update on the parsing of cytometry data
        logging.info("Created hdi data")

    def Slice(self,channels):
        """Subset channels of ndarray and associated channels and class
        attributes.

        Parameters
        ----------
        channels: integer or list of integers
            Indicates which indices to extract from c axis of image
        """
        self.data.image = self.data.image[:,:,channels] if self.data.image is not None else None
        self.data.pixel_table = self.data.pixel_table.iloc[:,channels] if self.data.pixel_table is not None else None
        self.data.channels = [self.data.channels[i] for i in channels] if self.data.channels is not None else None
        self.data.image_shape = (self.data.image_shape[0], self.data.image_shape[1],len(channels)) if self.data.image_shape is not None else None
