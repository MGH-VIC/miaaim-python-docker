# Module for high-dimensional imaging data importing
# Developer: Joshua M. Hess, BSc
# Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

# Import external modules
from pathlib import Path
import os
import logging

# Import custom modules
from miaaim.io.imread._cyt_reader import CYTreader
from miaaim.io.imread._imzml_reader import imzMLreader
from miaaim.io.imread._nifti1_reader import NIFTI1reader
from miaaim.io.imread._prep_bridger import PrepBridge


# Create class object to store high-dimensional imaging data
class HDIreader:
    """Class for importing high-dimensional imaging data or histology data."""

    def __init__(
        self,
        path_to_data=None,
        path_to_markers=None,
        flatten=False,
        subsample=True,
        method="default",
        mask=None,
        save_mem=False,
        data=None,
        image=None,
        channels=None,
        filename=None,
        **kwargs
    ):
        """Initialize class to store data in. Ensure appropriate file format
        and return a data object with pixel table.

        path_to_data: path to imaging data (Ex: path/mydata.extension)
        path_to_markers: path to marker list (Ex: path/mymarkers.csv or None)
        flatten: True to return a flattened pixel data table for dimension reduction
        mask: Path to tif mask to use for selecting a region to focus on in downstream preparation
        **kwargs: - input to SubsetCoordinates utils function
        """

        # Initialize objects
        self.hdi = None

        # Set the file extensions that we can use with this class
        all_ext = [
            ".ome.tif",
            ".ome.tiff",
            ".tif",
            ".tiff",
            ".h5",
            ".hdf5",
            ".imzML",
            ".nii",
        ]
        # Get file extensions for cytometry files
        cyt_ext = [".ome.tif", ".ome.tiff", ".tif", ".tiff", ".h5", ".hdf5"]
        # Get file exntensions for h(df)5 files
        imzML_ext = [".imzML"]
        # Get file extensions for NIFTI1 files
        nii_ext = [".nii"]

        # create dictionary for import options
        self.import_kwargs = {"path_to_markers":path_to_markers,
                                "flatten":flatten,
                                "subsample":subsample,
                                "method":method,
                                "mask":mask,
                                "save_mem":save_mem}
        # update with keywords
        self.import_kwargs.update(**kwargs)
        # create storage for padding and resizing
        self.padding = None
        self.target_size = None

        if path_to_data is not None:
            # Create a pathlib object for the path_to_cyt
            path_to_data = Path(path_to_data)

            # check if the path is a directory
            # this portion of the script will serve as the reading function for
            # file types that have paired files in folders (e.g. imzML MSI data
            # and some spatial transcriptomics do this)
            # within each data reading function are compatible functions for parsing
            # folders -- here we go ahead and parse the file extensions so that new
            # data types can be easily added
            if path_to_data.is_dir():
                # parse the inputs collectively
                files = [x for x in path_to_data.rglob('*')]
                # create list for additional files to store (like ibds)
                pairs = []
                # iterate through the files
                for f in files:
                    if str(f).endswith(tuple(all_ext)):
                        # check for mask conditional
                        if ((mask) and ("mask" in str(f.stem))):
                            # check for the mask suffix
                            mask = f
                            # print update
                            logging.info("mask: "+str(mask))
                        else:
                            # return the path -- note this assumes that you only have
                            # a single image in the directory!
                            path_to_data = f
                    # otherwise add the extra files to the list of extras
                    else:
                        # append list
                        pairs.append(f)
                
            # Check to see if there is a valid file extension for this class
            if str(path_to_data).endswith(tuple(cyt_ext)):
                # Read the data with CYTreader
                self.hdi = CYTreader(
                    path_to_cyt=path_to_data,
                    path_to_markers=path_to_markers,
                    subsample=subsample,
                    flatten=flatten,
                    mask=mask,
                    method=method,
                    **kwargs
                )

            # Otherwise read imzML file
            elif str(path_to_data).endswith(tuple(imzML_ext)):
                # Read the data with imzMLreader (CURRENTLY DOES NOT SUPPORT A MASK -- set default to None in class object)
                self.hdi = imzMLreader(
                    path_to_imzML=path_to_data,
                    path_to_markers=path_to_markers,
                    subsample=subsample,
                    flatten=flatten,
                    mask=mask,
                    method=method,
                    **kwargs
                )

            # Otherwise read nifti file
            elif str(path_to_data).endswith(tuple(nii_ext)):
                # Read the data with NIFTI1reader
                self.hdi = NIFTI1reader(
                    path_to_nifti=path_to_data,
                    path_to_markers=path_to_markers,
                    subsample=subsample,
                    flatten=flatten,
                    mask=mask,
                    method=method,
                    **kwargs
                )
            # If none of the above print an update and an error
            else:
                # Raise an error saying that the file extension is not recognized
                logging.error(f'Unable to load {path_to_data}')

        # check for path to data
        elif path_to_data is None:
            # use prep bridger
            self.hdi = PrepBridge(
                data=data,
                image=image,
                mask=mask,
                channels=channels,
                flatten=flatten,   # set flatten to true for reload
                filename=filename,
                subsample=subsample,
                method=method,
                **kwargs
            )

        # If none of the above print an update and an error
        else:
            # Raise an error saying that the file extension is not recognized
            logging.error("Unable to load data.")

        # check to see if saving storage
        if save_mem:
            # set image arrays to none
            self.hdi.data.image = None
            # self.hdi.data.mask = None

    def Slice(self, channels):
        """Slice pixel table and image to include only the given channels
        """
        # log
        logging.info("Slicing image array and pixel table.")
        ### !!!! ### you will have to make this a function in each
        #### of the hdi subclasses then merge it in this one so metadata is preserved
        self.hdi = self.hdi.Slice(channels=channels)








































#


























#
