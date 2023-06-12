# Module for imzML parsing of imaging mass spectrometry (IMS) data
# Developer: Joshua M. Hess, BSc
# Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

# Import external modules
from pathlib import Path
import os
from pyimzml.ImzMLParser import getionimage
from pyimzml.ImzMLParser import ImzMLParser
import numpy as np
import pandas as pd
from operator import itemgetter
import scipy
import skimage
import skimage.io
import h5py
import nibabel as nib
import warnings
import logging
logging.captureWarnings(True)

# Import custom modules
from miaaim.io.imread._utils import SubsetCoordinates


# Create a class object to store attributes and functions in
class imzMLreader:
    """Class for parsing and storing IMS data that is in the imzML format. Depends
    on and contains the pyimzML python package distributed from the Alexandrov team:
    https://github.com/alexandrovteam/pyimzML in a data object.

    path_to_imzML: string indicating path to imzML file (Ex: 'path/IMSdata.imzML')
    """

    def __init__(
        self,
        path_to_imzML,
        flatten,
        subsample=True,
        method="default",
        mask=None,
        path_to_markers=None,
        **kwargs
    ):
        """Initialize class to store data in. Ensure appropriate file format
        and return a data object with pixel table.
        """

        # Create a pathlib object for the path_to_imzML
        path_to_imzML = Path(path_to_imzML)
        # check to see if the input is a folder
        if path_to_imzML.is_dir():
            # parse the inputs as ibd and imzml
            path_to_ibd = [x for x in path_to_imzML.rglob('*.ibd')][0]
            path_to_imzML = [x for x in path_to_imzML.rglob('*.imzML')][0]

        # Set the file extensions that we can use with this class
        ext = [".imzML"]

        # Check to make sure the string is a valid path
        if not os.path.exists(str(path_to_imzML)):
            logging.error("Not a valid path. Try again")
        else:
            logging.info("Valid path...")
            # Check to see if there is a valid file extension for this class
            if str(path_to_imzML).endswith(tuple(ext)):
                logging.info("Valid file extension...")
                logging.info(f'file name: {str(path_to_imzML)}')

                # Read imzML and return the parsed data
                self.data = ImzMLParser(str(path_to_imzML))
                logging.info("Finished parsing imzML")
            else:
                logging.error("Not a valid file extension")

        # Add the image size to the data object
        self.data.array_size = (
            self.data.imzmldict["max count of pixels y"],
            self.data.imzmldict["max count of pixels x"],
        )

        # Check to see if the mask exists
        if mask is not None:
            # Check to see if the mask is a path (string)
            if isinstance(mask, Path):
                ##############Change in future to take arbitrary masks not just tiff################
                mask = skimage.io.imread(str(mask), plugin="tifffile")
            # Ensure the mask is a sparse boolean array
            mask = scipy.sparse.coo_matrix(mask, dtype=np.bool_)

        # Add the mask to the class object -- even if it is none. Will not be applied to image yet
        self.data.mask = mask
        # Create an object for a filtered/processed working
        self.data.processed_image = None
        # create object for processed mask
        self.data.processed_mask = None
        # create subsampled mask object
        self.data.subsampled_mask = None

        # Check to see if creating a pixel table (used for dimension reduction)
        if flatten:

            # Check to see if we are using a mask
            if mask is not None:

                # Ensure that the mask is boolean
                mask = np.array(mask.toarray(), dtype=np.bool_)
                # Get the coordinates where the mask is
                where = np.where(mask)
                # Create list of tuples where mask coordinates are (1-indexed) -- form (x,y,z) with z=1 (same as imzML)
                coords = list(
                    zip(
                        where[1] + 1, where[0] + 1, np.ones(len(where[0]), dtype=np.int)
                    )
                )
                # intersect the mask coordinates with the IMS coordinates from imzML parser
                mask_coords = list(set(coords) & set(self.data.coordinates))

                # Reorder the mask coordinates for F style column major format (imzml format)
                mask_coords = sorted(mask_coords, key=itemgetter(0, 1))

                # Clear the old coordinates for memory
                coords, where, mask = None, None, None

                # Zip the coordinates into dictionary with list index (Faster with itemgetter)
                full_coords_dict = dict(
                    zip(self.data.coordinates, range(0, len(self.data.coordinates)))
                )
                # Find the indices of the mask coordinates -- need for creating dataframe
                coords_idx = list(itemgetter(*mask_coords)(full_coords_dict))

                # Remove the dictionary to save memory
                full_coords_dict = None

                # Reset the coordinates object to be only the mask coordinates
                self.data.coordinates = mask_coords

            # Otherwise create a coords_idx from the full list of coordinates
            else:
                # Create list
                coords_idx = [x for x in range(len(self.data.coordinates))]

            # Check to see if subsampling
            if subsample:
                # check to see if default subsampling
                if method == "default":
                    # check for below 50k pixels
                    if len(self.data.coordinates) < 50000:
                        # set subsampling to none
                        warnings.warn("subsampling set to default, but there are less than 50k pixels...")
                        # raise warning to revert to no subsampling
                        warnings.warn("subsampling reverted to None")

                        # Keep the full list of coordinates
                        coords = self.data.coordinates
                        # Add the subset coordinates as None
                        self.data.sub_coordinates = None

                    # check for 50k-100k
                    elif 50000 < sz < 100000:
                        # Subset the coordinates using custom function
                        sub_mask, sub_coords = SubsetCoordinates(
                            coords=self.data.coordinates, array_size=self.data.array_size, method="pseudo_random", n=0.55, grid_spacing=(2, 2)
                        )
                    # check for 50k-100k
                    elif 100000 < sz < 150000:
                        # Subset the coordinates using custom function
                        sub_mask, sub_coords = SubsetCoordinates(
                            coords=self.data.coordinates, array_size=self.data.array_size, method="pseudo_random", n=0.15, grid_spacing=(3, 3)
                        )
                    elif sz > 150000:
                        # Subset the coordinates using custom function
                        sub_mask, sub_coords = SubsetCoordinates(
                            coords=coords, array_size=self.data.array_size, method="grid", grid_spacing=(3, 3)
                        )
                # otherwise use the given method
                else:
                    # Use the coordinates for subsampling
                    sub_mask, coords = SubsetCoordinates(
                        coords=self.data.coordinates,
                        array_size=self.data.array_size,
                        **kwargs
                    )

                # Alter the order to be in column major format Fortran style
                coords = sorted(coords, key=itemgetter(0, 1))

                # add subsampled mask
                self.data.subsampled_mask = sub_mask

                # Get the indices now of these coordinates from the coords_idx
                # coords_idx = [self.data.coordinates.index(x) for x in coords]
                # Zip the coordinates into dictionary with list index (Faster with itemgetter)
                tmp_coords_dict = dict(
                    zip(self.data.coordinates, range(0, len(self.data.coordinates)))
                )
                # Find the indices of the mask coordinates -- need for creating dataframe
                coords_idx = list(itemgetter(*coords)(tmp_coords_dict))

                # Clear the coordinates dictionary to save memory
                tmp_coords_dict = None

                # Add the subset coordinates to our object
                self.data.sub_coordinates = coords

            # Otherwise there is no subsampling so leave the coordinates as they are
            else:
                # Keep the full list of coordinates
                coords = self.data.coordinates
                # Add the subset coordinates as None
                self.data.sub_coordinates = None

            # Create numpy array with cols = m/zs an rows = pixels (create pixel table)
            tmp = np.empty([len(coords), len(self.data.getspectrum(0)[0])])

            # iterate through pixels and add to the array
            logging.info("Fetching Spectrum Table...")
            for i, (x, y, z) in enumerate(coords):
                # Get the coordinate index
                idx = coords_idx[i]
                # Now use the index to extract the spectrum
                mzs, intensities = self.data.getspectrum(idx)
                # Use the original i index to add to the array the data
                tmp[i, :] = intensities
                # Clear memory by removing mzs and intensities
                mzs, intensities = None, None

            # Create a pandas dataframe from numpy array
            tmp_frame = pd.DataFrame(
                tmp, index=coords, columns=self.data.getspectrum(0)[0]
            )
            # Delete the temporary object to save memory
            tmp = None
            # Assign the data to an array in the data object
            self.data.pixel_table = tmp_frame

            # Get the image shape of the data
            self.data.image_shape = (
                self.data.imzmldict["max count of pixels y"],
                self.data.imzmldict["max count of pixels x"],
                self.data.pixel_table.shape[1],
            )
        else:
            # Create a pixel table as None
            self.data.pixel_table = None
            # Set the image shape as None
            self.data.image_shape = None

        # Add the filename to the data object
        self.data.filename = path_to_imzML
        # Add None to the data image (not currently parsing full array)
        self.data.image = None
        # get number of channels
        # here, we assume that each of the pixels has the same number of
        # m/z peaks, so we can take only the first element of the list
        self.data.num_channels = self.data.mzLengths[0]
        # update the data type
        self.data.hdi_type = "raster"
        # Print an update that the import is finished
        logging.info("Finished")

    def SubsetRange(self, range):
        """Subset an IMS peak list to fall between a range of values.

        range: tuple indicating range (Ex (400,1000)). Note for memory reasons
        the PixelTable is overwritten, and a new subset of the peak list is created.
        """
        # Print an update on the parsing of cytometry data
        logging.info(f"Subsetting peak range to m/z {range}")
        # Get the lowest value
        low = next(
            x for x, val in enumerate(self.data.pixel_table.columns) if val >= range[0]
        )
        # Get the highest value
        hi = [n for n, i in enumerate(self.data.pixel_table.columns) if i <= range[1]][
            -1
        ]
        # Assign the new peak list to the pixel_table (add +1 because python isnt inclusive)
        self.data.image = self.data.image[:,:,low : hi + 1] if self.data.image is not None else None
        self.data.pixel_table = self.data.pixel_table.iloc[:, low : hi + 1]
        self.data.channels = self.data.channels[low : hi + 1]
        self.data.image_shape = (
            self.data.imzmldict["max count of pixels y"],
            self.data.imzmldict["max count of pixels x"],
            self.data.pixel_table.shape[1],
        )

    def Slice(self,channels):
        """Subset channels of ndarray and associated channels and class
        attributes.

        Parameters
        ----------
        channels: integer or list of integers
            Indicates which indices to extract from c axis of image
        """
        self.data.image = self.data.image[:,:,channels] if self.data.image is not None else None
        self.data.pixel_table = self.data.pixel_table.iloc[:,channels] if self.data.pixel_table.iloc[:,channels] is not None else None
        self.data.channels = [self.data.channels[i] for i in channels] if self.data.channels is not None else None
        self.data.image_shape = (self.data.image_shape[0], self.data.image_shape[1],len(channels)) if self.data.image_shape is not None else None

    def ExportChannels(self):
        """Export a txt file with channel names for downstream analysis."""

        # Print a sheet for m/z and channel numbers
        sheet = pd.DataFrame(self.data.pixel_table.columns, columns=["m/z peaks"])
        # Write out the sheet to csv
        sheet.to_csv(path_to_imzML.stem + "_channels.csv", sep="\t")

    def CreateSingleChannelArray(self, idx):
        """
        Function for extracting a single channel image from the array given an index
        """
        # create temporary image of all 0s to fill
        im = np.zeros((self.data.array_size[0], self.data.array_size[1]), dtype=np.float32)
        # Run through the data coordinates and fill array
        for i, (x, y, z) in enumerate(self.data.coordinates):
            # Add data to this slice -- only extact this index for each pixel
            # getspectrum returns mzs, intensities for pixels --> take only the intensity
            im[y - 1, x - 1] = self.data.getspectrum(i)[1][idx]
        # return the filled array
        return im

    def ConstructImage(self,channels=None):
        """
        Function for extracting an image from m/z spectrum and intensities
        """
        # create temporary image of all 0s to fill
        im = np.zeros(self.image_shape, dtype=np.float32)
        # check for channels to use
        if channels is None:
            channels = self.data.pixel_table.columns
        # iterate through channels
        for c, idx in enumerate(channels):
            # Run through the data coordinates and fill array
            for i, (x, y, z) in enumerate(self.data.coordinates):
                # Add data to this slice -- only extact this index for each pixel
                # getspectrum returns mzs, intensities for pixels --> take only the intensity
                im[y - 1, x - 1, c] = self.data.getspectrum(i)[1][idx]
        # add to object
        self.data.image = im
        # return the filled array
        return im

    def CreateCoordinateMask(self):
        """
        Create mask indicating coordinates of MSI data acquisition.
        """
        # create temporary image of all 0s to fill
        im = np.zeros((self.data.array_size[0], self.data.array_size[1]), dtype=np.bool_)
        # Run through the data coordinates and fill array
        for i, (x, y, z) in enumerate(self.data.coordinates):
            # Add data to this slice -- only extact this index for each pixel
            # getspectrum returns mzs, intensities for pixels --> take only the intensity
            im[y - 1, x - 1] = True
        # update the mask
        self.data.mask = im
		# return results
        return im

    def ExportCoordinateMask(self,outdir):
        """
        Create mask indicating coordinates of MSI data acquisition and export
        as tiff file.
        """
        # create name
        nm = Path(outdir).joinpath(self.data.filename.name + "_mask.tiff")
        # create temporary image of all 0s to fill
        im = self.CreateCoordinateMask()
        # single channel exporter
        skimage.io.imsave(nm, im, plugin="tifffile")
		# return results
        return im

    def ExportNIFTI(self,outdir,channels=None):
        """
        Export imzml file as nifti while preserving metadata.
        """
        # create name
        nm = Path(outdir).joinpath(self.data.filename.name + ".nii")
        # create image
        im = self.ConstructImage(channels)
        # Create temporary array for nifti writing
        arr = nib.Nifti1Image(arr.transpose(1,0,2), affine=np.eye(4), extra=self.data.imzmldict)
		#Write the results to a nifti file
        nib.save(arr,str(nm))

    def ExportHDF5(self,outdir,channels=None):
        """
        Export imzml file as hdf5 while preserving metadata.
        """
        # create name
        nm = Path(outdir).joinpath(self.data.filename.name + ".hdf5")
        # check for channels to use
        if channels is None:
            channels = self.data.pixel_table.columns
        # iterate through channels
        for c, idx in enumerate(channels):
            # create image
            im = self.CreateSingleChannelArray(idx=idx)
            #Add a color axis when reshaping instead
            im = im.reshape((1,im.shape[0],im.shape[1],1))
            #Create an hdf5 dataset if idx is 0 plane
            if c == 0:
                #Create hdf5
                h5 = h5py.File(nm, "w")
                h5.create_dataset(str(im_stem), data=im,chunks=True,maxshape=(1,None,None,None))
                # copy over the imzml information to metadata in h5 file
                h5['imzml metadata'] = self.data.imzmldict
                h5.close()
            else:
                #Append hdf5 dataset
                h5 = h5py.File(nm, "a")
                #Add step size to the z axis
                h5[str(im_stem)].resize((c+1), axis = 3)
                #Add the image to the new channels
                h5[str(im_stem)][:,:,:,c:c+1] = im
                h5.close()
