# General utily functions for HDI data preparation
# Developer: Joshua M. Hess, BSc
# Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

# Import external modules
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from skimage.transform import resize
from skimage.color import rgb2gray
from ast import literal_eval
import logging

# Import miaaim module
from miaaim.io.imwrite import _export

# Define function
def CreateHyperspectralImage(embedding, array_size, coordinates, scale=True):
    """Fill a hyperspectral image from n-dimensional embedding of high-dimensional
    imaging data by rescaling each channel from 0-1 (optional). All coordinates
    in the image not listed in coordinates object will be masked
    and set to 0 (background).

    Parameters
    ----------
    embedding: Pandas DataFrame
        Indicates embedding coordinates from UMAP or another method.

    array_size: tuple
        Indicates size of image.

    coordinates: 1-indexed list of tuples
        Indicates pixel coordinates of image.

    scale: Bool (Default: True)
        Rescale pixel intensities on the range of 0-1.

    Returns
    -------
    im: array
        Reconstructed image.
    """

    # Create zeros array to fill with number channels equal to embedding dimension
    im = np.zeros((array_size[0], array_size[1], embedding.shape[1]), dtype=np.float32)

    # Run through the data coordinates and fill array
    for i, (x, y, z) in enumerate(coordinates):
        # Run through each slice of embedding (dimension)
        for dim in range(embedding.shape[1]):
            # Add data to this slice
            im[y - 1, x - 1, dim] = embedding.values[i, dim]

    # Create a mask to use for excluding pixels not used in dimension reduction
    im_bool = np.zeros((array_size[0], array_size[1]), dtype=np.bool)

    # Fill the mask array with True values at the coordinates used
    for i, (x, y, z) in enumerate(coordinates):
        # Add boolean mask
        im_bool[y - 1, x - 1] = True

    # Check to see if scaling the pixel values 0 to 1
    if scale:
        # Scale the data 0-1 for hyperspectral image construction
        for dim in range(im.shape[2]):
            # min-max scaler
            im[:, :, dim] = (im[:, :, dim] - im[:, :, dim].min()) / (
                im[:, :, dim].max() - im[:, :, dim].min()
            )

    # Mask the image with the boolean array to remove unused pixels
    im[~im_bool] = 0

    # Return the hyperspectral image
    return im

# Define function
def CreateHyperspectralImageRectangular(embedding, array_size, coordinates, scale=True):
    """Fill a hyperspectral image from n-dimensional embedding of high-dimensional
    imaging data by rescaling each channel from 0-1 (optional). All coordinates
    in the image not listed in coordinates object will be masked
    and set to 0 (background). This function assumes that the data you want
    to reconstruct can be automatically reshaped into a rectangular array.

    Parameters
    ----------
    embedding: Pandas DataFrame
        Indicates embedding coordinates from UMAP or another method.

    array_size: tuple
        Indicates size of image.

    coordinates: 1-indexed list of tuples
        Indicates pixel coordinates of image.

    scale: Bool (Default: True)
        Rescale pixel intensities on the range of 0-1.

    Returns
    -------
    im: array
        Reconstructed image.
    """

    # get the embedding shape
    number_channels = embedding.shape[1]
    # Create zeros array to fill with number channels equal to embedding dimension
    im = embedding.values.reshape((array_size[0], array_size[1], number_channels))

    # Check to see if scaling the pixel values 0 to 1
    if scale:
        # Scale the data 0-1 for hyperspectral image construction
        for dim in range(im.shape[2]):
            # min-max scaler
            im[:, :, dim] = (im[:, :, dim] - im[:, :, dim].min()) / (
                im[:, :, dim].max() - im[:, :, dim].min()
            )

    # Return the hyperspectral image
    return im

def ExportNifti(image, filename, padding=None, target_size=None,grayscale=False):
    """Export processed images resulting from UMAP and
    spatially mapping UMAP, or exporting processed histology images.

    Parameters
    ----------
    image: array
        Array that represents image to export.

    filename: string
        Path, including filename, to export processed nifti image to.

    padding: string of tuple of type integer (padx,pady; Default: None)
        Indicates height and length padding to add to the image before exporting.

    target_size: string of tuple of type integer (sizex,sizey; Default: None)
        Resize image using bilinear interpolation before exporting.
    """

    # Create pathlib object from the filename
    filename = Path(filename)

    # convert the padding and target size to tuple if present
    if padding is not None:
        padding = literal_eval(padding)
    if target_size is not None:
        target_size = literal_eval(target_size)

    # Print update
    logging.info(f'Exporting f{filename}...')
    # Check to see if padding
    if padding is not None:
        image = np.pad(
            image,
            [(padding[0], padding[0]), (padding[1], padding[1]), (0, 0)],
            mode="constant",
        )
    # Check to see if resizing
    if target_size is not None:
        image = resize(image,target_size)
    # check for grayscale conversion
    if grayscale:
        # convert
        image = rgb2gray(image)
    # Create nifti object -- transpose axes because of the transformation!
    # Check size
    if len(image.shape) > 2:
        # Create nifti object -- transpose axes because of the transformation!
        nifti_im = nib.Nifti1Image(image.transpose(1, 0, 2), affine=np.eye(4))
    else:
        # Create nifti object -- transpose axes because of the transformation!
        nifti_im = nib.Nifti1Image(image.T, affine=np.eye(4))
    # Save the image
    nib.save(nifti_im, str(filename))
    # Print update
    logging.info(f'Finished exporting {filename}')


# Add function for exporting UMAP nifti image
def Export1(image, filename, padding=None, target_size=None,grayscale=False):
    """Export processed images resulting from UMAP and
    spatially mapping UMAP, or exporting processed histology images.

    Parameters
    ----------
    output_dir: string
        Path to output directory to store processed nifti image.

    padding: string of tuple of type integer (padx,pady; Default: None)
        Indicates height and length padding to add to the image before exporting.

    target_size: string of tuple of type integer (sizex,sizey; Default: None)
        Resize image using bilinear interpolation before exporting.
    """

    # Create pathlib object from the filename
    filename = Path(filename)

    # convert the padding and target size to tuple if present
    if padding is not None:
        # check for string or integer
        if isinstance(padding,str):
            padding = literal_eval(padding)
    if target_size is not None:
        # check for string or integer
        if isinstance(target_size,str):
            target_size = literal_eval(target_size)

    # Print update
    logging.info(f'Exporting f{filename}...')
    # Check to see if padding
    if padding is not None:
        # check dimensions of the image
        if image.ndim > 2:
            image = np.pad(
                image,
                [(padding[0], padding[0]), (padding[1], padding[1]), (0, 0)],
                mode="constant",
            )
        elif image.ndim == 2:
            image = np.pad(
                image,
                [(padding[0], padding[0]), (padding[1], padding[1])],
                mode="constant",
            )
        else:
            raise ValueError(f'Image with {image.ndim} not supported.')
    # Check to see if resizing
    if target_size is not None:
        image = resize(image,target_size)
    # check for grayscale conversion
    if grayscale:
        # convert
        image = rgb2gray(image)
    # export using io
    _export.HDIexporter(arr=image, out=filename)
    # Print update
    logging.info(f'Finished exporting {filename}')


def Exp(x, a, b, c):
    """Exponential function to use for regression.
    """
    return a * np.exp(-b * x) + c
