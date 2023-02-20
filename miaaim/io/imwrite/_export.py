# Module for exporting image data in a variety of formats
# Developer: Joshua M. Hess, BSc
# Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

# Import external modules
from pathlib import Path
import skimage.io
import numpy as np
import h5py
import nibabel as nib

#arr = r"D:\Josh_Hess\01_10_19_MSI_peak_pick_20191025\Methods_Analysis_20190110\CyCIF\lung\LUNG-1-LN_40X.ome_CyCIF_processed.nii"
#arr = nib.load(arr).get_fdata().transpose(1,0,2)
#out = r"D:\Josh_Hess\01_10_19_MSI_peak_pick_20191025\Test\HDIexport.hdf5"
#HDIexporter(arr,out)

# Define class object
class HDIexporter:
    """Class to export data into multiple imaging data formats"""

    def __init__(self, arr, out):
        """Initialize the class by using the path to the image and array
        that is in the xyc data format. transpositions will convert the input
        to the appropriate shape for data exporting

        arr: array that will be exported (numpy)
        out: Path to image (Ex: path/to/image.extension)
        """

        # Initialize the objects in this class
        self.path = Path(out)
        self.ext = "".join(self.path.suffixes)
        self.stem = self.path.stem
        self.image_shape = arr.shape

        # Set the file extensions that we can use with this class
        all_ext = [
            ".ome.tif",
            ".ome.tiff",
            ".tif",
            ".tiff",
            ".h5",
            ".hdf5",
            #".imzML",
            ".nii",
        ]
        # Get file extensions for cytometry files
        h5_ext = [".h5", ".hdf5"]
        tif_ext = [".ome.tif", ".ome.tiff", ".tif", ".tiff"]
        #imzML_ext = [".imzML"]
        nii_ext = [".nii"]
        # ensure the image contains and ending that is currently supported
        if not self.ext in all_ext:
            # raise error
            raise(ValueError(str(self.ext) + " files are not currently supported!"))

        # tiff writer -- currently only works with 2D multiplex arrays
        if self.ext in tif_ext:
            # check the data structure (this is used to make sure that exported data matched ImageJ visualization)
            if len(self.image_shape)>2:
                # adjust hdiimporter array structure to export properly
                skimage.io.imsave(self.path, arr.transpose(2,0,1), plugin="tifffile")
            # otherwise this is a single-channel image and only transpose the xy
            else:
                # single channel exporter
                skimage.io.imsave(self.path, arr, plugin="tifffile")

        # nifti writer
        elif self.ext in nii_ext:
            # Check size
            if len(self.image_shape) > 2:
                # Create nifti object -- transpose axes because of the transformation!
                arr = nib.Nifti1Image(arr.transpose(1, 0, 2), affine=np.eye(4))
            else:
                # Create nifti object -- transpose axes because of the transformation!
                arr = nib.Nifti1Image(arr.T, affine=np.eye(4))
			#Write the results to a nifti file
            nib.save(arr,str(self.path))

        # h5 writer -- will export single channels to hd
        elif self.ext in h5_ext:
            #Check if step size is 1 or two (again, if 1, then no color channel)
            if len(self.image_shape)>2:
                #Reshape the array
                arr = arr.reshape((1,self.image_shape[0],self.image_shape[1],self.image_shape[2]))
            else:
                #Add a color axis when reshaping instead
                arr = arr.reshape((1,self.image_shape[0],self.image_shape[1],1))
            #Create hdf5
            h5 = h5py.File(self.path, "w")
            h5.create_dataset(str(self.stem), data=arr[:,:,:,:],chunks=True,maxshape=(1,None,None,None))
            h5.close()

        # otherwise raise an error for now
        else:
            raise(ValueError("Unable to export image properly"))


def getMetadata(path,commit):
    """From s3 segmenter tiff file metadata.
    """
    with tifffile.TiffFile(path) as tif:
        if not tif.ome_metadata:
            try:
                x_res_tag = tif.pages[0].tags['XResolution'].value
                y_res_tag = tif.pages[0].tags['YResolution'].value
                physical_size_x = x_res_tag[0] / x_res_tag[1]
                physical_size_y = y_res_tag[0] / y_res_tag[1]
            except KeyError:
                physical_size_x = 1
                physical_size_y = 1
            metadata_args = dict(
            pixel_sizes=(physical_size_y, physical_size_x),
            pixel_size_units=('µm', 'µm'),
            software= 's3segmenter v' + commit
            )
        else:
            metadata=ome_types.from_xml(tif.ome_metadata)
            metadata = metadata.images[0].pixels
            metadata_args = dict(
            pixel_sizes=(metadata.physical_size_y,metadata.physical_size_x),
            pixel_size_units=('µm', 'µm'),
            software= 's3segmenter v' + commit
            )
        return metadata_args

    

#
#
# tif.pages.keyframe.get_resolution()
# tif.pages.keyframe.resolutionunit.value


#
