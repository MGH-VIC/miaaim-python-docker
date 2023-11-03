# Regionprops based single-cell quantification
# Developer: Joshua M. Hess, BSc
# Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

# import modules
import skimage.io
import h5py
import pandas as pd
import numpy as np
import os
import skimage.measure as measure
from pathlib import Path
import nibabel as nib
import yaml
import logging
import sys

# import custom modules
import miaaim
from miaaim.cli.quant import _parse



# =============================================================================
# root = Path("/Users/joshuahess/Desktop/ROI010_PROSTATE_TMA001")
# x = list(range(1,62))
# y = list(range(1,702))
# 
# x = pd.DataFrame(x)
# y = pd.DataFrame(y)
# 
# x.to_csv("/Users/joshuahess/Desktop/imc.csv",index=None)
# y.to_csv("/Users/joshuahess/Desktop/msi.csv",index=None)
# 
# test = HDIquantification(root_folder=root,
#             masks=[ root.joinpath("segmentation/imc/hdiseg-ilastik/ROI010_PROSTATE_TMA001_core_probabilities_mask.tiff") ],
#             images=[ root.joinpath("preprocessing/imc/ROI010_PROSTATE_TMA001_core.ome.tiff"),
#                     root.joinpath("registration/msi-imc/transformix/ROI010_PROSTATE_TMA001_result.nii")
#                     ],       
#             channel_names=[Path("/Users/joshuahess/Desktop/imc-markers.csv"),
#                            Path("/Users/joshuahess/Desktop/msi-markers.csv")],
#             names=["imc", "msi"],
#             probabilities_method="ilastik",
#             segmentation_method="hdiseg",
#             mask_props=None,
#             intensity_props=["intensity_mean"])
#     
# 
# test.Quantify(output=None)
# test.QC()
# 
# =============================================================================



def check_image_dir(dir,endings=["mask.","UMAP."]):
    """
    Helper function to remove extra files from directory when
    parsing images.

    Parameters
    ----------
    dir : TYPE
        DESCRIPTION.
    endings : TYPE, optional
        DESCRIPTION. The default is ["mask","UMAP"].

    Returns
    -------
    None.

    """
    # get list of all contents in a directory
    all_list = SearchDir(ending="",dir=dir)
    # create list to return
    f_list=[]
    # remove anything starting with . and all endings
    for f in all_list:
        # get name
        nm = f.name
        # check for starting .
        if nm[0] == ".":
            # do not include
            continue

        else:
            # create tagger
            ADD = True
            # now check endings
            endng = str(nm.split("_")[-1])
            # iterate
            for e in endings:
                if e in endng:
                    ADD = False
            # now check if adding
            if ADD:
                f_list.append(f)

    # return the list
    return f_list


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

def gini_index(mask, intensity):
    x = intensity[mask]
    sorted_x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(sorted_x, dtype=float)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

def median_intensity(mask, intensity):
    return np.median(intensity[mask])

def MaskChannel(mask_loaded, image_loaded_z, intensity_props=["intensity_mean"]):
    """Function for quantifying a single channel image

    Returns a table with CellID according to the mask and the mean pixel intensity
    for the given channel for each cell"""
    # Look for regionprops in skimage
    builtin_props = set(intensity_props).intersection(measure._regionprops.PROP_VALS)
    # Otherwise look for them in this module
    extra_props = set(intensity_props).difference(measure._regionprops.PROP_VALS)
    dat = measure.regionprops_table(
        mask_loaded, image_loaded_z,
        properties = tuple(builtin_props),
        extra_properties = [globals()[n] for n in extra_props]
    )
    return dat


def MaskIDs(mask, mask_props=None):
    """This function will extract the CellIDs and the XY positions for each
    cell based on that cells centroid

    Returns a dictionary object"""

    all_mask_props = set(["label", "centroid", "area", "major_axis_length", "minor_axis_length", "eccentricity", "solidity", "extent", "orientation"])
    if mask_props is not None:
        all_mask_props = all_mask_props.union(mask_props)

    dat = measure.regionprops_table(
        mask,
        properties=all_mask_props
    )

    name_map = {
        "CellID": "label",
        "X_centroid": "centroid-1",
        "Y_centroid": "centroid-0",
        "Area": "area",
        "MajorAxisLength": "major_axis_length",
        "MinorAxisLength": "minor_axis_length",
        "Eccentricity": "eccentricity",
        "Solidity": "solidity",
        "Extent": "extent",
        "Orientation": "orientation",
    }
    for new_name, old_name in name_map.items():
        dat[new_name] = dat[old_name]
    for old_name in set(name_map.values()):
        del dat[old_name]

    return dat


def PrepareData(image,z):
    """Function for preparing input for maskzstack function. Connecting function
    to use with mc micro ilastik pipeline"""

    image_path = Path(image)

    #Check to see if image tif(f)
    if image_path.suffix == '.tiff' or image_path.suffix == '.tif':
        #Check to see if the image is ome.tif(f)
        if  str(image_path).endswith(('.ome.tif','.ome.tiff')):
            #Read the image
            image_loaded_z = skimage.io.imread(image_path,img_num=z,plugin='tifffile')
            #print('OME TIF(F) found')
        else:
            #Read the image
            image_loaded_z = skimage.io.imread(image_path,img_num=z,plugin='tifffile')
            #print('TIF(F) found')
            # Remove extra axis
            #image_loaded = image_loaded.reshape((image_loaded.shape[1],image_loaded.shape[3],image_loaded.shape[4]))

    #Check to see if image is hdf5
    elif image_path.suffix == '.h5' or image_path.suffix == '.hdf5':
        #Read the image
        f = h5py.File(image_path,'r+')
        #Get the dataset name from the h5 file
        dat_name = list(f.keys())[0]
        ###If the hdf5 is exported from ilastik fiji plugin, the dat_name will be 'data'
        #Get the image data
        image_loaded = np.array(f[dat_name])
        #Remove the first axis (ilastik convention)
        image_loaded = image_loaded.reshape((image_loaded.shape[1],image_loaded.shape[2],image_loaded.shape[3]))
        ###If the hdf5 is exported from ilastik fiji plugin, the order will need to be
        ###switched as above --> z_stack = np.swapaxes(z_stack,0,2) --> z_stack = np.swapaxes(z_stack,0,1)

    #Raise error if not supported image type
    else:
        raise(ValueError("Unsupported image type!"))

    #Return the objects
    return image_loaded_z


def MaskZstack(masks_loaded,image,channel_names_loaded, mask_props=None, intensity_props=["intensity_mean"]):
    """This function will extract the stats for each cell mask through each channel
    in the input image

    mask_loaded: dictionary containing Tiff masks that represents the cells in your image.

    z_stack: Multichannel z stack image"""

    #Create pathlib object for the input image
    image_path = Path(image)

    #Get the names of the keys for the masks dictionary
    mask_names = list(masks_loaded.keys())
    #Get the CellIDs for this dataset by using only a single mask (first mask)
    IDs = pd.DataFrame(MaskIDs(masks_loaded[mask_names[0]]))
    #Create empty dictionary to store channel results per mask
    dict_of_chan = {m_name: [] for m_name in mask_names}

    #Check to see if the image is nifti image
    if image_path.suffix == '.nii':
        #Can load the full image because it is a memmap
        nii_im = nib.load(str(image_path)).get_fdata().transpose(1,0,2)
        #Get the z channel and the associated channel name from list of channel names
        for z in range(len(channel_names_loaded)):
            #Run the data Prep function
            image_loaded_z = nii_im[:,:,z]

            #Iterate through number of masks to extract single cell data
            for nm in range(len(mask_names)):
                #Use the above information to mask z stack
                dict_of_chan[mask_names[nm]].append(
                    MaskChannel(masks_loaded[mask_names[nm]],image_loaded_z, intensity_props=intensity_props)
                )
            #Print progress
            print("Finished "+str(z))

    #Otherwise use the prepare data funtion
    else:

        #Get the z channel and the associated channel name from list of channel names
        for z in range(len(channel_names_loaded)):
            #Run the data Prep function
            image_loaded_z = PrepareData(image,z)

            #Iterate through number of masks to extract single cell data
            for nm in range(len(mask_names)):
                #Use the above information to mask z stack
                dict_of_chan[mask_names[nm]].append(
                    MaskChannel(masks_loaded[mask_names[nm]],image_loaded_z, intensity_props=intensity_props)
                )
            #Print progress
            print("Finished "+str(z))

    # Column order according to histoCAT convention (Move xy position to end with spatial information)
    last_cols = (
        "X_centroid",
        "Y_centroid",
        "column_centroid",
        "row_centroid",
        "Area",
        "MajorAxisLength",
        "MinorAxisLength",
        "Eccentricity",
        "Solidity",
        "Extent",
        "Orientation",
    )
    def col_sort(x):
        if x == "CellID":
            return -2
        try:
            return last_cols.index(x)
        except ValueError:
            return -1

    #Iterate through the masks and format quantifications for each mask and property
    for nm in mask_names:
        mask_dict = {}
        # Mean intensity is default property, stored without suffix
        mask_dict.update(
            zip(channel_names_loaded, [x["intensity_mean"] for x in dict_of_chan[nm]])
        )
        # All other properties are suffixed with their names
        for prop_n in set(dict_of_chan[nm][0].keys()).difference(["intensity_mean"]):
            mask_dict.update(
                zip([f"{n}_{prop_n}" for n in channel_names_loaded], [x[prop_n] for x in dict_of_chan[nm]])
            )
        # Get the cell IDs and mask properties
        mask_properties = pd.DataFrame(MaskIDs(masks_loaded[nm], mask_props=mask_props))
        mask_dict.update(mask_properties)
        dict_of_chan[nm] = pd.DataFrame(mask_dict).reindex(columns=sorted(mask_dict.keys(), key=col_sort))

    # Return the dict of dataframes for each mask
    return dict_of_chan



def ExtractSingleCells(masks,image,channel_names,output, mask_props=None, intensity_props=["intensity_mean"]):
    """Function for extracting single cell information from input
    path containing single-cell masks, z_stack path, and channel_names path."""

    #Create pathlib object for output
    output = Path(output)

    # check for channel names type
    if isinstance(channel_names,list):
        channel_names_loaded_checked = channel_names.copy()
        
    # otherwise try to read csv file
    else:
        #Read csv channel names
        channel_names_loaded = pd.read_csv(channel_names)
        #Check for size of columns
        if channel_names_loaded.shape[1] > 1:
            #Get the marker_name column if more than one column (CyCIF structure)
            channel_names_loaded_list = list(channel_names_loaded.marker_name)
        else:
            #old one column version -- re-read the csv file and add column name
            channel_names_loaded = pd.read_csv(channel_names, header = None)
            #Add a column index for ease and for standardization
            channel_names_loaded.columns = ["marker"]
            channel_names_loaded_list = list(channel_names_loaded.marker)
    
        #Check for unique marker names -- create new list to store new names
        channel_names_loaded_checked = []
        for idx,val in enumerate(channel_names_loaded_list):
            #Check for unique value
            if channel_names_loaded_list.count(val) > 1:
                #If unique count greater than one, add suffix
                channel_names_loaded_checked.append(str(val) + "_"+ str(channel_names_loaded_list[:idx].count(val) + 1))
            else:
                #Otherwise, leave channel name
                channel_names_loaded_checked.append(str(val))

    #Clear small memory amount by clearing old channel names
    channel_names_loaded, channel_names_loaded_list = None, None

    #Read the masks
    masks_loaded = {}
    #iterate through mask paths and read images to add to dictionary object
    for m in masks:
        m_full_name = os.path.basename(m)
        m_name = m_full_name.split('.')[0]
        masks_loaded.update({str(m_name):skimage.io.imread(m,plugin='tifffile')})

    scdata_z = MaskZstack(masks_loaded,image,channel_names_loaded_checked, mask_props=mask_props, intensity_props=intensity_props)
    #Write the singe cell data to a csv file using the image name

    im_full_name = os.path.basename(image)
    im_name = im_full_name.split('.')[0]

    # iterate through each mask and export csv with mask name as suffix
    for k,v in scdata_z.items():
        # export the csv for this mask name
        scdata_z[k].to_csv(
                            str(Path(os.path.join(str(output),
                            str(im_name+"_{}"+".csv").format(k)))),
                            index=False
                            )


def MultiExtractSingleCells(masks,images,channel_names,output,mask_props=None, intensity_props=["intensity_mean"]):
    """Function for iterating over a list of z_stacks and output locations to
    export single-cell data from image masks"""


    # create indexer for output directories
    o = 0
    # iterate through z stacks and channel csvs to export data
    for i in range(len(images)):

        print("Extracting single-cell data for "+str(images[i])+'...')

        #Run the ExtractSingleCells function for this image
        ExtractSingleCells(masks,images[i],channel_names[i],output[o], mask_props=mask_props, intensity_props=intensity_props)

        #Print update
        im_full_name = os.path.basename(images[i])
        im_name = im_full_name.split('.')[0]
        print("Finished "+str(im_name))
        # update the indexer for output
        o = o + 1


class HDIquantification:

    def __init__(
            self,
            root_folder,
            masks,
            images,       
            channel_names=None,
            names=["imc"],
            probabilities_method="ilastik",
            segmentation_method="hdiseg",
            mask_props=None,
            intensity_props=["intensity_mean"],
            ):
        
        # create logger format
        FORMAT = '%(asctime)s | [%(pathname)s:%(lineno)s - %(funcName)s() ] | %(message)s'

        # check for root folder name
        root_folder = Path(root_folder)
        self.root_folder=root_folder

        # get paths
        self.masks = [Path(m) for m in masks]
        self.images = [ Path(im) for im in images ]            
        self.channel_names = [ Path(c) for c in channel_names ] if channel_names is not None else [None]*len(images)
        self.names = names
        self.probabilities_method = probabilities_method
        self.segmentation_method = segmentation_method
        self.mask_props = mask_props
        self.intensity_props = intensity_props
        
        # create direectory specific for process
        quant_dir = root_folder.joinpath("quantification")
        # check if exists already
        if not quant_dir.exists():
            # make it
            quant_dir.mkdir()
        # create direectory specific for process
        quant_dir = quant_dir.joinpath(f"{self.segmentation_method}-{self.probabilities_method}")
        # check if exists already
        if not quant_dir.exists():
            # make it
            quant_dir.mkdir()
        self.quant_dir = quant_dir

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
            f"miaaim-quant-{self.segmentation_method}-{self.probabilities_method}"+".yaml"
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
        qc_quant_dir = qc_dir.joinpath("quantification")
        # check if exists already
        if not qc_quant_dir.exists():
            # make it
            qc_quant_dir.mkdir()
        self.qc_quant_dir = qc_quant_dir

        qc_quant_name_dir = qc_quant_dir.joinpath(
            f"{self.segmentation_method}-{self.probabilities_method}"
            )
        # check if exists already
        if not qc_quant_name_dir.exists():
            # make it
            qc_quant_name_dir.mkdir()
        self.qc_quant_name_dir = qc_quant_name_dir

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
            f"miaaim-quant"+f'-{self.segmentation_method}-{self.probabilities_method}'+".log"
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
                                format=FORMAT,
                                force=True)

        # get logger
        logger = logging.getLogger()
        # writing to stdout
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        # handler.setFormatter(FORMAT)
        logger.addHandler(handler)

        # create name of shell command
        sh_name = os.path.join(Path(prov_dir),"miaaim-quant"+f'-{self.segmentation_method}-{self.probabilities_method}'+".sh")
        
        # command to capture print functions to log
        self.log_name = log_name
        self.logger = logger
        self.sh_name = sh_name
        self.yaml_name = yaml_name

        
        # print first log
        logging.info("MIAAIM QUANTIFICATION")
        logging.info(f'MIAAIM VERSION {miaaim.__version__}')
        logging.info(f'METHOD: HDIquantification')
        logging.info(f'ROOT FOLDER: {self.root_folder}')
        logging.info(f'PROVENANCE FOLDER: {self.prov_dir}')
        logging.info(f'QC FOLDER: {self.qc_quant_name_dir} \n')

        # Get file extensions for tiff probability images
        # tiff_ext = [".tif", ".tiff"]
        # get file extensions for data going to quantification
        # image_ext = [".hdf5",".h5",".nii"]
        
        # update yaml file
        self.yaml_log.update({'MODULE':"Quantification"})
        self.yaml_log.update({'METHOD':"hdiquant"})
        self.yaml_log.update({'ImportOptions':{'root_folder':str(self.root_folder),
                                                'masks':[ str(m) for m in self.masks ],
                                                'images':[ str(i) for i in self.images ],
                                                'channel_names':[ str(c) for c in self.channel_names ] if self.channel_names is not None else None,
                                                'names':names,
                                                'probabilities_method':self.probabilities_method,
                                                'segmentation_method':self.segmentation_method,
                                                'mask_props':self.mask_props,
                                                'intensity_props':self.intensity_props}})

        # update logger
        logging.info(f'\n')
        logging.info("PROCESSING DATA")        
        
    def Quantify(self, output=None):
        """
        Quantify single-cell measurements using regionprops.

        Returns
        -------
        None.

        """
        
        # update logger
        logging.info(f'Quantify: extracting single-cell measurements')
        # update logger
        self.yaml_log.update({'ProcessingSteps':[]})
        self.yaml_log['ProcessingSteps'].append("Quantify")
        
        # check for output directory
        if output is None:
            # set output based on input names and quant directory
            output = [ self.quant_dir.joinpath(n) for n in self.names ]
            # check if exists
            for o in output:
                if not o.exists():
                    # warn
                    logging.warning("Creating directories for quantification export")
                    # create directory
                    o.mkdir()
        else:
            # otherwise set as input parameter
            output = [output]*len(self.images)
        
        #Run the ExtractSingleCells function for this image
        MultiExtractSingleCells(
            self.masks,
            self.images,
            self.channel_names,
            output, 
            mask_props=self.mask_props, 
            intensity_props=self.intensity_props
            )
        
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
        proc_fname = os.path.join(Path(_parse.__file__).parent,"_cli_quantification.py")
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

        # provenance
        self._exportYAML()
        self._exportSH()
        # close the logger
        self.logger.handlers.clear()




#
