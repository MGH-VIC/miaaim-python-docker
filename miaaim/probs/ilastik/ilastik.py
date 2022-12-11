#Exporting images to hdf5 format for ilastik random forest pixel classification

# import modules
import tifffile
import numpy as np
import h5py
import os
import random
from skimage import filters
import scipy.io
import math
import yaml
from pathlib import Path
import logging
import platform
import time
import pathlib
import sys
from ast import literal_eval

# import custom modules
import miaaim
from miaaim.io.imread import _import
# for logging purposes
from miaaim.cli.probs.ilastik import _parse
#
# test = Path("/Users/joshuahess/Desktop/miaaim-python-dev/miaaim/proc")


def SearchDir(dir, string = "ilastik-"):
    """Search only in given directory for files that end with
    the specified suffix.

    Parameters
    ----------
    string: string (Default: "ilastik-")
        String to search for in the given directory

    dir: string (Default: None, will search in current working directory)
        Directory to search for files in.

    Returns
    -------
    full_list: list
        List of pathlib objects for each file found with the given string.
    """

    #Search the directory only for files
    full_list = []
    for f in sorted(os.listdir(dir)):
        if string in f:
            full_list.append(Path(os.path.join(dir,f)))
    #Return the list
    return full_list



def _ilastik_headless_search():
    """Light function to check system platform and search common
    installation locations for ilastik.

    Returns
    -------
    exe: pathlib.Path with path to ilastik application

    comm: pathlib.path with path to executable
        (run_ilastik.sh for linux/OS, ilastik.exe for Windows)
    """
    # check for linux
    if platform.system() == "Windows":
        # search program files in C:\ drive first
        p = Path("C:\Program Files")
        if len(p)==0:
            p = Path("\Program Files")
        # get executable location
        exe = SearchDir(dir=p)[-1]
        # shell script inside version folder
        comm = exe.joinpath("ilastik.exe")
    # check for linux
    elif platform.system() == "Linux":
        # assume docker image and search opt/ilastik
        p = Path("/opt")
        # search
        exe = SearchDir(dir=p,string = "ilastik")[-1]
        # shell script inside version folder
        comm = exe.joinpath("run_ilastik.sh")
    # check for OS (needs work)
    elif platform.system() == "Darwin" or os.name == 'posix':
        # search applications
        p = Path("/Applications")
        # get executable (will be .app)
        exe = SearchDir(dir=p)[-1]
        # get shell script location
        comm = exe.joinpath("Contents/ilastik-release/run_ilastik.sh")
    # search and return latest ilastik
    return exe, comm



# =============================================================================
# IL = Ilastik(root_folder="/Users/joshuahess/Desktop/test_new/ROI024_PROSTATE_TMA019",
# input="input/imc",
# name=None,
# executable=None,
# qc=True,
# resume=False)
#
# IL.log_name
# IL.PrepareTraining(input = "/Users/joshuahess/Desktop/test_new/ROI023_LIVER_D12/input/imc/ROI023_LIVER D12.ome.tiff",channelIDs=list(range(62)))
# IL.PredictHeadless(ilp="/Users/joshuahess/MyProject.ilp")
# IL.QC()
# =============================================================================


class Ilastik():
    """Ilastik pixel classification and cell segmentation.
    """
    def __init__(self,
                root_folder,
                input_image,
                name=None,
                executable=None,
                command=None,
                qc=True,
                resume=True):

        self.root_folder = Path(root_folder)
        self.input_image = self.root_folder.joinpath(input_image)
        self.in_directory = None
        self.executable = executable
        self.command = None
        self.executed_command = None
        self.qc = qc
        self.resume = resume
        # other parameters
        self.ilp = None
        self.crops = None
        self.crops_coords = None
        self.train_dir = None
        self.full_hdf5 = None
        # qc and provenance info
        self.name = name
        self.qc_dir = None
        self.qc_prob_dir = None
        self.qc_prob_name_dir = None
        self.docs_dir = None
        self.pred_dir = None
        self.log_name = None
        self.logger = None
        self.sh_name = None
        self.yaml_name = None

        # check for executable
        if executable is None:
            try:
                # check for it
                executable, command = _ilastik_headless_search()
            except:
                raise ValueError('Cannot find ilastik application. Please specify its location manually.')

        # update executable information
        self.executable = executable
        self.command = str(command) # convert pathlib to string for sending to os

        # create logger format
        FORMAT = '%(asctime)s | [%(pathname)s:%(lineno)s - %(funcName)s() ] | %(message)s'

        # check for root folder name
        if root_folder is not None:
            # make pathlib
            root_folder = Path(root_folder)
        else:
            # use current working directory
            root_folder = Path(os.getcwd())

        # create names
        self.in_directory = self.input_image.parent

        if self.name is None:
            self.name = self.in_directory.name

        # create direectory specific for process
        pred_dir = root_folder.joinpath("probabilities")
        # check if exists already
        if not pred_dir.exists():
            # make it
            pred_dir.mkdir()
        # create direectory specific for process
        pred_dir = pred_dir.joinpath(self.name)
        # check if exists already
        if not pred_dir.exists():
            # make it
            pred_dir.mkdir()
        # create direectory specific for process
        pred_dir = pred_dir.joinpath("ilastik")
        # check if exists already
        if not pred_dir.exists():
            # make it
            pred_dir.mkdir()

        # create direectory specific for process
        train_dir = root_folder.joinpath("probabilities")
        # check if exists already
        if not train_dir.exists():
            # make it
            train_dir.mkdir()
        # create direectory specific for process
        train_dir = train_dir.joinpath(self.name)
        # check if exists already
        if not train_dir.exists():
            # make it
            train_dir.mkdir()
        # create direectory specific for process
        train_dir = train_dir.joinpath("ilastik-training")
        # check if exists already
        if not train_dir.exists():
            # make it
            train_dir.mkdir()

        # create a docs directory
        docs_dir = Path(root_folder).joinpath("docs")
        # check if exists already
        if not docs_dir.exists():
            # make it
            docs_dir.mkdir()

        # create a parameters directory
        pars_dir = docs_dir.joinpath("parameters")
        # check if exists already
        if not pars_dir.exists():
            # make it
            pars_dir.mkdir()
        # create name of shell command
        yaml_name = os.path.join(Path(pars_dir),"miaaim-prob-ilastik"+f'-{self.name}'+".yaml")

        # create a qc directory
        qc_dir = docs_dir.joinpath("qc")
        # check if exists already
        if not qc_dir.exists():
            # make it
            qc_dir.mkdir()

        # create direectory specific for process
        qc_prob_dir = qc_dir.joinpath("probabilities")
        # check if exists already
        if not qc_prob_dir.exists():
            # make it
            qc_prob_dir.mkdir()
        # create direectory specific for process
        qc_prob_dir = qc_prob_dir.joinpath(f'{self.name}')
        # check if exists already
        if not qc_prob_dir.exists():
            # make it
            qc_prob_dir.mkdir()

        qc_prob_name_dir = qc_prob_dir.joinpath('ilastik')
        # check if exists already
        if not qc_prob_name_dir.exists():
            # make it
            qc_prob_name_dir.mkdir()

        # create a qc directory
        prov_dir = docs_dir.joinpath("provenance")
        # check if exists already
        if not prov_dir.exists():
            # make it
            prov_dir.mkdir()
        # create name of logger
        log_name = os.path.join(Path(prov_dir),"miaaim-prob-ilastik"+f"-{self.name}"+".log")

        # check for resume
        if self.resume:
            # Open the yaml file
            with open(yaml_name, "r") as stream:
                # Try to load the yaml
                yml = yaml.full_load(stream)
                logging.info(yml)
            # check for versioning compatibility
            # v = yml["VERSION"]
            # ex = yml["EXECUTABLE"]
            self.yaml_log = yml

        else:
            # start yaml log
            self.yaml_log = {}

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


        # command to capture print functions to log
        print = logger.info

        # create name of shell command
        sh_name = os.path.join(Path(prov_dir),"miaaim-prob-ilastik"+f'-{self.name}'+".sh")

        # update attributes
        self.root_folder = root_folder
        self.pred_dir = pred_dir
        self.train_dir = train_dir
        self.docs_dir = docs_dir
        self.qc_dir = qc_dir
        self.qc_prob_dir = qc_prob_dir
        self.qc_prob_name_dir = qc_prob_name_dir
        self.prov_dir = prov_dir
        self.log_name = log_name
        self.logger = logger
        self.sh_name = sh_name
        self.yaml_name = yaml_name

        # update logger with version number of miaaim
        self.yaml_log.update({"MIAAIM VERSION":miaaim.__version__})
        # update yaml file
        self.yaml_log.update({'MODULE':"Probabilities"})
        self.yaml_log.update({'METHOD':"Ilastik"})
        self.yaml_log.update({'ImportOptions':{'root_folder':str(root_folder),
                                                'input_image':str(self.input_image),
                                                'name':self.name,
                                                'executable':str(self.executable),
                                                'command':str(self.command),
                                                'qc':qc,
                                                'resume':resume}})



        # print first log
        logging.info("MIAAIM PROBABILITIES")
        logging.info(f'MIAAIM VERSION {miaaim.__version__}')
        logging.info(f'METHOD: Ilastik')
        logging.info(f'ROOT FOLDER: {self.root_folder}')
        logging.info(f'PROVENANCE FOLDER: {self.prov_dir}')
        logging.info(f'QC FOLDER: {self.qc_prob_name_dir} \n')


        # update logger
        logging.info(f'\n')
        logging.info("PROCESSING DATA")

        # check for resuming
        if self.resume:
            logging.info(f'Resuming Ilastik workflow...')
            # get the image name (remove ".ome" if it exists)
            im_stem = self.input_image.stem.replace(".ome","")
            # create hdf5 name
            h5_name = im_stem + ".hdf5"
            # set the hdf5 name
            self.full_hdf5 = self.train_dir.joinpath(h5_name)

    def _IlastikPrep(self,
                input_image,
                output,
                crop,
                crop_size,
                nonzero_fraction,
                nuclei_index,
                channelIDs,
                crop_amount):

        """
        Export ome.tiff and other image formats as an
        hdf5 image for training ilastik random forest pixel classifier along with
        cropped regions.

        Parameters
        ----------
        input: str
            Path to image to be exported and cropped.



        Returns
        -------
        :dict: dictionary containing indices for cropped regions.


        """

        # check for string or integer
        if isinstance(crop_size,str):
            crop_size = literal_eval(crop_size)
        # check for string or integer
        if isinstance(nonzero_fraction,str):
            nonzero_fraction = literal_eval(nonzero_fraction)
        # check for string or integer
        if isinstance(nuclei_index,str):
            nuclei_index = literal_eval(nuclei_index)
        # check for string or integer
        if isinstance(crop_amount,str):
            crop_amount = literal_eval(crop_amount)

        # Set the file extensions that we can use with this class
        all_ext = [
            ".ome.tif",
            ".ome.tiff",
            ".tif",
            ".tiff",
            ".h5",
            ".hdf5",
            ".nii",
        ]
        # Get file extensions for cytometry files
        tiff_ext = [".ome.tif", ".ome.tiff", ".tif", ".tiff"]
        input_image = Path(input_image)
        tiff_flag = True if input_image.suffix in tiff_ext else False
        # get the image name (remove ".ome" if it exists)
        im_stem = input_image.stem.replace(".ome","")
        # create hdf5 name
        h5_name = im_stem + ".hdf5"
        # get the image suffix
        extension = input_image.suffix

        # #Condition 1
        # if channelIDs is None:
        #     #Set number of channels to length of channel IDs
        #     num_channels = len(channelIDs)
        #     #Set channelIDs to be first n channels for num_channels
        #     channelIDs = range(0,num_channels)
        #     #Check if number of channels and channelIDs agree
        # else:
        #      raise ValueError
        num_channels = len(channelIDs)

        # check for tiff
        if tiff_flag:
            #Read the tif image - Reads the image as cyx
            logging.info("Reading "+input_image.stem+"...")
            tif = tifffile.TiffFile(input_image)
            #Set the index for the loop
            idx = 0
            #Add counter for channel index
            chan_idx = 0
            for i in range(num_channels):
                #Get the channel indices based on the step
                chan_idx = channelIDs[idx:idx+1]
                #Convert the tifffile object to array
                im = tif.asarray(series=0,key=chan_idx)
                #Reshape the array according to single slice
                im = im.reshape((1,im.shape[0],im.shape[1],1))
                #Create an hdf5 dataset if idx is 0 plane
                if idx == 0:
                    #Create hdf5
                    h5 = h5py.File(pathlib.Path(os.path.join(output,h5_name)), "w")
                    h5.create_dataset(str(im_stem), data=im[:,:,:,:],chunks=True,maxshape=(1,None,None,None))
                    h5.close()
                else:
                    #Append hdf5 dataset
                    h5 = h5py.File(pathlib.Path(os.path.join(output,h5_name)), "a")
                    #Add step size to the z axis
                    h5[str(im_stem)].resize((idx+1), axis = 3)
                    #Add the image to the new channels
                    h5[str(im_stem)][:,:,:,idx:idx+1] = im[:,:,:,:]
                    h5.close()
                #Update the index
                idx = idx+1
            #Finished exporting the image
            logging.info('Finished exporting image')

        # same function, just data must be read (if .nii or hdf5, data is memory mapped)
        else:
            #Read the tif image - Reads the image as cyx
            logging.info("Reading "+input_image.stem+"...")
            # create temporary new data
            hdi_imp = _import.HDIreader(
                            path_to_data=input_image,
                            path_to_markers=None,
                            flatten=False,
                            subsample=False,
                            method=None,
                            mask=None,
                            save_mem=False,
                            data=None,
                            image=None,
                            channels=None,
                            filename=None,
                            )
            #Set the index for the loop
            idx = 0
            #Add counter for channel index
            chan_idx = 0
            for i in range(num_channels):
                #Get the channel indices based on the step
                chan_idx = channelIDs[idx:idx+1]
                #Convert the tifffile object to array
                im = hdi_imp.hdi.data.image[:,:,chan_idx]
                #Reshape the array according to single slice
                im = im.reshape((1,im.shape[0],im.shape[1],1))
                #Create an hdf5 dataset if idx is 0 plane
                if idx == 0:
                    #Create hdf5
                    h5 = h5py.File(pathlib.Path(os.path.join(output,h5_name)), "w")
                    h5.create_dataset(str(im_stem), data=im[:,:,:,:],chunks=True,maxshape=(1,None,None,None))
                    h5.close()
                else:
                    #Append hdf5 dataset
                    h5 = h5py.File(pathlib.Path(os.path.join(output,h5_name)), "a")
                    #Add step size to the z axis
                    h5[str(im_stem)].resize((idx+step), axis = 3)
                    #Add the image to the new channels
                    h5[str(im_stem)][:,:,:,idx:idx+step] = im[:,:,:,:]
                    h5.close()
                #Update the index
                idx = idx+step
            #Finished exporting the image
            logging.info('Finished exporting image')

        # create place to store crops names
        crops = []
        #Optional to crop out regions for ilastik training
        if crop:
            #Get the index of nuclei in channelIDs
            nuclei_index = channelIDs.index(nuclei_index)
            #Run through each cropping iteration
            full_h5 = h5py.File(pathlib.Path(os.path.join(output,h5_name)), 'r')
            im_nuc = full_h5[str(im_stem)][:,:,:,nuclei_index]
            im = full_h5[str(im_stem)][:,:,:,:]
            indices = {}
            count = 0
            thresh = filters.threshold_otsu(im_nuc[:,:,:])
            while count < crop_amount:
                #Get random height value that falls within crop range of the edges
                extension_h = crop_size[0]//2
                h = random.randint(extension_h,im_nuc.shape[1]-extension_h)
                h_up, h_down = h-extension_h,h+extension_h
                #Get random width value that falls within crop range of the edges
                extension_w = crop_size[1]//2
                w = random.randint(extension_w,im_nuc.shape[2]-extension_w)
                w_lt, w_rt = w-extension_w,w+extension_w
                #Crop the image with these coordinates expanding from center
                crop = im_nuc[:, h_up:h_down, w_lt:w_rt]
                crop_name = pathlib.Path(os.path.join(output,(im_stem+"_crop"+str(count)+".hdf5")))
                #Check to see if the crop passes the nonzero fraction test
                if ((crop[0,:,:] > thresh).sum()/(crop.shape[1]*crop.shape[2])) >= nonzero_fraction:
                    #Export the image to hdf5
                    logging.info("Export "+crop_name.stem+".hdf5...")
                    crop = im[:, h_up:h_down, w_lt:w_rt,:]
                    h5_crop = h5py.File(crop_name, "w")
                    h5_crop.create_dataset(str(im_stem)+"_"+str(count), data=crop,chunks=True)
                    h5_crop.close()
                    logging.info('Finished exporting '+crop_name.stem+".hdf5")
                    #Add one to the counter
                    count=count+1
                    #Add the indices to a table to store the cropped indices
                    indices.update({crop_name.stem:[(h_up,h_down),(w_lt,w_rt)]})
                    crops.append(crop_name)
            #Export the indices to a text file to track the cropped regions
            summary = open(pathlib.Path(os.path.join(output,im_stem)+"_CropSummary.txt"),"w")
            summary.write(str(indices))
            summary.close()

        # update class attributes
        self.crops = crops
        self.crops_coords = indices
        self.full_hdf5 = str(pathlib.Path(os.path.join(output,h5_name)))

        return indices

    def PrepareTraining(self,
                input_image=None,
                output=None,
                nuclei_index=1,
                channelIDs=None,
                crop=True,
                crop_size=(250,250),
                nonzero_fraction=0.1,
                crop_amount=2):
        """Function to export yaml log to file for documentation
        """

        # log
        logging.info(f'TRAINING DATA FOLDER: {self.train_dir}')
        logging.info("PREPARING DATA FOR PIXEL CLASSIFICATION")
        # get in and out parameters
        input_image = Path(self.input_image) if input_image is None else Path(input_image)
        output = Path(self.train_dir) if output is None else Path(output)

        # add processing steps
        self.yaml_log.update({'ProcessingSteps':[]})
        self.yaml_log['ProcessingSteps'].append({"PrepareTraining":{'input_image':str(input_image),
                                        'output':str(output),
                                        'nuclei_index':nuclei_index,
                                        'channelIDs':channelIDs,
                                        'crop':crop,
                                        'crop_size':str(crop_size),
                                        'nonzero_fraction':nonzero_fraction,
                                        'crop_amount':crop_amount}})

        # prepare
        self._IlastikPrep(input_image=input_image,
                        nuclei_index=nuclei_index,
                        channelIDs=channelIDs,
                        output=output,
                        crop=crop,
                        crop_size=crop_size,
                        nonzero_fraction=nonzero_fraction,
                        crop_amount=crop_amount)

        # check for qc to go ahead and export qc files (in order to resume workflow)
        if self.qc:
            self.QC()


    def PredictHeadless(self,
                        ilp,
                        input_hdf5=None,
                        output=None,
                        output_format="tiff",
                        output_filename_format="{nickname}_probabilities.tiff",
                        export_source = "Probabilities",
                        export_dtype = "uint16",
                        pipeline_result_drange = "[0,1.0]",
                        export_drange = "[0,65535]"
                        ):
        """Function to export yaml log to file for documentation
        """

        # convert to pathlib
        self.ilp = Path(ilp)
        input_hdf5 = self.full_hdf5 if input_hdf5 is None else Path(input_hdf5)
        output = self.pred_dir if output is None else Path(output)

        # log
        logging.info("ILASTIK PIXEL CLASSIFICATION")
        logging.info(f'PROBABILITY IMAGES FOLDER: {output}')

        # update yaml logger
        self.yaml_log['ProcessingSteps'].append({"PredictHeadless":{'ilp':str(ilp),
                                        'input_hdf5':str(input_hdf5),
                                        'output':str(output),
                                        'output_format':output_format,
                                        'output_filename_format':output_filename_format,
                                        'export_source':export_source,
                                        'pipeline_result_drange':pipeline_result_drange,
                                        'export_dtype':export_dtype,
                                        'export_drange':export_drange}})

        # check other attributes and prepare for os
        output_filename_format = str(output.joinpath(output_filename_format))
        # add command
        command = self.command + " --headless"
        command = command + " --output_format=" + '"' + f"{output_format}" + '"'
        command = command + f" --project=" + '"' + f"{str(ilp)}" + '"'
        command = command + f" --output_filename_format=" + '"' + f"{output_filename_format}" + '"'
        command = command + f" --export_source=" + '"' + f"{export_source}" + '"'
        command = command + f" --export_dtype=" + '"' + f"{export_dtype}" + '"'
        command = command + f" --pipeline_result_drange=" + '"' + f"{pipeline_result_drange}" + '"'
        command = command + f" --export_drange="+ '"' + f"{export_drange}" + '"'
        command = command + ' "' + f"{str(input_hdf5)}"+'"'
        self.executed_command = command
        os.system(command)

    def _exportYAML(self):
        """Function to export yaml log to file for documentation
        """
        logging.info(f'Exporting {self.yaml_name}')
        # open file and export
        with open(self.yaml_name, 'w') as outfile:
            yaml.dump(self.yaml_log, outfile, sort_keys=False)

    def _exportSH(self):
        """Function to export sh command to file for documentation
        """
        logging.info(f'Exporting {self.sh_name}')
        # get name of the python path and cli file
        proc_fname = os.path.join(Path(_parse.__file__).parent,"_cli_ilastik.py")
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

        # provenance
        self._exportYAML()
        self._exportSH()
        # close the logger
        self.logger.handlers.clear()
