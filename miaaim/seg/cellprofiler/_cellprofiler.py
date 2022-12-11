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
from subprocess import Popen

# import custom modules
import miaaim
#

import cellprofiler_core.pipeline
import cellprofiler_core.preferences
import cellprofiler_core.utilities.java
import pathlib

cellprofiler_core.preferences.set_headless()



test = Path("/Users/joshuahess/Desktop/test_new/ROI023_LIVER_D12/probabilities/imc/ilastik")
test = CellProfiler(root_folder="/Users/joshuahess/Desktop/test_new/ROI023_LIVER_D12")
test.PredictHeadless(p="/Users/joshuahess/Desktop/segmentation.cppipe",
                    plugins_directory="/Users/joshuahess/Desktop/CellProfiler-plugins")
test.executed_command

def SearchDir(dir, string = "CellProfiler"):
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

# exe,comm = _cellprofiler_headless_search()

def _cellprofiler_headless_search():
    """Light function to check system platform and search common
    installation locations for cellprofiler if not in system path.

    Returns
    -------
    exe: pathlib.Path with path to cellprofiler application

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
        comm = exe.joinpath("CellProfiler/CellProfiler.exe")
    # check for linux
    elif platform.system() == "Linux":
        # assume docker image and search opt/ilastik
        p = Path("/opt/cellprofiler")
        # search
        exe = SearchDir(dir=p)[-1]
        # shell script inside version folder
        comm = exe.joinpath("Contents")
    # check for OS (needs work)
    elif platform.system() == "Darwin" or os.name == 'posix':
        # search applications
        p = Path("/Applications")
        # get executable (will be .app)
        exe = SearchDir(dir=p)[-1]
        # get shell script location
        comm = exe.joinpath("Contents/MacOS/cp")
    # search and return latest ilastik
    return exe, comm



class CellProfiler():
    """CellProfiler single cell segmentation.
    """
    def __init__(self,
                root_folder,
                probs_type="ilastik",
                name="imc",
                executable=None,
                qc=True):

        self.root_folder = root_folder
        self.in_directory = None
        self.executable = executable
        self.command = None
        self.executed_command = None
        self.qc = qc
        # other parameters
        self.cppipe = None
        self.prob_dir = None
        # qc and provenance info
        self.name = name
        self.qc_dir = None
        self.qc_seg_dir = None
        self.qc_seg_name_dir = None
        self.docs_dir = None
        self.seg_dir = None
        self.log_name = None
        self.logger = None
        self.sh_name = None
        self.yaml_name = None

        # check for executable
        if executable is None:
            try:
                # check for it
                executable, command = _cellprofiler_headless_search()
            except:
                raise ValueError('Cannot find cellprofiler application. Please specify its location manually.')

        # update executable information
        self.executable = executable
        self.command = str(command) # convert pathlib to string for sending to os


        # create logger format
        FORMAT = '%(asctime)s | [%(pathname)s:%(lineno)s - %(funcName)s() ] | %(message)s'

        root_folder = Path(root_folder)

        # create names
        self.in_directory = root_folder.joinpath(f"probabilities/{self.name}/{probs_type}")
        # !!!! get image (just return all contents) -- assumes only one image !!!!
        # !!!!!!!!
        probabilities_image = SearchDir(self.in_directory ,".tiff")[0]
        if probabilities_image is None:
            probabilities_image = SearchDir(self.in_directory ,".tif")[0]

        # create direectory specific for process
        pred_dir = root_folder.joinpath("segmentation")
        # check if exists already
        if not pred_dir.exists():
            # make it
            pred_dir.mkdir()
        # create direectory specific for process
        pred_dir = pred_dir.joinpath(f"cellprofiler-{probs_type}")
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
        yaml_name = os.path.join(Path(pars_dir),"miaaim-seg-cellprofiler"+f'-{self.name}'+".yaml")

        # create a qc directory
        qc_dir = docs_dir.joinpath("qc")
        # check if exists already
        if not qc_dir.exists():
            # make it
            qc_dir.mkdir()

        # create direectory specific for process
        qc_seg_dir = qc_dir.joinpath("segmentation")
        # check if exists already
        if not qc_seg_dir.exists():
            # make it
            qc_seg_dir.mkdir()
        # create direectory specific for process
        qc_seg_dir = qc_seg_dir.joinpath(f'{self.name}')
        # check if exists already
        if not qc_seg_dir.exists():
            # make it
            qc_seg_dir.mkdir()

        qc_seg_name_dir = qc_seg_dir.joinpath(f"cellprofiler-{probs_type}")
        # check if exists already
        if not qc_seg_name_dir.exists():
            # make it
            qc_seg_name_dir.mkdir()

        # create a qc directory
        prov_dir = docs_dir.joinpath("provenance")
        # check if exists already
        if not prov_dir.exists():
            # make it
            prov_dir.mkdir()
        # create name of logger
        log_name = os.path.join(Path(prov_dir),"miaaim-seg-cellprofiler"+f'-{self.name}'+".log")

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
                                level=logging.INFO,
                                format=FORMAT)

        # get logger
        logger = logging.getLogger()
        # command to capture print functions to log
        print = logger.info

        # create name of shell command
        sh_name = os.path.join(Path(prov_dir),"miaaim-seg-cellprofiler"+f'-{self.name}'+".sh")

        # update attributes
        self.root_folder = root_folder
        self.probabilities_image = probabilities_image
        self.pred_dir = pred_dir
        self.docs_dir = docs_dir
        self.qc_dir = qc_dir
        self.qc_prob_dir = qc_seg_dir
        self.qc_prob_name_dir = qc_seg_name_dir
        self.prov_dir = prov_dir
        self.log_name = log_name
        self.logger = logger
        self.sh_name = sh_name
        self.yaml_name = yaml_name

        # print first log
        logging.info("MIAAIM SEGMENTATION")
        logging.info(f'MIAAIM VERSION {miaaim.__version__}')
        logging.info(f'METHOD: CellProfiler')
        logging.info(f'ROOT FOLDER: {self.root_folder}')
        logging.info(f'PROVENANCE FOLDER: {self.prov_dir}')
        logging.info(f'QC FOLDER: {self.qc_seg_name_dir} \n')

        # update yaml file
        self.yaml_log.update({'MODULE':"CELL SEGMENTATION"})
        self.yaml_log.update({'METHOD':"CellProfiler"})
        self.yaml_log.update({'ImportOptions':{'root_folder':root_folder,
                                                'name':self.name,
                                                'executable':self.executable,
                                                'qc':qc}})
        # update logger
        logging.info(f'\n')
        logging.info("PROCESSING DATA")


    def PredictHeadless(self,p,i=None,o=None,plugins_directory=None):
        """Function to export yaml log to file for documentation
        """

        # convert to pathlib
        p = Path(p)
        plugins_directory = Path(plugins_directory) if plugins_directory is not None else None
        i = Path(i) if i is not None else self.in_directory
        o = Path(o) if o is not None else self.pred_dir
        self.cppipe = p

        # log
        logging.info("CELLPROFILER SEGMENTATION")
        logging.info(f'SEGMENTATION FOLDER: {str(o)}')

        # add processing steps
        self.yaml_log.update({'ProcessingSteps':[]})
        self.yaml_log['ProcessingSteps'].append({"PredictHeadless":{'p':str(p),
                                        'i':str(i),
                                        'o':str(o)}})

        # add command
        command = self.command + " -c -r"
        command = command + f" -p " + '"' + f"{str(p)}" + '"'
        command = command + f" -o " + '"' + f"{str(o)}" + '"'
        command = command + f" -i " + '"' + f"{str(i)}" + '"'
        command = command + f" --plugins-directory=" + '"' + f"{str(plugins_directory)}" + '"'
        self.executed_command = command
        #os.system(command)
        sts = Popen(command, shell=True).wait()


    def _exportYAML(self):
        """Function to export yaml log to file for documentation
        """
        logging.info(f'Exporting {self.yaml_name}')
        # open file and export
        with open(self.yaml_name, 'w') as outfile:
            yaml.dump(self.yaml_log, outfile, default_flow_style=False)

    def _exportSH(self):
        """Function to export sh command to file for documentation
        """
        logging.info(f'Exporting {self.sh_name}')
        # get name of the python path and cli file
        proc_fname = os.path.join(Path(_parse.__file__).parent,"_cli_cellprofiler.py")
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
