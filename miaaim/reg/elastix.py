# Elastix image registration module
# Developer: Joshua M. Hess, BSc
# Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

#Import modules
import numpy as np
import sys
import time
import os
import re
import nibabel as nib
from pathlib import Path
import pandas as pd
import tempfile
import warnings
import yaml
import logging
logging.captureWarnings(True)

# import custom modules
import miaaim.reg._utils as utils
import miaaim.reg.transformix as transformix
from miaaim.io.imread import _import
from miaaim.io.imwrite import _export
import miaaim
# for logging purposes
from miaaim.cli.reg import _parse


#Add main elastix component
def RunElastix(command):
	"""
	Run the elastix registration. You must be able to call elastix
	from your command shell to use this. You must also have your parameter
	text files set before running (see elastix parameter files).

	Parameters
	----------
	command: string
		Sent to the system for elastix running (see elastix command line implementation).
	"""

	# print command
	logging.info(f'Elastix Command: {str(command)}')
	# print elastix update
	logging.info('Running elastix...')
	# start timer
	start = time.time()
	# send the command to system
	os.system(command)
	# stop timer
	stop = time.time()
	# print update
	logging.info('Finished -- computation took '+str(stop-start)+'sec.')
	# return command
	return command


#Define elastix class structure
class ElastixReg():
    def __init__(self, mkdir=False):
        
        # input parameter
        self.mkdir = mkdir
        # parameters added by registration
        self.out_dir = None
        self.p = None
        self.moving = None
        self.fp = None
        self.mp = None
        self.fMask =None
        # other attributes
        self.fixed_channels = []
        self.moving_channels = []
        self.multichannel = None
        self.temp_dir = None
        self.elastix_dir = None
        self.tps = []
		# initialize command line strings
        self.command = "elastix"

        
    def Register(self, fixed, moving, out_dir,p,landmark_p=None,fp=None,mp=None,fMask=None,multichannel=True):
        """Register images using elastix image registration and
		parameters initialized in the class instance.

		Parameters
		----------
		fixed: string
			Path to fixed (reference) image.

		moving: string
			Path to moving image (image to be transformed).

		out_dir: string
			Path to output directory.

		p: list (length number of registration parameter files)
			Path to elastix image registration parameter files (in order of application).

		fp: string (*.txt)
			Path to fixed image landmark points for manual guidance registration.

		mp: string (*.txt)
			Path to moving image landmark points for manual guidance registration.

		fMask: string (*.nii)
			Path to fixed image mask that defines region on image to draw samples
			from during registration.
		"""
        
        # set parameters
        self.fixed = Path(fixed)
        self.out_dir = Path(out_dir)
        self.p = [Path(par_file) for par_file in p]
        self.landmark_p = [Path(par_file) for par_file in landmark_p] if landmark_p is not None else None
        self.moving = Path(moving)
        self.fp = None if fp is None else Path(fp)
        self.mp = None if mp is None else Path(mp)
        self.fMask = None if fMask is None else Path(fMask)
        self.multichannel = multichannel
        
        
	    # check for making new directories
        if self.mkdir:
            n=0
            while n>=0:
                tmp_name = "elastix"+str(n)
                if not os.path.exists(Path(os.path.join(out_dir,tmp_name))):
                    os.mkdir(Path(os.path.join(out_dir,tmp_name)))
                    elastix_dir = Path(os.path.join(out_dir,tmp_name))
                    break
                n+=1
        else:
            elastix_dir = out_dir
            
        # update elastix directory
        self.elastix_dir = elastix_dir        
        
        # now we need to check for landmarks and multichannel
        if (self.multichannel) and (self.fp is not None):
            # we must have an intermediate landmark registration
            self.landmark_reg_dir = out_dir.joinpath('landmark_registered')
            # check if exists
            if not self.landmark_reg_dir.exists():
                self.landmark_reg_dir.mkdir()
                
            # create a flag for landmarks
            self.landmark_initialize = True
        else:
            self.landmark_initialize = False

		# log information
        logging.info(f'FIXED IMAGE: {str(self.fixed)}')
        logging.info(f'MOVING IMAGE: {str(self.moving)}')
        logging.info(f'OUTPUT FOLDER: {str(self.out_dir)}')


		# load images
        niiFixed = _import.HDIreader(path_to_data=self.fixed,
									 path_to_markers=None,
									 flatten=False,
									 subsample=False,
									 method=None,
									 mask=self.fMask,
									 save_mem=False,
									 data=None,
									 image=None,
									 channels=None,
									 filename=None
									 )
        
        niiMoving = _import.HDIreader(path_to_data=self.moving,
									path_to_markers=None,
									flatten=False,
									subsample=False,
									method=None,
									mask=None,
									save_mem=False,
									data=None,
									image=None,
									channels=None,
									filename=None
									)
        

        # !! Here we check for landmark intermediate !!
        # check for landmark intermediate
        if self.landmark_initialize:
            # print updates
            logging.info("Detected landmark initialization...")
            # can use the whole image as input here -- elastix will only read
            # the first image channel unless you specify multiple channel inputs!
            # now we register using affine and landmark correspondence
            
    		# multichannel registration check for validating KNN mutual information
            if multichannel:
    			# get number of fixed image parameters
                _,_,cF = niiFixed.hdi.data.image.shape
    			# get number of moving image parameters
                _,_,mF = niiMoving.hdi.data.image.shape
                
    			#create a temporary directory using the context manager for channel-wise images
                with tempfile.TemporaryDirectory(dir=self.landmark_reg_dir) as tmpdirname:
    				#Print update
                    logging.info('Created temporary directory', tmpdirname)
    				#Print update
                    logging.info('Exporting single channel image for landmark alignment')
    		        #Read the images
                    niiFixed = niiFixed.hdi.data.image.copy()
                    niiMoving = niiMoving.hdi.data.image.copy()
                    
    				#Export single channel images for each channel of fixed image
    				#Create a filename
                    fname = Path(os.path.join(tmpdirname,str(self.fixed.stem+str(0)+self.fixed.suffix)))
    				#Update the list of names for fixed image
                    self.command = self.command + ' -f ' + str(fname)
    				#Check to see if the path exists
                    if not fname.is_file():
    					#Create a nifti image
                        nii_im = nib.Nifti1Image(niiFixed[:,:,0].T, affine=np.eye(4))
    					#Save the nifti image
                        nib.save(nii_im,str(fname))

    				#Remove the fixed image from memory
                    niiFixed = None

    				#Export single channel images for each channel of fixed image
    				#Create a filename
                    mname = Path(os.path.join(tmpdirname,str(self.moving.stem+str(0)+self.moving.suffix)))
    				#Update the list of names for moving image
                    self.command = self.command + ' -m ' + str(mname)
    				#Check to see if the path exists
                    if not mname.is_file():
    					#Create a nifti image
                        nii_im = nib.Nifti1Image(niiMoving[:,:,0].T, affine=np.eye(4))
    					#Save the nifti image
                        nib.save(nii_im,str(mname))

    				#Remove the moving image from memory
                    niiMoving = None
                    
                    # add the parameter files
                    # self.command = self.command+' '.join([" -p "+str(self.landmark_p[0])])
                    self.command = self.command + ' '.join([" -p " + str(self.landmark_p[par_file]) for par_file in range(len(self.landmark_p))])
        			#Add to the command
                    self.command = self.command +" -fp "+str(self.fp)+" -mp "+str(self.mp)
                	    #Check for fixed mask
                    if fMask is not None:
                        #Add the fixed mask to the command if it exists
                        self.command = self.command +" -fMask "+str(fMask)
                    #Add the output directory to the command
                    self.command = self.command +" -out "+str(self.landmark_reg_dir)       
        			#Run elastix without creating temporary directory
                    RunElastix(self.command)
            
            # otherwise, only use the fixed and moving images as usual
            else:
    	        #Add fixed and moving image to the command string
                self.command = self.command+" -f "+str(self.fixed)+ " -m "+str(self.moving)     
            
                # add the parameter files
                self.command = self.command+' '.join([" -p "+str(self.landmark_p[0])])
    			#Add to the command
                self.command = self.command +" -fp "+str(self.fp)+" -mp "+str(self.mp)
            	    #Check for fixed mask
                if fMask is not None:
                    #Add the fixed mask to the command if it exists
                    self.command = self.command +" -fMask "+str(fMask)
                #Add the output directory to the command
                self.command = self.command +" -out "+str(self.landmark_reg_dir)       
    			#Run elastix without creating temporary directory
                RunElastix(self.command)
            
            # now we must use transformix to transform the input image according 
            # to the landmark initialization...
            # print updates
            logging.info("Transforming according to landmark initialization...")
            
            # access the transform parameters from the landmark registration
            _, landmark_tps = utils.GetFirstTransformParameters(dir=self.landmark_reg_dir)
            # add these transform parameters to list of all transformation pars
            self.tps.append(landmark_tps)
            
            # create transformix
            transformer = transformix.Transformix(in_im=self.moving,
                                                  out_dir=self.landmark_reg_dir, 
                                                  tps=landmark_tps, 
                                                  target_size=None, 
                                                  pad=None, 
                                                  trim=None, 
                                                  crops=None, 
                                                  out_ext=None)
            # we should be able to access the exported file name
            # we add this export file name as the new fixed image
            self.moving = transformer.out_name
            # rest of script proceeds using intermediate landmark image as the
            # fixed image
            # print update
            logging.info("Finished landmark initialization")
            
        
        # reset the command!
        self.command = "elastix"
        
		# load images
        niiFixed = _import.HDIreader(path_to_data=self.fixed,
									 path_to_markers=None,
									 flatten=False,
									 subsample=False,
									 method=None,
									 mask=self.fMask,
									 save_mem=False,
									 data=None,
									 image=None,
									 channels=None,
									 filename=None
									 )
        
        niiMoving = _import.HDIreader(path_to_data=self.moving,
									path_to_markers=None,
									flatten=False,
									subsample=False,
									method=None,
									mask=None,
									save_mem=False,
									data=None,
									image=None,
									channels=None,
									filename=None
									)
        
		# multichannel registration check for validating KNN mutual information
        if multichannel:
			# get number of channels for fixed and moving
            if niiFixed.hdi.data.image.ndim == 2 or niiMoving.hdi.data.image.ndim == 2:
				# raise warning that aMI not recommended for single parameters
                warnings.warn("KNN alpha MI not recommended for single channel images. Use mattes MI metric instead.", RuntimeWarning)
			# get number of fixed image parameters
            _,_,cF = niiFixed.hdi.data.image.shape
			# get number of moving image parameters
            _,_,mF = niiMoving.hdi.data.image.shape

			# validate parameters
            new_ps = utils.ValidateKNNaMIparams(cF,mF,p)
			# update self
            self.p = [Path(par_file) for par_file in new_ps]

	    #Add the parameter files
        self.command = self.command+' '.join([" -p "+str(self.p[par_file]) for par_file in range(len(self.p))])

	    #Check for corresponding points in registration (must have fixed and moving set both)
        if self.fp and self.mp is not None:
			#Add to the command
            self.command = self.command +" -fp "+str(self.fp)+" -mp "+str(self.mp)

	    #Check for fixed mask
        if fMask is not None:
			#Add the fixed mask to the command if it exists
            self.command = self.command +" -fMask "+str(fMask)

	    #Add the output directory to the command
        self.command = self.command +" -out "+str(self.elastix_dir)

	    #Check to see if there is single channel input (grayscale)
        if niiFixed.hdi.data.image.ndim == 2 and niiMoving.hdi.data.image.ndim == 2:
            logging.info('Detected single channel input images')
	        #Add fixed and moving image to the command string
            self.command = self.command+" -f "+str(self.fixed)+ " -m "+str(self.moving)
			#Update the fixed channels
            self.fixed_channels.append(self.fixed)
			#Update the moving channels
            self.moving_channels.append(self.moving)
			#Update whether this is a multichannel input or not
            self.multichannel = False

			#Run elastix without creating temporary directory
            RunElastix(self.command)

	    #Check to see if there is multichannel input
        else:
			#create a temporary directory using the context manager for channel-wise images
            with tempfile.TemporaryDirectory(dir=self.elastix_dir) as tmpdirname:
				#Print update
                logging.info('Created temporary directory', tmpdirname)
				#Print update
                logging.info('Exporting single channel images for multichannel input')
		        #Read the images
                niiFixed = niiFixed.hdi.data.image.copy()
                niiMoving = niiMoving.hdi.data.image.copy()
				#Update multichannel class option
                self.multichannel = True

				#Export single channel images for each channel of fixed image
                for i in range(niiFixed.shape[2]):
					#Create a filename
                    fname = Path(os.path.join(tmpdirname,str(self.fixed.stem+str(i)+self.fixed.suffix)))
					#Update the list of names for fixed image
                    self.fixed_channels.append(fname)
					#Update the list of names for fixed image
                    self.command = self.command + ' -f' + str(i) + ' ' + str(fname)
					#Check to see if the path exists
                    if not fname.is_file():
						#Create a nifti image
                        nii_im = nib.Nifti1Image(niiFixed[:,:,i].T, affine=np.eye(4))
						#Save the nifti image
                        nib.save(nii_im,str(fname))

				#Remove the fixed image from memory
                niiFixed = None

				#Export single channel images for each channel of fixed image
                for i in range(niiMoving.shape[2]):
					#Create a filename
                    mname = Path(os.path.join(tmpdirname,str(self.moving.stem+str(i)+self.moving.suffix)))
					#Update the list of names for moving image
                    self.moving_channels.append(mname)
					#Update the list of names for moving image
                    self.command = self.command + ' -m' + str(i) + ' ' + str(mname)
					#Check to see if the path exists
                    if not mname.is_file():
						#Create a nifti image
                        nii_im = nib.Nifti1Image(niiMoving[:,:,i].T, affine=np.eye(4))
						#Save the nifti image
                        nib.save(nii_im,str(mname))

				#Remove the moving image from memory
                niiMoving = None
				#Run the command using the function created
                RunElastix(self.command)

        # gather transform parameter files and add to class
        _, tps = utils.GetFirstTransformParameters(dir=self.elastix_dir)
        # update the list of transform parameters
        self.tps.append(tps)

class InverseElastixReg():
	"""Invert an image registration process using elastix.

	Parameters
	----------
	fixed: string
		Path to fixed (reference) image.

	out_dir: string
		Path to output directory.

	t: string
		final transform parameter file of forward registration. Example:
		't=path/to/TransformParameters.txt'

	p: list (length number of registration parameter files)
		Path to inverse elastix image registration parameter files (in order of application).

	p_forward: list (length number of registration parameter files)
		Path to elastix forward image registration parameter files (in order of application).
	"""

	def __init__(self,fixed,out_dir,t,p=None,p_forward=None,mkdir=True):

		#Create pathlib objects and set class parameters
		self.fixed = Path(fixed)
		self.multichannel = None
		self.out_dir = Path(out_dir)
		self.mkdir = mkdir
	    #Check for making new directories
		if mkdir is True:
			n=0
			while n>=0:
				tmp_name = "elastixInverse"+str(n)
				if not os.path.exists(Path(os.path.join(out_dir,tmp_name))):
					os.mkdir(Path(os.path.join(out_dir,tmp_name)))
					elastix_dir = Path(os.path.join(out_dir,tmp_name))
					break
				n+=1
		else:
			elastix_dir = out_dir
		self.elastix_dir = elastix_dir
		self.temp_dir = None
		if (p_forward is not None) & (p is None):
			# create transform files
			p = utils.CreateInverseTransformFiles(p_forward)
		else:
			p = p
		self.p = [Path(par_file) for par_file in p]
		self.t = Path(t)
		self.command = "elastix"

		#Load the images to check for dimension number
		print('Loading images...')
		#Load images
		niiFixed = nib.load(str(self.fixed))
		#Print update
		print('Done loading')

	    #Add the parameter files
		self.command = self.command+' '.join([" -p "+str(self.p[par_file]) for par_file in range(len(self.p))])
	    #Add the output directory to the command
		self.command = self.command +" -out "+str(self.elastix_dir)
	    #Add forard transform parameters to command
		self.command = self.command +" -t0 "+str(self.t)

	    #Check to see if there is single channel input (grayscale)
		if niiFixed.ndim == 2:
			print('Detected single channel input images...')
	        #Add fixed and moving image to the command string
			self.command = self.command+" -f "+str(self.fixed)
			self.command = self.command+" -m "+str(self.fixed)
			#Update the fixed channels
			self.fixed_channels.append(self.fixed)
			#Update whether this is a multichannel input or not
			self.multichannel = False

			#Run elastix without creating temporary directory
			RunElastix(self.command)

	    #Check to see if there is multichannel input
		else:
			#create a temporary directory using the context manager for channel-wise images
			with tempfile.TemporaryDirectory(dir=self.elastix_dir) as tmpdirname:
				#Print update
				logging.info('Created temporary directory', tmpdirname)
				#Read the images
				niiFixed = niiFixed.get_fdata()
				#Update multichannel class option
				self.multichannel = True

				#Export single channel images for each channel of fixed image
				for i in range(1):
					#Create a filename
					fname = Path(os.path.join(tmpdirname,str(self.fixed.stem+str(i)+self.fixed.suffix)))
					#Update the list of names for fixed image
					self.fixed_channels.append(fname)
					#Update the list of names for fixed image and stand in for moving
					self.command = self.command + ' -f' + str(i) + ' ' + str(fname)
					self.command = self.command + ' -m' + str(i) + ' ' + str(fname)
					#Check to see if the path exists
					if not fname.is_file():
						#Create a nifti image
						nii_im = nib.Nifti1Image(niiFixed[:,:,i].T, affine=np.eye(4))
						#Save the nifti image
						nib.save(nii_im,str(fname))

				#Remove the fixed image from memory
				niiFixed = None
				#Run the command using the function created
				RunElastix(self.command)

		# read first transform parameter file
		first_t, _ = utils.GetFirstTransformParameters(dir=self.elastix_dir)
		# update initial transformation file w/ no initial transform
		utils.AlterInverseTransform(input=first_t)

		def Invert():
			raise NotImplementedError




class Elastix():
    def __init__(self, root_folder, name, resume=True):
        
        # create containers for each elastix, transformix, etc.
        self.elastix = None
        self.transformix = None
        self.inverseElastix = None
        self.inverseTransformix = None
        self.maskTransformix = None
        self.boundaryMaskTransformix = None
        self.landmarksTransform = None
        
        # parameters for workflow
        self.name = name
        self.root_folder = Path(root_folder)
        self.registration_dir = None
        self.registration_name_dir = None
        self.qc_dir = None
        self.qc_reg_dir = None
        self.qc_reg_name_dir = None
        self.docs_dir = None
        self.log_name = None
        self.logger = None
        self.sh_name = None
        self.yaml_name = None
        
        # input parameters not yet included
        self.resume = resume
        self.qc = True
        
        
        # create logger format
        FORMAT = '%(asctime)s | [%(pathname)s:%(lineno)s - %(funcName)s() ] | %(message)s'
        
        # create process directory
        registration_dir = Path(root_folder).joinpath("registration")
        # check if exists already
        if not registration_dir.exists():
            # make it
            registration_dir.mkdir()
  
        # create named directory
        registration_name_dir = Path(registration_dir).joinpath(f"{self.name}")
        # check if exists already
        if not registration_name_dir.exists():
            # make it
            registration_name_dir.mkdir()        
  
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
        yaml_name = os.path.join(Path(pars_dir),"miaaim-registration"+f'-{self.name}'+".yaml")

        # create a qc directory
        qc_dir = docs_dir.joinpath("qc")
        # check if exists already
        if not qc_dir.exists():
            # make it
            qc_dir.mkdir()

        # create direectory specific for process
        qc_reg_dir = qc_dir.joinpath("registration")
        # check if exists already
        if not qc_reg_dir.exists():
            # make it
            qc_reg_dir.mkdir()
        
        qc_reg_name_dir = qc_reg_dir.joinpath(f'{self.name}')
        # check if exists already
        if not qc_reg_name_dir.exists():
            # make it
            qc_reg_name_dir.mkdir()

        # create a qc directory
        prov_dir = docs_dir.joinpath("provenance")
        # check if exists already
        if not prov_dir.exists():
            # make it
            prov_dir.mkdir()
        # create name of logger
        log_name = os.path.join(Path(prov_dir),"miaaim-registration"+f'-{self.name}'+".log")

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
            # update logger with version number of miaaim
            self.yaml_log.update({"MIAAIM VERSION":miaaim.__version__})
            # update yaml file
            self.yaml_log.update({'MODULE':"Registration"})
            self.yaml_log.update({'METHOD':"elastix"})
            self.yaml_log.update({'ImportOptions':{'root_folder':str(self.root_folder),
                                                    'name':str(self.name),
                                                    'resume':self.resume}})            
            # update yaml_log with processing steps (initialize to empty)
            self.yaml_log.update({'ProcessingSteps':[]})

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
        sh_name = os.path.join(Path(prov_dir),"miaaim-registration"+f'-{self.name}'+".sh")

        # update attributes
        self.root_folder = root_folder
        self.registration_dir = registration_dir
        self.registration_name_dir = registration_name_dir
        self.docs_dir = docs_dir
        self.qc_dir = qc_dir
        self.qc_reg_dir = qc_reg_dir
        self.qc_reg_name_dir = qc_reg_name_dir
        self.prov_dir = prov_dir
        self.log_name = log_name
        self.logger = logger
        self.sh_name = sh_name
        self.yaml_name = yaml_name
        # create objects to be filled
        self.fixed = None
        self.moving = None
        self.p = None
        self.tps = None
        
    def Register(self, fixed, moving, p, landmark_p=None, out_dir=None, fp=None, mp=None, fMask=None, multichannel=True):
        # create pathlib objects
        fixed = Path(fixed)
        moving = Path(moving)
        p = [ Path(i) for i in p ]
        out_dir = Path(out_dir) if out_dir is not None else self.registration_name_dir
        # update class attributes
        self.fixed = fixed
        self.moving = moving
        self.p = p
    
		# log
        logging.info('ELASTIX IMAGE REGISTRATION')
        # update logger
        self.yaml_log['ProcessingSteps'].append(({"Register":{'fixed':str(fixed),
															  'moving':str(moving),
															  'out_dir':str(out_dir),
															  'p':[ str(i) for i in p ],
                                                               'landmark_p': landmark_p,
															  'fp':fp,
															  'mp':mp,
															  'fMask':fMask,
															  'multichannel':multichannel}}))

		# run elastix registration class
        self.elastix = ElastixReg()
        self.elastix.Register(fixed=fixed,
                        moving=moving,
                        out_dir=out_dir,
                        p=p,
                        landmark_p=landmark_p,
                        fp=fp,
                        mp=mp,
                        fMask=fMask,
                        multichannel=multichannel
                        )
        
    def Transform(self, in_im=None, out_dir=None, tps=None, target_size=None, pad=None, trim=None, crops=None, out_ext=".nii"):
        # create pathlib objects and set defaults
        in_im = Path(in_im) if out_dir is not None else self.moving
        out_dir = Path(out_dir) if out_dir is not None else self.registration_name_dir
        tps = [Path(par_file) for par_file in tps] if tps is not None else self.p
        target_size = tuple(target_size) if target_size is not None else None
        pad = pad if pad is not None else None
        trim = trim if trim is not None else None
        # !!! set crops to None for now !!!
        crops = None
         
		# log
        logging.info('TRANSFORMIX IMAGE TRANSFORMER')

        # update logger
        self.yaml_log['ProcessingSteps'].append(({"Transform":{'fixed':str(in_im),
															   'out_dir':str(out_dir),
															   'tps':[ str(i) for i in tps ],
															   'target_size':str(target_size),
                                                                'pad':str(pad),
                                                                'trim':str(trim),
                                                                'crops':crops,
															   'out_ext':str(out_ext)}}))

		# run transformix class
        self.transformix = transformix.Transformix()
        self.transformix.Transform(in_im=in_im, 
                                   out_dir=out_dir, 
                                   tps=tps, 
                                   target_size=target_size, 
                                   pad=pad, 
                                   trim=trim, 
                                   crops=crops,
                                   out_ext=".nii"
                                   )
        
    def Invert(self, fixed=None, out_dir=None, t=None, p_forward=None, p=None, mkdir=True):
		# log
        # logging.info(f'COMPUTING APPROXIMATE INVERSE TRANSFORMATION')
        # update logger
        raise NotImplementedError
		# run elastix registration class
        # self.inverseElastix = InverseElastixReg()
        # self.inverseElastix.Invert(fixed=fixed, out_dir=out_dir, t=t, p_forward=p_forward, p=p, mkdir=mkdir)
    
    def InverseTransform(self):
		# log
        # logging.info(f'APPLYING APPROXIMATE INVERSE TRANSFORMATION')
        # update logger
        raise NotImplementedError
		# run elastix registration class
        # self.inverseTransformix = Transformix()
        # self.inverseTransformix.Transform(fixed=fixed, out_dir=out_dir, t=t, p_forward=p_forward, p=p, mkdir=mkdir)
        
    def MaskTransform(self):
        """Pull from the QC prep directory
        """
		# log
        # logging.info(f'TRANSFORMIX MASK TRANSFORMER')
        # update logger
        raise NotImplementedError
		# run elastix registration class
        # self.maskTransform = Transformix()
        # self.maskTransform.Transform(fixed=fixed, out_dir=out_dir, t=t, p_forward=p_forward, p=p, mkdir=mkdir)
        
    def TransformVectorField(self):
        raise NotImplementedError
        
    def BoundaryMaskTransform(self):
        """Pull from the QC prep directory
        """
        raise NotImplementedError
        
    def LandmarksTransform(self):
        raise NotImplementedError
        
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
        proc_fname = os.path.join(Path(_parse.__file__).parent,"_cli_elastix.py")
        # get path to python executable
        # create shell command script
        with open (self.sh_name, 'w') as rsh:
            rsh.write(f'''\
            #! /bin/bash
			{sys.executable} {proc_fname} --pars {self.yaml_name}''')
            
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












































#
