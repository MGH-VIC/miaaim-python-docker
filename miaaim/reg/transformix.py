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
from skimage.transform import resize
import shutil
from tqdm import tqdm
from ast import literal_eval

#Import external modules
from miaaim.io.imread import _import
from miaaim.io.imwrite import _export

def GetCropCoords(coords_csv, correction = 80):
	"""
	Read csv file containing coordinates for rectangular crop through
	FIJI and return the coordinates as a list.

	Parameters
	----------
	coords_csv: string.
		Path to csv file containing coordinates.

	correction: integer (Default: 80)
		Extra pixels to take on each side of the crop contour for buffer.

	Returns
	-------
	list with min row, max row, min col, max col coordinates of bounding box.
	"""

	#Ensure that the csv is pathlib object
	coords_csv = Path(coords_csv)
	#Read the file
	file_cont = pd.read_csv(coords_csv)

	#Get the XY coordinates from the dataframe
	col = list(file_cont['X']-1)
	row = list(file_cont['Y']-1)

	#Get min and max for rows
	min_row = min(row)
	max_row = max(row)
	#Get column min and max
	min_col = min(col)
	max_col = max(col)

	#Check to see if we are adding overlap to our ROIs
	if correction != None:
		#Get the bounding box
		min_row = (min_row-(correction))
		max_row = (max_row+(correction))
		min_col = (min_col-(correction))
		max_col = (max_col+(correction))

	#Return the contour of the rectangular crop
	return [min_row,max_row,min_col,max_col]



def CropROI(coords,full_img,target_size = None):
	"""Extract region of interest from full image array.

	Parameters
	----------
	coords: list
		Returned from ``GetCropCoords``

	full_img: array
		Contains the image to be cropped from.

    target_size: tuple of type integer (sizex,sizey; Default: None)
        Resize image using bilinear interpolation before exporting.

	Returns
	-------
	nifti_ROI: object of class nibabel.Nifti1Image
		Returns nibabel nifti object that is the cropped region to be exported.
	"""

	#Get region of interest crop from the full array using coorinates
	tmp_ROI = full_img[int(coords[2]):int(coords[3]),int(coords[0]):int(coords[1])]
	#Get the size of the imc image
    #multiple = int(imc_im.shape[1]/tmp_ROI.shape[0])
    #Use the rounded multiple to resize our ROI for image registration
    #HiRes_size = (tmp_ROI.shape[1]*multiple,tmp_ROI.shape[0]*multiple)
	#HiRes_size = (imc_im.shape[0],imc_im.shape[1])

	if target_size is not None:
		#Remember that cv2 does the axis in the opposite order as numpy
		# note here that we are no longer using cv2 -- this is skimage resizing...
		tmp_ROI = resize(tmp_ROI,target_size)

	#Create a nifti object
	nifti_ROI = nib.Nifti1Image(np.rot90(tmp_ROI.T,1), affine=np.eye(4))
	#Return image
	return nifti_ROI



def ExtractROIcrop(coords_csv, full_img, target_size = None, correction = 80):
	"""
	Combine ``CropROI`` and ``GetCropCoords`` to return
	a nifti object available for export.

	Parameters
	----------
	coords_csv: string.
		Path to csv file containing coordinates.

	full_img: array
		Contains the image to be cropped from.

    target_size: tuple of type integer (sizex,sizey; Default: None)
        Resize image using bilinear interpolation before exporting.

	correction: integer (Default: 80)
		Extra pixels to take on each side of the crop contour for buffer.

	Returns
	-------
	crop_im: object of class nibabel.Nifti1Image
		Returns nibabel nifti object that is the cropped region to be exported.
	"""

	#Extract the coordinates from the csv file
	coords = GetCropCoords(coords_csv, correction)
	#Extract the nifti cropped region
	crop_im = CropROI(coords,full_img,target_size)

	#Return the cropped image -- nifti formatted
	return crop_im



#Define base component transformix
def RunTransformix(command):
	"""Run the transfomix transformation. You must be able to call transformix
	from your command shell to use this. You must also have your transformation
	parameter files set before running (see transformix parameter files).

	Parameters
	----------
	command: string
		Sent to the system for transformix running (see elastix command line implementation).

	Returns
	-------
	command: string
		Same string as above.
	"""

	#Print command
	print(str(command))
	#Print transformix update
	#print('Running transformix...')
	#Send the command to the shell
	os.system(command)
	#Print update
	#print('Finished')
	#Return values
	return command


def CreateCompositeTransforms(tps, out_dir):
	"""Apply a series of transformations to an image using transformix. Prior
	results show that composing transforms in elastix do not yield the same
	result as appying each transformation in series. This will take more time
	but will give the expected result. A new copy of transform parameters will
	be created and exported to the out_dir

	Parameters
	----------
	tps: list (length number of transformation parameter files)
		Path to elastix image registration transform parameter files (in order of application).

	out_dir: string
		Path to output directory for transform parameters and output results.

	Returns
	-------
	transform_calls: list
		List of transform parameters to insert into the transformix command "--tp".

	init_trans_list: list
		List of initial transformations in the tranform parameter files. These
		are part of the actual transform parameter .txt files.
	"""

	#Ensure the input tps are pathlib objects
	tps = [Path(t) for t in tps]
	#Ensure the out directory is a pathlib object
	out_dir = Path(out_dir)

	#Create a list of new filenames that will be exported
	new_tps = [Path(os.path.join(str(out_dir),"TransformParameters_comp."+str(i)+".txt")) for i in range(len(tps))]
	#Create a list of items to store for no initial transforms -- these are going
	init_trans_list = []
	#Create a lsit of files that transformix will call on
	transform_calls = []

	#Create a dictionary that stores old tps and new tps
	tp_dict = {}
	#Iterate through tps and add to dictionary
	for t in range(len(tps)):
		#Access the old tp name and new, add to the dictionary
		tp_dict.update({tps[t]:new_tps[t]})

	#Iterate through all other files and change inital transforms
	for t in range(0,len(tps[1:])+1):
		#Extract and modify the first file in the list of transform parameters
		with open(tps[t], 'r') as f:
			#Read the file
			filedata = f.read()
			#Search for the initial transform parameter
			for l in list(filedata.split("\n")):
				#Check if inital transform string is in the line
				if "InitialTransformParametersFileName" in l:
					#Extract the inital transform parameter string
					init_trans = l.split(" ")[1].strip(")").strip('"')
			#Update the list of init trans
			init_trans_list.append(init_trans)

	#Now iterate through the init trans files and add true or false for whether they are essential (called in transformix)
	for t in range(0,len(init_trans_list[1:])):
		#Extract the init trans file and the one above -- skip last one -- always used!
		if init_trans_list[t+1] == 'NoInitialTransform':
			#Update the list at t to be true
			transform_calls.append(new_tps[t])

	#Add the last transform parameter to the transform call by default
	transform_calls.append(new_tps[len(new_tps)-1])

	#Iterate through all other files and change inital transforms
	for t in range(0,len(tps[1:])+1):
		#Extract and modify the first file in the list of transform parameters
		with open(tps[t], 'r') as f:
			#Read the file
			filedata = f.read()
			#Search for the initial transform parameter
			for l in list(filedata.split("\n")):
				#Check if inital transform string is in the line
				if "InitialTransformParametersFileName" in l:
					#Extract the inital transform parameter string
					init_trans = l.split(" ")[1].strip(")").strip('"')

		#Check to ensure that the initial transform is no initial transform
		if init_trans != 'NoInitialTransform':
			#Get the corresponding new file name from the old
			#n_tp = tp_dict[Path(init_trans)]
			n_tp = new_tps[t-1]
			#Then replace the initial transform in the file with the new tp
			filedata = filedata.replace(init_trans, str(n_tp))

		#Write out the new data to the new transform parameter filename
		with open(new_tps[t], 'w') as file:
			#Write the new file
			file.write(filedata)

	#Remove the filedata for now
	filedata = None

	#Return the list of new parameter files
	return transform_calls,init_trans_list



def MultiTransformix(in_im, out_dir, tps):
	"""Applies multiple transform files in sequence to an image.

	Parameters
	----------
	in_im: string
		Input image path.

	tps: list (length number of transformation parameter files)
		Path to elastix image registration transform parameter files (in order of application).

	out_dir: string
		Path to output directory for transform parameters and output results.
	"""

	#Ensure the input image is pathlib object
	in_im = Path(in_im)
	#Ensure the input tps are pathlib objects
	tps = [Path(t) for t in tps]
	#Ensure the out directory is a pathlib object
	out_dir = Path(out_dir)

	#Create transformix command
	command = "transformix"

	#Run the CreateCompositeTransforms(tps, out_dir) function
	trans_calls,init_trans_list = CreateCompositeTransforms(tps, out_dir)
	print("Created transform parameters length:" + str(len(trans_calls)))

	#Create temporary directory in the out_dir
	with tempfile.TemporaryDirectory(dir=out_dir) as nestdirname:
		#Run the first transformation
		tmp_command = command +' -out ' + str(nestdirname)
		tmp_command = tmp_command + ' -tp ' + str(trans_calls[0])
		tmp_command = tmp_command + ' -in ' + str(in_im)

		#Iterate through each transform parameter file and run transformix
		RunTransformix(tmp_command)

		#rint("Getting first results")

		#Get a result name for the output of transformix (assumes nifti for now)
		res_name = Path(os.path.join(str(nestdirname),"result"+in_im.suffix))

		#print("Got first results")

		#Check to see if the list is larger than 2 (if no then only take the last parameter file)
		if len(trans_calls) > 1:
			#Print update
			#print("Number of transform calls:" + str(len(trans_calls)))

			#Now iterate through all the other transform parameter files and run transformix
			for t in range(1,len(trans_calls)):
				#print("Running transform call:" + str(t))

				#Create the temporary transformix command
				tmp_command = command

				#Check to see if this is the last iteration
				if t == (len(trans_calls)-1):
					#Add the result name
					tmp_command = tmp_command + ' -in ' + str(res_name)
					#Make the output directory the final out_dir
					tmp_command = tmp_command +' -out ' + str(out_dir)
					#Update result name
					res_name = Path(os.path.join(str(out_dir),"result"+in_im.suffix))

				#Otherwise, leave it as the tmp directory
				else:
					#Add the result name to the command
					tmp_command = tmp_command + ' -in ' + str(res_name)
					#Run transformixs
					tmp_command = tmp_command +' -out ' + str(nestdirname)
					#Get a result name for the output of transformix (assumes nifti for now)
					res_name = Path(os.path.join(str(nestdirname),"result"+in_im.suffix))

				#Add the transform parameters
				tmp_command = tmp_command + ' -tp ' + str(trans_calls[t])
				#Iterate through each transform parameter file and run transformix
				RunTransformix(tmp_command)


		else:
			#Just change the results to the output directory
			new_name = Path(os.path.join(str(out_dir),"result"+in_im.suffix))
			#Get the resulting image to rename (so we don't overwrite results)
			res_name.rename(new_name)
			#Set back the res name
			res_name = new_name

	#Return the result name
	return res_name


#Create class structure for transformix implementation
class Transformix():
    """Python implementation of Transformix with automated image pre-processing.

	Parameters
	----------
	in_im: string
		Input image path.

	out_dir: string
		Path to output directory for transform parameters and output results.

	tps: list (length number of transformation parameter files)
		Path to elastix image registration transform parameter files (in order of application).

    target_size: tuple of type integer (sizex,sizey; Default: None)
        Resize image using bilinear interpolation before exporting.

    pad: tuple of type integer (padx,pady; Default: None)
        Indicates height and length padding to add to the image before exporting.

    trim: tuple of type integer (trimx,trimy; Default: None)
        Indicates height and length trimming to remove from the image before exporting.

	crops: dictionary (Default: None)
		Dictionary of dictionaries containing {crop_name(s): {coords_csv: , target_size: , correction: , tp: , fixed_pad: }}.
		All arguments for each cropped region (i.e., region of interest) are performed
		only on the given crop.

			* coords_csv: string.
				Path to csv file containing coordinates.

			* target_size: tuple of type integer (sizex,sizey; Default: None)
				Resize image using bilinear interpolation before exporting.

			* correction: integer
				Extra pixels to take on each side of the crop contour for buffer.

			* tp: list (length number of transformation parameter files)
				Path to elastix image registration transform parameter files
				(in order of application) for this crop.

		    * fixed_pad: integer
				Indicates trim to remove from the cropped image before exporting. This should be the
				same value as the amount of padding applied to
				the cropped region's corresponding fixed image that was used for
				registration.
    """
                
    def __init__(
            self, 
            in_im, 
            out_dir, 
            tps, 
            target_size = None, 
            pad = None, 
            trim = None, 
            crops = None, 
            out_ext = ".nii"):
        
        #Create pathlib objects and set class parameters
        self.in_im = Path(in_im)
        self.in_channels = []
        self.out_channels = []
        self.multichannel = None
        self.out_dir = Path(out_dir)
        self.tps = [Path(t) for t in tps]
        self.command = "transformix"
        self.intermediate = False
        self.out_ext = out_ext
        self.target_size = target_size
        self.pad = pad
        self.trim = trim
        self.crops = crops
        self.out_name = None
		# Check for input list or none
        if self.target_size!=None:
            # convert to tuple and back to pathlib after logging
            if isinstance(self.target_size,str):
                self.target_size = literal_eval(self.target_size)
        			# convert it to tuple from list (command line parser)
                self.target_size = tuple(self.target_size)
		# Check for input list or none
        if self.pad!=None:
            # convert to tuple and back to pathlib after logging
            if isinstance(self.pad,str):
                self.self.pad = literal_eval(self.pad)
        			# convert it to tuple from list (command line parser)
                self.pad = tuple(self.pad)

		#Load images
        niiIn = _import.HDIreader(path_to_data=in_im,
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
		# update file information
        self.baseName = niiIn.hdi.data.filename.stem.replace(".ome","")
        self.ext = "".join(niiIn.hdi.data.filename.suffixes)

		# Check for input extension or none
        if self.out_ext==None:
			# convert it to be whatever extension the input image contains as default
            self.out_ext = self.ext
			# make sure that the export type is not .imzML for now...
            if self.out_ext==".imzML":
				# change to nifti
                self.out_ext=".nii"

		# check for number of channels in the image by accessing the class
        if len(niiIn.hdi.data.image.shape) >= 2:
			# Update multichannel class option
            self.multichannel = True
		# otherwise this is a single channel image
        else:
			# multichannel is false
            self.multichannel = False

	    #Check to see if there is single channel input (grayscale)
        if not self.multichannel:
			# run single channel registration on array
            self._singlechannelTransformixArray(niiIn)

		# Otherwise there is multichannel input. here we run multichannel registration
		# and all operations for preprocessing are performed on a per slice basis to
		# save disk space (dont have to export the full z stack at once). For now
		# all processing steps are tailored for nifti formats. In the future this
		# should easily be changed to allow for any data format
        else:
			# print update
            print('Detected multichannel input')
			# Check to see if cropping the resulting image
            if self.crops==None:
				# check for raster data
                if niiIn.hdi.data.hdi_type == 'raster':
					# raster transformix
                    self._multichannelTransformixRaster(niiIn)
				# check for image data type
                else:
					# run multichannel transformix with the array type
                    self._multichannelTransformixArray(niiIn)

			# !!!!!!!!!!!!!!!
			# Cropping is true
			# !!!!!!!!!!!!!!!
            else:
				#Read the image
                niiIn = niiIn.get_fdata()

				#Create a temporary outer directory
                with tempfile.TemporaryDirectory(dir=self.out_dir) as tmphomedir:

					#Create a dictionary to store the temporary folders in
                    tmpfolds = {}
					#Create a dictionary to store list of channels in for each roi
                    roi_channels = {}
					#Iterate through each crop and create a temporary directory for it
                    for roi, pars in crops.items():
						#Create a temporary directory and reserve the roi name from the crops input
                        tmpfolds.update({roi: tempfile.mkdtemp(dir = self.out_dir)})
						#Update the channels dictionary to be empty lists
                        roi_channels.update({roi:[]})

					#Print update on array shape
                    print('Image has ' +str(niiIn.shape[2])+' channels')

					#Iterate through each channel in the data
                    for i in range(niiIn.shape[2]):
						#Print update
                        print('Working on slice '+str(i))
						#print("Multichannel crop is true")
						#print('Shape of NiiIn is' +str(niiIn.shape[2]))
						#Update the list of names for image channels
						#self.in_channels.append(im_name)

						#create a temporary directory using the context manager for channel-wise images
                        with tempfile.TemporaryDirectory(dir=tmphomedir) as tmpdirname:
							#Create a name for a temporary image
                            im_name = Path(os.path.join(tmpdirname,self.in_im.stem+str(i)+self.in_im.suffix))
							#Check to see if the path exists
                            if not im_name.is_file():
								#Check to see if there is a target size for the image
                                if target_size != None:
									#print("Got the resize")
									#Create a nifti image from this slice
                                    nii_im = nib.Nifti1Image(resize(niiIn[:,:,i],target_size), affine=np.eye(4))
									#print(" resized")
								#No resize
                                else:
									#Leave the image unchanged
                                    nii_im = nib.Nifti1Image(niiIn[:,:,i], affine=np.eye(4))
								#Save the nifti image
                                nib.save(nii_im,str(im_name))
								#print("saved "+ str(im_name))
								#Remove the nifti slice to clear memory
                                nii_im = None

							#add transform -- check for list size
                            if len(self.tps) > 1:
								#print('GOt the multi tp input')
								#Run the composition function for transformix
                                res_name = MultiTransformix(in_im = im_name, out_dir = tmpdirname, tps = self.tps)
                                
                            else:

								#Create a temporary command to be sent to the shell
                                tmp_command = self.command + ' -in ' + str(im_name) + ' -out ' + str(tmpdirname)
								#Add full tissue transform paramaeters
                                tmp_command = tmp_command + ' -tp ' + str(self.tps[0])
								#Send the command to the shell
                                RunTransformix(tmp_command)

								#Get a temporary result name for the output of transformix (assumes nifti for now)
                                res_name = Path(os.path.join(tmpdirname,"result"+self.in_im.suffix))

							#Load the resulting transformed image
                            niiResult = nib.load(str(res_name)).get_fdata()

							#Iterate through each of the crops
                            for roi, pars in crops.items():

								#Create a temporary directory to store the
                                with tempfile.TemporaryDirectory(dir=tmpdirname) as roitmp:
									#Create a temporary name
                                    tmproiname = Path(os.path.join(roitmp,str(roi)+str(i)+self.in_im.suffix))
									#Extract a temporary crop for this region
                                    tmpcrop = ExtractROIcrop(coords_csv = pars['coords_csv'],full_img = niiResult, target_size = pars['target_size'], correction = pars['correction'])
									#Save this temporary crop to the roi temp directory
                                    nib.save(tmpcrop,str(tmproiname))

									#add transform -- check for list size
                                    if len(pars['tps']) > 1:
										#Run the composition function for transformix
                                        roires_name = MultiTransformix(in_im = tmproiname, out_dir = roitmp, tps = pars['tps'])
                                        
                                    else:

										#Create a temporary command to be sent to the shell -- output directory in the tmp folder created previously
                                        roi_command = self.command + ' -in ' + str(tmproiname) + ' -out ' + str(roitmp)
										#Add these transform parameters for this roi
                                        roi_command = roi_command + ' -tp ' + str(Path(pars['tp']))
										#Send the command to the shell
                                        RunTransformix(roi_command)

										#Get a temporary result name for the output of transformix (assumes nifti for now)
                                        roires_name = Path(os.path.join(roitmp,"result"+self.in_im.suffix))

									#Create a new name for the roi result
                                    roinew_name = Path(os.path.join(tmpfolds[roi],str(roi)+'_result'+str(i)+self.in_im.suffix))
									#Rename the produced file :)
                                    roires_name.rename(roinew_name)
									#Update the list of output channel names
                                    roi_channels[str(roi)].append(str(roinew_name))

									#Clear the nifti objects
                                    tmpcrop = None

							#Clear the nifti full tissue result
                            niiResult = None


					#Access each temporary roi results folder and concatenate the results
                    for roi, pars in crops.items():
						#Concatenate the output channels into a single result file in the output directory
                        roi_results = nib.concat_images([str(i) for i in roi_channels[str(roi)]])
						#Check to see if there is fixed image padding
                        if pars['fixed_pad'] != None:
							#Get the value of the padding
                            pads = pars['fixed_pad']
							#Extract only the needed region
                            roi_results = roi_results.get_fdata()[pads:-pads,pads:-pads,:]
						#create a filename for the full nifti results
                        roi_name = Path(os.path.join(str(self.out_dir),str(roi)+"_result"+self.out_ext))
						# use the exported from hdiutils
                        _export.HDIexporter(full_result.get_fdata().transpose(1,0,2),roi_name)

						#Remove roi results from memory
                        roi_results = None

						#Now delete the temporary folder stored for the single channel ROI results
                        shutil.rmtree(tmpfolds[roi], ignore_errors=True)

		# remove the temporary image if there was a nifti-1 intermediate
        if self.intermediate:
			# remove using pathlib
            self.in_im.unlink()
		#Print update
        print("Finished")

		#Return the command
		#return self.command
        
    def _singlechannelTransformixArray(self, niiIn):
        """Helper function for single channel transformix call on array data.

		Parameters
		----------
		niiIn: class HDIreader
			Input image that is read initially with the hdiutils python package.
			The details of the image are stored inside of this class object,
			and the ``Transformix`` class will access those to automatically
			process the image.
		"""
		# here we will get the extension of the image and will convert it to the nift-1
		# format if it is not already in that format. While users can supply their own
		# nifti formatted image to the pipeline, this ensures that other file formats
		# can be used, although, it creates additionally overhead
		# here we supply all preprocessing commands that were used to preprocess or morph
		# the array size of the input image through the hdiprep workflow. Transformix
		# must be run on images with the same size as the elastix registration
        if ((self.out_ext!=".nii") or (self.target_size!=None) or (self.pad!=None)):
			# get the shape of the image
            shp = len(niiIn.hdi.data.image_shape)
			# create new name for the temporary image
            tmp_nm = os.path.join(self.out_dir, next(tempfile._get_candidate_names())+".nii")
			# export nifti intermediate
            print('Creating nifti-1 intermediate for registration')
			# check for padding
            if self.pad!=None:
				# pad the single-channel
                niiIn.hdi.data.image = np.pad(niiIn.hdi.data.image,[(self.pad[0], self.pad[0]), (self.pad[1], self.pad[1])],mode='constant')
			# check for image resizing
            if (self.target_size != None) and (self.crops==None):
				# transform the image
                niiIn.hdi.data.image = resize(niiIn.hdi.data.image,self.target_size)

			# Create nifti oject -- transpose axes because of the transformation!
            nii_im = nib.Nifti1Image(niiIn.hdi.data.image.T, affine=np.eye(4))
			#Save the nifti image
            nib.save(nii_im,str(tmp_nm))
			# remove the nifit memory
            nii_im = None
			# update the image name
            print('Using nifti-1 intermediate for registration')
			# update the input image
            self.in_im = Path(tmp_nm)
			# update the intermediate flag
            self.intermediate = True
			#Remove loaded image to clear memory
            niiIn = None

		#Print update
        print('Detected single channel input images...')
		#Update the fixed channels
        self.in_channels.append(self.in_im)

		#add transform -- check for list size
        if len(self.tps) > 1:
			#Run the composition function for transformix
            res_name = MultiTransformix(in_im = self.in_im, out_dir = self.out_dir, tps = self.tps)

		#Otherwise only use the first transform parameter
        else:
			#Updatethe command with the single channel path alone
            self.command = self.command + ' -in ' + str(self.in_im)
			#use the first transform parameter file
            self.command = self.command + ' -tp ' + str(self.tps[0])
			#Update the command with the output directory
            self.command = self.command + ' -out ' + str(self.out_dir)
			#Run single channel transformix without temporary directories
            RunTransformix(self.command)
			#Get a result name for the output of transformix (assumes nifti for now)
            res_name = Path(os.path.join(self.out_dir,"result"+self.in_im.suffix))

		#Create a new name
        new_name = Path(os.path.join(self.out_dir,self.baseName+'_result'+self.out_ext))
        # add the new name to the class object
        self.out_name = new_name

		# check if the output format needs to be switched -- set by the user
        if (self.out_ext!=".nii") or (self.trim!=None):
			# use HDIreader for now to parse image and exporter to export
            niiIn = _import.HDIreader(
			    path_to_data=self.in_im,
			    path_to_markers=None,
			    flatten=False,
			    subsample=None,
			    mask=None,
			    save_mem=False
			)
			# check the trim
            if self.trim!=None:
				# trim the image borders
                niiIn.hdi.data.image = niiIn.hdi.data.image[self.trim:-self.trim,self.trim:-self.trim]
			# export new data
            _export.HDIexporter(niiIn.hdi.data.image,new_name)
        else:
			# simply rename the file that is already in the nifti format
            res_name.rename(new_name)
            
    def _multichannelTransformixArray(self, niiIn):
        """Helper function for running multichannel transformix on array data.

		Parameters
		----------
		niiIn: class HDIreader
			Input image that is read initially with the hdiutils python package.
			The details of the image are stored inside of this class object,
			and the ``Transformix`` class will access those to automatically
			process the image.
		"""
		# create a temporary directory using the context manager for channel-wise images
        with tempfile.TemporaryDirectory(dir=self.out_dir) as tmpdirname:
			# Print update
            print('Created temporary directory', tmpdirname)
			# Iterate through the channels
            for i in tqdm(range(niiIn.hdi.data.image.shape[2])):
				# Print update
				# print('Working on slice '+str(i))
				# Create a name for a temporary image
                im_name = Path(os.path.join(tmpdirname,self.in_im.stem+str(i)+".nii"))
				# Update the list of names for image channels
                self.in_channels.append(im_name)
				# set a temporary channel to work with throughout the data prep stage
                slice_in = niiIn.hdi.data.image[:,:,i]

				# Check to see if the path exists
                if not im_name.is_file():
					# check for padding
                    if self.pad!=None:
						# pad the single-channel
                        slice_in = np.pad(slice_in,[(self.pad[0], self.pad[0]), (self.pad[1], self.pad[1])],mode='constant')
					# Check to see if there is a target size for the image
                    if self.target_size!=None:
						# Resize the image
                        slice_in = resize(slice_in,self.target_size)

					# Create a nifti image from this slice
                    nii_im = nib.Nifti1Image(slice_in.T, affine=np.eye(4))
					# remove memory
                    slice_in = None
					#Save the nifti image
                    nib.save(nii_im,str(im_name))
					#Remove the nifti slice to clear memory
                    nii_im = None

				#add transform -- check for list size
                if len(self.tps) > 1:
					#Run the composition function for transformix
                    res_name = MultiTransformix(in_im = im_name, out_dir = tmpdirname, tps = self.tps)
                    
                else:
					#Create a temporary command to be sent to the shell
                    tmp_command = self.command + ' -in ' + str(im_name) + ' -out ' + str(tmpdirname)
					#Add full tissue transform paramaeters
                    tmp_command = tmp_command + ' -tp ' + str(self.tps[0])
					#Send the command to the shell
                    RunTransformix(tmp_command)

					#Get a temporary result name for the output of transformix (assumes nifti for now)
                    res_name = Path(os.path.join(tmpdirname,"result"+".nii"))

				# remove the temporary image if there was a nifti-1 intermediate
                im_name.unlink()

				#Create a new name
                new_name = Path(os.path.join(tmpdirname,self.in_im.stem+str(i)+'_result'+".nii"))
				#Get the resulting image to rename (so we don't overwrite results)
                res_name.rename(new_name)
				#Update the list of output channel names
                self.out_channels.append(new_name)

			#Remove loaded image to clear memory
            niiIn = None
			#Concatenate the output channels into a single result file in the output directory
            full_result = nib.concat_images([str(i) for i in self.out_channels])
			#create a filename for the full nifti results
            full_name = Path(os.path.join(self.out_dir,self.baseName+"_result"+self.out_ext))
            # add the new name to the class object
            self.out_name = full_name

			# check if the output format needs to be switched -- set by the user
            if (self.out_ext!=".nii") or (self.trim!=None):
				# check the trim
                if self.trim!=None:
					# trim the image borders
                    full_result = full_result.get_fdata()[self.trim:-self.trim,self.trim:-self.trim,:]
					# export new data
                    _export.HDIexporter(full_result.transpose(1,0,2),full_name)
                else:
					# export the non trimmed image
                    _export.HDIexporter(full_result.get_fdata().transpose(1,0,2),full_name)
                    
            else:
				# export new data using the aggregated nifti objects
				# doesnt need to be formally read in because it is memory
				# mapped to the full_result object
                _export.HDIexporter(full_result.get_fdata().transpose(1,0,2),full_name)
                
    def _multichannelTransformixRaster(self, niiIn):
        """Helper function for running multichannel transformix on raster data.

		Parameters
		----------
		niiIn: class HDIreader
			Input image that is read initially with the hdiutils python package.
			The details of the image are stored inside of this class object,
			and the ``Transformix`` class will access those to automatically
			process the image.
		"""
		# create a temporary directory using the context manager for channel-wise images
        with tempfile.TemporaryDirectory(dir=self.out_dir) as tmpdirname:
			# Print update
            print('Created temporary directory', tmpdirname)
			# Iterate through the channels
            for i in tqdm(range(niiIn.hdi.data.image.shape)):
				# Print update
				# print('Working on slice '+str(i))
				# Create a name for a temporary image
                im_name = Path(os.path.join(tmpdirname,self.in_im.stem+str(i)+".nii"))
				# Update the list of names for image channels
                self.in_channels.append(im_name)
				# set a temporary channel to work with throughout the data prep stage
                slice_in = niiIn.hdi.CreateSingleChannelArray(i)

				# Check to see if the path exists
                if not im_name.is_file():
					# check for padding
                    if self.pad!=None:
						# pad the single-channel
                        slice_in = np.pad(slice_in,[(self.pad[0], self.pad[0]), (self.pad[1], self.pad[1])],mode='constant')
					# Check to see if there is a target size for the image
                    if self.target_size!=None:
						# Resize the image
                        slice_in = resize(slice_in,self.target_size)

					# Create a nifti image from this slice
                    nii_im = nib.Nifti1Image(slice_in.T, affine=np.eye(4))
					# remove memory
                    slice_in = None
					#Save the nifti image
                    nib.save(nii_im,str(im_name))
					#Remove the nifti slice to clear memory
                    nii_im = None

				#add transform -- check for list size
                if len(self.tps) > 1:
					#Run the composition function for transformix
                    res_name = MultiTransformix(in_im = im_name, out_dir = tmpdirname, tps = self.tps)
                    
                else:
					#Create a temporary command to be sent to the shell
                    tmp_command = self.command + ' -in ' + str(im_name) + ' -out ' + str(tmpdirname)
					#Add full tissue transform paramaeters
                    tmp_command = tmp_command + ' -tp ' + str(self.tps[0])
					#Send the command to the shell
                    RunTransformix(tmp_command)

					#Get a temporary result name for the output of transformix (assumes nifti for now)
                    res_name = Path(os.path.join(tmpdirname,"result"+".nii"))

				# remove the temporary image if there was a nifti-1 intermediate
                im_name.unlink()

				#Create a new name
                new_name = Path(os.path.join(tmpdirname,self.in_im.stem+str(i)+'_result'+".nii"))
				#Get the resulting image to rename (so we don't overwrite results)
                res_name.rename(new_name)
				#Update the list of output channel names
                self.out_channels.append(new_name)

			#Remove loaded image to clear memory
            niiIn = None
			#Concatenate the output channels into a single result file in the output directory
            full_result = nib.concat_images([str(i) for i in self.out_channels])
			#create a filename for the full nifti results
            full_name = Path(os.path.join(self.out_dir,self.baseName+"_result"+self.out_ext))
            # add the new name to the class object
            self.out_name = full_name

			# check if the output format needs to be switched -- set by the user
            if (self.out_ext!=".nii") or (self.trim!=None):
				# check the trim
                if self.trim!=None:
					# trim the image borders
                    full_result = full_result.get_fdata()[self.trim:-self.trim,self.trim:-self.trim,:]
					# export new data
                    _export.HDIexporter(full_result.transpose(1,0,2),full_name)
                else:
					# export the non trimmed image
                    _export.HDIexporter(full_result.get_fdata().transpose(1,0,2),full_name)
            else:
				# export new data using the aggregated nifti objects
				# doesnt need to be formally read in because it is memory
				# mapped to the full_result object
                _export.HDIexporter(full_result.get_fdata().transpose(1,0,2),full_name)




#
