#Image Registration Functions
#Joshua Hess

import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import os
import re
import nibabel as nib
from pathlib import Path
import pandas as pd
import shutil
import glob
import fileinput
import cv2
from joblib import Parallel, delayed
from numba import njit, prange
import subprocess

from miaaim.reg import _utils


#-----------------General Utility Functions for Registration-----------------------

def FormatFijiPointsFromCSV(input_file,selection_type):
    """This function will take the csv point selection file that you export from
    using control+m in fiji (ImageJ) to the correct text file format to use with
    elastix image registration

    selection_type must be the string 'index' or 'points'"""
    #Get the current directory
    tmp_path = Path('..')
    home_dir=tmp_path.cwd()
    #Read the csv file that you input
    data = pd.read_csv(input_file)
    #Get the image folder name
    parent=Path(input_file).parent
    #Change the directory to the parent folder so we know where to export
    os.chdir(parent)
    #Remove the file extension
    prefix=Path(input_file).stem
    #Rename the first column for ease of access
    data.rename(columns = {list(data)[0]:'Landmark'}, inplace=True)
    #Create a new text file
    txt_file = open(prefix+".txt","w+")
    #Create first string to write to your file
    str_list = [str(selection_type),str(max(data['Landmark']))]
    txt_file.writelines(i + '\n' for i in str_list)
    #Close and save the txt file
    txt_file.close()
    #Get only the data we need for the text file
    point_tab = data[['X','Y']]
    #Now append the data table to the txt file
    point_tab.to_csv(prefix+".txt", header=False, index=False, sep=' ', mode='a')
    #Change the current directory back to the home directory
    os.chdir(home_dir)

def FormatRegionFromFiji(input_file,selection_type="index"):
    """This function will take a set of points that you set and export in ImageJ
    using Analyze > Tools > Save XY coordinates plugin"""
    #Get the current directory
    tmp_path = Path('..')
    home_dir=tmp_path.cwd()
    #Read the csv file that you input
    data = pd.read_csv(input_file)
    #Get the image folder name
    parent=Path(input_file).parent
    #Change the directory to the parent folder so we know where to export
    os.chdir(parent)
    #Remove the file extension
    prefix=Path(input_file).stem
    #Create a new text file
    txt_file = open(prefix+".txt","w+")
    #Create first string to write to your file
    str_list = [str(selection_type),str(data.shape[0])]
    txt_file.writelines(i + '\n' for i in str_list)
    #Close and save the txt file
    txt_file.close()
    #Get only the data we need for the text file
    point_tab = data[['X','Y']]
    #Now append the data table to the txt file
    point_tab.to_csv(prefix+".txt", header=False, index=False, sep=' ', mode='a')
    #Change the current directory back to the home directory
    os.chdir(home_dir)


def GetROImask(dir,full_size,ROI_correction = None):
	"""This function will take the check all subdirectories
	for csv files exported from using Analyze > Tools > Save XY coordinates plugin. It will
	then create a mask for each of those ROIs based on the full image coordinates so that
	each region can be extracted efficiently.

	dir: The directory that you want to do a subdirectory search on for csv files.

	full_image: Full image to use for masking ROI from

	ROI_correction: correction to be added to the dimensions of the ROI

	Returns a dictionary object with keys being the folders detected and the values
	being the masks created based on csv files in folders and the full image

	Note: We will use this function with ROIs defined on the toluidine blue image and we use
	it to extract the ROIs from the final registered full MSI UMAP image """

	#Get the names of the folders in your directory
	filenames = [file for file in os.listdir(dir) if os.path.isdir(os.path.join(dir,file))]
	#Get all csv files in those directories
	#***Note that we are traversing all directory so may need to change to avoid prblems***
	csv_files = _utils.TraverseDir(ending="coordinates.csv")
	#Create a dictionary object for our ROIs that contains the folder names with csv full paths
	ROI_dict = dict(zip(filenames, csv_files))

	#Create a new dictionary object to store the masks and coordinates in
	ROI_masks = {}
	#Access the dictionary ROI object to load the mask coordinates from csv files
	for ROI, filename in ROI_dict.items():
		#Read csv
		file_cont = pd.read_csv(filename)
		#Get the XY coordinates from the dataframe
		point_tab = file_cont[['X','Y']]
		#Create a mask for each of these ROI regions
		tp_left = (point_tab[['X']].values.min()-1,point_tab[['Y']].values.min()-1)
		bt_right = (point_tab[['X']].values.max()-1,point_tab[['Y']].values.max()-1)

		#Create a blank mask - ***Order is 1 and 0 for the nifti image format*****
		mask = np.zeros(shape=(full_size[1],full_size[0]),dtype="uint8")
		#Draw rectangle on the mask using opencv
		cv2.rectangle(img=mask,pt1=tp_left,pt2=bt_right,color=(255,255,255),thickness=-1)
		#Extract sliced, nonzero regions from your original image
		nonzero = np.nonzero(mask)
		minx = min(nonzero[0])
		maxx = max(nonzero[0])
		miny = min(nonzero[1])
		maxy = max(nonzero[1])
		#Check to see if we are adding overlap to our ROIs
		if ROI_correction is not None:
			print('Detected ROI Correction Factor...')
			minx = (minx-(ROI_correction))
			maxx = (maxx+(ROI_correction))
			miny = (miny-(ROI_correction))
			maxy = (maxy+(ROI_correction))
		#Add the mask in this iteration to our dictionary object
		ROI_masks.update({str(ROI):[minx,maxx,miny,maxy]})
	#Report that the masking is complete
	print('Finished getting coordinates')
	#Return the dictionary object
	return ROI_masks


def CreateMaskFromCoords(filenames_list,full_image,invert=False):
	"""Function for reading in ROI coordinates and converting them to a mask.

	Currently only accepts nifti format for the full_image. Option to invert the mask"""

	#Read the full image
	full_im = nib.load(str(full_image)).get_fdata().T
	#Get the full array size
	full_size=full_im.shape

	#Create a blank mask - ***Order is 1 and 0 for the nifti image format*****
	mask = np.zeros(shape=(full_size[0],full_size[1]),dtype="uint8")

	#Iterate through each ROI csv file
	for roi in filenames_list:
		#Read csv
		file_cont = pd.read_csv(roi)
		#Get the XY coordinates from the dataframe
		point_tab = file_cont[['X','Y']]
		#Create a mask for each of these ROI regions
		tp_left = (point_tab[['X']].values.min()-1,point_tab[['Y']].values.min()-1)
		bt_right = (point_tab[['X']].values.max()-1,point_tab[['Y']].values.max()-1)

		#Draw rectangle on the mask using opencv
		cv2.rectangle(img=mask,pt1=tp_left,pt2=bt_right,color=(255,255,255),thickness=-1)

	#Check to see if inverting the mask
	if invert:
		mask = ~mask

	#return the mask
	return mask



def ROImaskExport(ROI_masks,full_img,flip_horz=False,flip_vert=False,prefix=None,export_image=True):
    """Function for exporting ROIs from an ROI_masks dictionary object that gets returned
    from GetROImask function

    ROI_masks: Returned dictionary from GetROImask function

    full_img: Array that contains the image to be cropped from

    flip_horz: Logical. If true, ROI is flipped horizontally

    flip_vert: Logical. If true, ROI is flipped vertically

    prefix: prefix to add to the ROIs that are exported. If left to None,
    then a default prefix is added that corresponds to the image folder that the ROI represents"""

    #Set your home directory
    tmp = Path('..')
    home_dir=tmp.cwd()

    #Create a dictionary of final rescaled sizes to use for MSI ROI Extraction
    fin_sizes = {}
    #Loop through the ROI_masks dictionary
    for ROI,mask in ROI_masks.items():
        os.chdir(str(ROI))
        #Read the imc image for resizing
        imc_im=mms.imc_file(tmp.cwd(),return_im=True,return_imc_cell_table=False,return_pix_table=False).image
        #Get our ROIs
        tmp_ROI = full_img[int(mask[2]):int(mask[3]),int(mask[0]):int(mask[1])]
        #Save the new ROI
        if flip_horz:
            tmp_ROI = np.flip(tmp_ROI,0)
        if flip_vert:
            tmp_ROI = np.flip(tmp_ROI,1)
            #Get the size of the imc image
        multiple = int(imc_im.shape[1]/tmp_ROI.shape[0])
        #Use the rounded multiple to resize our ROI for image registration
        HiRes_size = (tmp_ROI.shape[1]*multiple,tmp_ROI.shape[0]*multiple)
        #Remember that cv2 does the axis in the opposite order as numpy
        tmp_ROI_HR = cv2.resize(tmp_ROI,HiRes_size)
        #Create a nifti object and save the image
        nifti_ROI = nib.Nifti1Image(tmp_ROI_HR, affine=np.eye(4))
        #Get a prefix for the image that we are saving
        if prefix is None:
            prefix_tmp = str(ROI)
        else:
            prefix = prefix

        if export_image:
            #Save the ROI
            nib.save(nifti_ROI,prefix_tmp+'.nii')
            #Report the finished job
            print('Finished exporting '+str(ROI)+'...')
        #Add the final size to our dictionary
        fin_sizes.update({str(ROI):HiRes_size})
        #Change back to the original directory
        os.chdir(home_dir)
    #Report the finished export job
    print('Finsihed export all regions')
    #Return an object that contains the final size for each H&E ROI so we can apply the
    #transformix function for each MSI ROI
    return fin_sizes

def ROImaskExport_MSI_Transformix(ROI_masks_from_tolBlue,final_sizes,parameter_files,full_img,flip_horz=False,flip_vert=False,prefix=None):
    """Function for exporting ROIs from an ROI_masks dictionary object that gets returned
    from GetROImask function

    ROI_masks: Returned dictionary from GetROImask function

    full_img: Array that contains the image to be cropped from

    flip_horz: Logical. If true, ROI is flipped horizontally

    flip_vert: Logical. If true, ROI is flipped vertically

    prefix: prefix to add to the ROIs that are exported. If left to None,
    then a default prefix is added that corresponds to the image folder that the ROI represents"""

    #Set your home directory
    tmp = Path('..')
    home_dir=tmp.cwd()

    #Loop through the ROI_masks dictionary
    for ROI,mask in ROI_masks_from_tolBlue.items():
        #Create a directory for our ROI and switch to it
        if not os.path.exists(os.path.join(tmp.cwd(),str(ROI))):
            os.makedirs(str(ROI))
        os.chdir(str(ROI))
        #Get our ROIs
        tmp_ROI = full_img[int(mask[2]):int(mask[3]),int(mask[0]):int(mask[1])]
        #Save the new ROI
        if flip_horz:
            tmp_ROI = np.flip(tmp_ROI,0)
        if flip_vert:
            tmp_ROI = np.flip(tmp_ROI,1)

        #Use the dictionary of ROI final sizes to resize the image
        HiRes_size = final_sizes[str(ROI)]
        #Resize the image
        tmp_ROI_HR = cv2.resize(tmp_ROI,HiRes_size)
        #Create a nifti object and save the image
        nifti_ROI = nib.Nifti1Image(tmp_ROI_HR, affine=np.eye(4))
        #Get a prefix for the image that we are saving (will come from the loop that indicates m/z slice)
        prefix = prefix
        #Save the ROI
        nib.save(nifti_ROI,prefix+'.nii')
        #Report the finished job
        print('Finished exporting original crop for '+str(ROI)+'...')

        #Run transformix on this particular ROI using the false H&E and H&E registration parameters
        tmp_imagepath = Path(os.path.join(tmp.cwd(),str(prefix)+'.nii'))
        #Create a directory to store this slice in within each ROIs folder
        os.mkdir(str(prefix))
        #Create an output directory path (here tmp.cwd() is the ROI folder and prefix is m/z slice number)
        out_dir = Path(os.path.join(tmp.cwd(),str(prefix)))
        #Get the parameter file from this ROI (user will need to provide the paths for right now)
        par = parameter_files[str(ROI)]
        #Check to see if we are transforming this slice
        if par is not None:
            #Send the command to the shell
            print('Running Transformix for  '+str(ROI)+' '+str(prefix)+'...')
            os.system("transformix -in "+str(tmp_imagepath)+" -out "+str(out_dir)+" -tp "+str(par))
            #Delete the temporary m/z image that comes from exporting ROI from composition MSI slice
            os.remove(tmp_imagepath)

        #Change back to the original directory (outside of the ROI specific folder)
        os.chdir(home_dir)
    #Report the finished export job
    print('Finsihed export all regions for '+str(prefix))

#######----******Add in optional extra registration alternative to ROImaskExport for detecting transformation file in the folder




def GetParamFileImageSize(par):
	"""Function for parsing an elastix parameter file to get the size tuple for the registration.
	The size corresponds to the final size of the registered image.

	par: Path to the parameter file that you want to parse"""

	#Read the parameter file to pandas dataframe
	par_file = pd.read_csv(par, sep = "\n")
	#Get those rows that contain the string "Size"
	size_cols = par_file[par_file.iloc[:,0].str.contains("Size")]
	#Replace the right parentheses
	tmp_tab = size_cols.iloc[:,0].str.replace('(','')
	#Replace the left parentheses
	tmp_tab = pd.DataFrame(tmp_tab.iloc[:].str.replace(')',''))
	#String match again for "Size" and convert that to a string
	size_cols_fin = tmp_tab[tmp_tab.iloc[:,0].str.match("Size")].to_string(header=False,index=False)
	#Extract the positive integers from the final string and convert to tuple
	size_tup = tuple([int(s) for s in size_cols_fin.split() if s.isdigit()])
	#Return the final size tuple
	return size_tup




#-----------------General Registration Functions-----------------------

def ElastixRegistration(fixed,moving,out_dir,p0,p1=None,fp=None,mp=None,fMask=None,mkdir=False):
    """This is a python function for running the elastix image registration
    toolbox. You must be able to call elastix from your command shell to use this.
    You must also have your parameter text files set before running this through
    python."""

    #Start a timer
    start = time.time()

    #Get the name of the output directory
    out_dir = Path(out_dir)
    #Get the name of the parameter file
    p0 = Path(p0)
    #Get the names of the input images
    fixedName = Path(fixed)
    movingName = Path(moving)
    #Get the number of channels for the fixed and moving images
    niiFixed = nib.load(str(fixedName))
    niiMoving = nib.load(str(movingName))

    #Check to see if there is single channel input
    if niiFixed.ndim is 2 and niiMoving.ndim is 2:
        print('Detected single channel input images...')
        #Send the command to shell to run elastix
        command = "elastix -f "+str(fixed)+ " -m "+str(moving)
    #Check to see if there is multichannel input
    else:
        print('Exporting single channel images for multichannel input...')
        #Read the images
        niiFixed_im = niiFixed.get_fdata()
        niiMoving_im = niiMoving.get_fdata()
        #Set up list of names for the images
        fixedList = []
        movingList = []
        command = "elastix"
        #Export single channel images for each channel
        for i in range(niiFixed.shape[2]):
            #Create a filename
            fname = Path(os.path.join(fixedName.parent,str(fixedName.stem+str(i)+fixedName.suffix)))
            #Update the list of names for fixed image
            fixedList.append(fname)
            #Update the list of names for fixed image
            command = command + ' -f' + str(i) + ' ' + str(fname)
            #Create a nifti image
            #Check to see if the path exists
            if not fname.is_file():
                #Create a nifti image
                nii_im = nib.Nifti1Image(niiFixed_im[:,:,i], affine=np.eye(4))
                nib.save(nii_im,str(fname))
        for i in range(niiMoving.shape[2]):
            #Create a filename
            mname = Path(os.path.join(movingName.parent,str(movingName.stem+str(i)+movingName.suffix)))
            #Update the list of names for moving image
            movingList.append(mname)
            #Update the list of names for moving image
            command = command + ' -m' + str(i) + ' ' + str(mname)
            #Check to see if the path exists
            if not mname.is_file():
                #Create a nifti image
                nii_im = nib.Nifti1Image(niiMoving_im[:,:,i], affine=np.eye(4))
                nib.save(nii_im,str(mname))

    #Add the parameter files
    command = command+" -p "+str(p0)
    #Check for additional files
    if p1 is not None:
        #Create pathlib Path
        p1 = Path(p1)
        command = command + " -p "+str(p1)

    #Check for corresponding points in registration
    if fp and mp is not None:
        #Create pathlib Paths
        fp = Path(fp)
        mp = Path(mp)
        command = command +" -fp "+str(fp)+" -mp "+str(mp)

    #Check for fixed mask
    if fMask is not None:
        #Create pathlib Paths
        fMask = Path(fMask)
        command = command +" -fMask "+str(fMask)

    #Check for making new directories
    if mkdir is True:
        n=0
        while n>=0:
            tmp_name = "elastix"+str(n)
            if not os.path.exists(Path(os.path.join(out_dir,tmp_name))):
                os.mkdir(Path(os.path.join(out_dir,tmp_name)))
                out_dir = Path(os.path.join(out_dir,tmp_name))
                break
            n+=1
    #Add the output directory to the command
    command = command +" -out "+str(out_dir)

    #Send the command to the shell
    print('Running elastix...')
    os.system(command)
    end = time.time()
    print('Initial registration finished\n'+'Computation time: '+str(end-start)+' sec.')

    #Return values
    return command




def TransformixRegistration(input_img,output_dir,parameter_file,conc=False,points=None):
    """This is a python function for running transformix image registration. Again,
    must be able to call elastix from the command line to be able to run this.

    If your channel is multichannel, this script must first export each channel
    in the nifti format. By default, images will be exported to your output_dir
    and will include the suffix 'Unregistered' followed by the channel number"""
    #Create a timer
    trans_start = time.time()
    #Check to see if the image is multichannel or grayscale (note: need to remove " in your filename)
    tmp_data = nib.load(str(input_img)).get_data()
    if tmp_data.ndim is not 2:
        print('Detected multichannel image. Creating channel images...')
        #Get the current directory
        tmp_path = Path('..')
        home_dir=tmp_path.cwd()
        #Set the working directory as the output directory
        parent=Path(output_dir)
        os.chdir(parent)
        #Now take each of the channels and export a separate image for registration
        filenames_channels = []
        for i in range(tmp_data.shape[2]):
            #Get the image name from your input image path
            im_name=Path(tmp_data.filename).parts[-1]
            #Remove the file extension
            prefix,extension=Path(im_name).stem,Path(im_name).suffix
            #Create new image channel i in each iteration
            nifti_col = nib.Nifti1Image(tmp_data[:,:,i], affine=np.eye(4))
            #Create the image path for this iteration
            tmp_image=prefix+"_Unregistered"+str(i)+extension
            #Save the nifti image
            print("Saving a temporary image for channel "+str(i)+"...")
            nib.save(nifti_col,str(tmp_image))
            #Now load the image and run transformix on that channel in the shell
            print("Running Transformix for channel "+str(i)+"...")
            #Creat a new file for your transformix results
            transformix_path = Path(os.path.join(str(output_dir),str(prefix)+"_Transformix_Registered"+str(i)))
            transformix_path.mkdir()
            os.system("transformix -in " +str(tmp_image)+ " -out "+str(transformix_path)+" -tp "+str(parameter_file))
            print("Finished Transforming Channel "+str(i))
            #add filenames to the list
            filenames_channels.append(os.path.join(str(transformix_path),"result.nii"))
        #Check to see if we are concatenating images
        if conc is True:
            tmp_nii = nib.concat_images(filenames_channels)
            #Create a path and save the image
            conc_path = Path(os.path.join(str(output_dir),str(prefix)+"_Transformix_Registered"))
            conc_path.mkdir()
            os.chdir(conc_path)
            nib.save(tmp_nii,"result.nii")

            #Create a return path
            ret_path = conc_path

        #Set working directory back to its original
        os.chdir(home_dir)
    else:
        print("Single channel image detected...")
        print("Running Transformix...")
        #Send the command to the shell to run transformix
        os.system("transformix -in " +str(input_img)+ " -out "+str(output_dir)+" -tp "+str(parameter_file))
        trans_stop = time.time()
        print('Finished transforming\n'+'Transformix Computation Time: '+str(trans_stop-trans_start)+' sec.')

        #Create a return path
        ret_path = output_dir

    return ret_path


def ApplyTransformixDir(dir,ending,remove_from_names=None,**kwargs):
    """Function for applying a transformix registration for all images contained
    within a folder (indicated by dir parameter)"""

    #Set a path object
    tmp = Path('..')

    #Set the home directory and record its position
    home = Path(dir)
    os.chdir(home)
    #Get the names of all nifti images in the directory
    im_names = _utils.SearchDir(ending = ending)
    #Run the the images in the directory and export their transformix
    for im in im_names:
        #Get the name of the image stem
        im_stem = im.stem
        #Check to see if removing a string from image name
        if remove_from_names is not None:
            #Remove the nonsense from the image name
            im_stem = im_stem.replace(remove_from_names,"")
        #Run transformix on this iteration image
        ret_path = TransformixRegistration(input_img=im,output_dir=tmp.cwd(),**kwargs)
        #Rename the image in this iteration
        trans_im = Path(os.path.join(ret_path,"result.nii"))
        #Rename this file to the correct ROI for dice score
        trans_im.replace(trans_im.with_name(im_stem+"_result.nii"))
        trans_im = trans_im.with_name(im_stem+"_result.nii")
        #Change back to the home directory
        os.chdir(home)



def AlphaMIParameterFileGrid(template,num_neighbors,error,alpha,out):
    """Function for creating parameter file grid for alpha mutual information
    based registration"""

    #Get the current directory
    home_dir = Path(os.getcwd())
    #Get the parent directory and filename
    out = Path(out)
    os.chdir(out)
    #Get the current k value
    k_val,_ = _utils.ParseElastix(template,'KNearestNeighbours')
    #Get the current error value
    err_bound,_ = _utils.ParseElastix(template,'ErrorBound')
    #Get the current alpha value
    alph,_ = _utils.ParseElastix(template,'Alpha')
    #Set up a list for all new parameter file names
    nms_list = []
    #Iterate through the options for k,error,and alpha
    for nn in num_neighbors:
        for e in error:
            for a in alpha:
                #Read the transform parameters
                with open(template, 'r') as file:
                    filedata = file.read()
                #Replace the Alpha value
                filedata = filedata.replace("(KNearestNeighbours "+k_val+")", "(KNearestNeighbours "+str(nn)+")")
                #Replace the file type from nifti to tif
                filedata = filedata.replace("(ErrorBound "+err_bound+")", "(ErrorBound "+str(e)+")")
                #Replace the resulting image pixel type from double to 8bit
                filedata = filedata.replace("(Alpha "+alph+")", "(Alpha "+str(a)+")")
                #Get a temporary name
                tmp_name = 'aMI_'+str(nn)+'NN_'+str(e)+'ERR_'+str(a)+'A'
                #Make directories for the new names
                if not os.path.exists(tmp_name):
                    #make directory
                    os.mkdir(tmp_name)
                #Change to the new subdirectory
                os.chdir(tmp_name)
                #Create name for the new transform parameter file
                new_name = Path(os.path.join(os.getcwd(),str(tmp_name+'.txt')))
                #Write out the new file
                with open(new_name, 'w+') as new_file:
                    new_file.write(filedata)
                #Close the files
                file.close()
                new_file.close()
                #Add the new names to a list
                nms_list.append(new_name)
                #Switch back to the out directory
                os.chdir(out)
    #Change back to the home directory
    os.chdir(home_dir)
    #Return the path to the TransformParameters
    return nms_list


def MIParameterFileGrid(template,num_resolutions,out,nonlinear=False,grid_spacing=None):
    """Function for creating parameter file grid for alpha mutual information
    based registration"""

    #Get the current directory
    home_dir = Path(os.getcwd())
    #Get the parent directory and filename
    out = Path(out)
    os.chdir(out)
    #Get the current k value
    num_res,_ = _utils.ParseElastix(template,'NumberOfResolutions')
    #Set up a list for all new parameter file names
    nms_list = []
    #Check if nonlinear transformation
    if nonlinear:
        #Get the original grid spacing
        og_grid,_ = _utils.ParseElastix(template,'FinalGridSpacingInVoxels')
        #iterate through grid spacing schedule
        for gg in grid_spacing:
            #Iterate through the options for k,error,and alpha
            for nn in num_resolutions:
                #Read the transform parameters
                with open(template, 'r') as file:
                    filedata = file.read()
                #Replace the Alpha value
                filedata = filedata.replace("(NumberOfResolutions "+num_res+")", "(NumberOfResolutions "+str(nn)+")")
                #Replace the grid spacing schedule
                filedata = filedata.replace("(FinalGridSpacingInVoxels "+og_grid+")", "(FinalGridSpacingInVoxels "+str(gg)+")")
                #Get a temporary name
                tmp_name = 'MI_'+str(nn)+'Resolutions'+'_'+str(gg)+'GridSpacing'
                #Make directories for the new names
                if not os.path.exists(tmp_name):
                    #make directory
                    os.mkdir(tmp_name)
                #Change to the new subdirectory
                os.chdir(tmp_name)
                #Create name for the new transform parameter file
                new_name = Path(os.path.join(os.getcwd(),str(tmp_name+'.txt')))
                #Write out the new file
                with open(new_name, 'w+') as new_file:
                    new_file.write(filedata)
                #Close the files
                file.close()
                new_file.close()
                #Add the new names to a list
                nms_list.append(new_name)
                #Switch back to the out directory
                os.chdir(out)
            #Change back to the home directory
            os.chdir(home_dir)
    else:
        #Iterate through the options for k,error,and alpha
        for nn in num_resolutions:
            #Read the transform parameters
            with open(template, 'r') as file:
                filedata = file.read()
            #Replace the Alpha value
            filedata = filedata.replace("(NumberOfResolutions "+num_res+")", "(NumberOfResolutions "+str(nn)+")")
            #Get a temporary name
            tmp_name = 'MI_'+str(nn)+'Resolutions'
            #Make directories for the new names
            if not os.path.exists(tmp_name):
                #make directory
                os.mkdir(tmp_name)
            #Change to the new subdirectory
            os.chdir(tmp_name)
            #Create name for the new transform parameter file
            new_name = Path(os.path.join(os.getcwd(),str(tmp_name+'.txt')))
            #Write out the new file
            with open(new_name, 'w+') as new_file:
                new_file.write(filedata)
            #Close the files
            file.close()
            new_file.close()
            #Add the new names to a list
            nms_list.append(new_name)
            #Switch back to the out directory
            os.chdir(out)
        #Change back to the home directory
        os.chdir(home_dir)
    #Return the path to the TransformParameters
    return nms_list


def HyperParameterElastixRegistration(par_files,p=None,run_elastix = True,outdir=None,subfolder=None,**kwargs):
    """Function for iterating over a list of parameter files and running elastix
    on those parameter files"""

    #Get the current directory
    home_dir = Path(os.getcwd())

    #Create a list to store elastix log files and another for transform parameter files
    logs = []
    trans = []
    #Iterate through the files
    for par in par_files:
        #Set the paramter files
        if p is None:
            p0 = par
            p1=None
        else:
            p0=p
            p1 = par
        #Switch to the directory
        os.chdir(str(par.parent))
        #Check if moving to a subfolder (Use for multiple images in this function)
        if subfolder is not None:
            #Create a path object
            tmp_subfolder = Path(os.path.join(os.getcwd(),str(subfolder)))
            #Create folder if it doesnt exist
            if not tmp_subfolder.is_dir():
                os.mkdir(str(tmp_subfolder))
                #Change to subfolder
                os.chdir(str(tmp_subfolder))
        #Get the output path as the current working directory
        outdir = Path(os.getcwd())
        #Run elastix for this file if you choose
        if run_elastix:
            ElastixRegistration(p0=p0,p1=p1,out_dir=outdir,**kwargs)
        #Get the elastix log file from this directory
        tmp_log = _utils.SearchDir(ending = "elastix.log")
        #Add the log to the new list
        logs.append(tmp_log)
        #Get the final transform file
        tmp_trans,_ = _utils.GetFinalTransformParameters()
        #Add the transform parameter file to the list
        trans.append(tmp_trans)
        #Change back to the home_dir
        os.chdir(home_dir)
    #Return the parameter files
    return logs,trans,par_files



#-----------------Composition Registration Functions for MSI full image registration-----------------------

#Working towards a more succinct script
def CompositionElastix(out,f0,m0,p0_0,f1,m1,p0_1,p1_0=None,fp0=None,mp0=None,p1_1=None,fp1=None,mp1=None):
    """This function will perform multiple elastix registrations and save the output
    so that we can directly call transformix composition of functions.

    Input for points here will now allow for the csv files straight from Image J. We assume
    that you have run the FormatFijiPointsFromCSV function and exported"""
    #Perform the registration
    ElastixRegistration(fixed=f0,moving=m0,out_dir=out,p0=p0_0,p1=p0_1,fp=fp0,mp=mp0,mkdir=True)
    ElastixRegistration(fixed=f1,moving=m1,out_dir=out,p0=p1_0,p1=p1_1,fp=fp1,mp=mp1,mkdir=True)
    #Get the list of elastix files
    tmp_path = Path('..')
    home_dir=tmp_path.cwd()
    os.chdir(out)
    #Get the list of files in order
    dirFolders =  [f for f in os.listdir(out) if os.path.isdir(os.path.join(out, f))]
    dirFolders.sort(key=lambda f: int(f.split("elastix")[1]))
    #Create a new directory for the composition
    os.mkdir('elastix_composition')
    #Save the path to the first elastix folder
    first_trans = Path(os.path.join(out,dirFolders[0]))
    #Access the last file that elastix produced
    os.chdir(os.path.join(out,dirFolders[-1]))
    #List the files in this elastix directory that correspond to transform paramters
    elastix_dir=tmp_path.cwd()
    dirFiles=glob.glob('TransformParameters*')
    dirFiles.sort(key=lambda f: int(f.split(".")[1]))
    #Add new transform parameters to the created elastix_composition directory
    for i in range(len(dirFiles)):
        shutil.copyfile(dirFiles[i],os.path.join(out,'elastix_composition',Path(dirFiles[i]).stem)+'.comp.txt')
    #Go into the 1st registration parameter file and change to composition of Functions
    os.chdir(os.path.join(out,'elastix_composition'))
    dirComp = glob.glob('TransformParameters*')
    #Sort the files
    dirComp.sort(key=lambda f: int(f.split(".")[1]))
    #Only access the first file for composition (The second depends on first, if second exists)
    with open(dirComp[0], 'r') as file :
        filedata = file.read()
    #Access the first transform file
    trans_file=glob.glob(os.path.join(first_trans,'TransformParameters*'))
    trans_file.sort(key=lambda f: int(f.split(".")[1]))
    # Replace the target string
    filedata = filedata.replace("NoInitialTransform", str(os.path.join(first_trans,trans_file[-1])))
    # Write the file out again
    with open(dirComp[0], 'w') as file:
        file.write(filedata)
    #If you have two transform files in the last transformation, you will need to change the files there too
    if len(dirFiles) is 2:
        comp_file = tmp_path.cwd()
        with open(dirComp[1], 'r') as file :
            filedata = file.read()
        filedata.replace(str(os.path.join(elastix_dir,dirFiles[0])), str(os.path.join(comp_file,dirComp[0])))
        with open(dirComp[1], 'w') as file:
            file.write(filedata)



#-----------------Composition Registration Functions for extracting IMC ROIs-----------------------

#Testing to read in data and slice the correction padding so we can overlay
def RGBextractROI(dir,full_image,ROI_correction=None,flip_horz=False,flip_vert=False,export_image=False):
    """Function for exporting 3 channel ROIs from full images"""
    #Set your home directory
    tmp_path = Path('..')
    home_dir=tmp_path.cwd()
    os.chdir(dir)
    #Load the full image - Note this is a nifti image so we can use the index order given by GetROImask
    full_dat = nib.load(str(full_image)).get_fdata()
    #Get the final image size
    full_size = (full_dat.shape[0],full_dat.shape[1])
    #Get the coordinates for our ROIs
    ROI_masks_from_tolBlue = GetROImask(dir,full_size,ROI_correction)
    #Mask each channel using our pre-made masks!
    final_sizes = ROImaskExport(ROI_masks_from_tolBlue,full_dat,flip_horz=flip_horz,flip_vert=flip_vert,prefix=None,export_image=export_image)
    #Return the mask object so we can use it easily for the transformix ROI Extraction
    return ROI_masks_from_tolBlue,final_sizes



######Need to include resizing function for the ROIs based on the IMC Images
#that exist in each of the folders. Need to be able to have another registration
#for the exported ROIs extracted from the toluidine blue image





#####Create general function for masking and exporting the ROIs from the full Images
#you are repeating yourself in the RGB extraction and MSI extraction functions. In
#both instances we need to resize the ROIs upon saving so that we can apply another
#registration step
#****Both functions can be run through the folder structuure





def TransformixROIExtractionMSI(data,ROI_masks_from_tolBlue,final_sizes,par_composition,parameter_files,dir,rot1=1,flip_horz=False,flip_vert=False):
    """MUST BE USED AFTER imzMLreader function! This function will take the composition elastix transform file and will
    apply it to each of the channels in your MSI dataset. Given the directory of filename ROIs
    from fiji, the regions will then be cropped from the transformed image and stored
    as an individual slice. data: An imzMLParser class object

    data: An imzMLParser class object

    ROI_masks: Object returned from the RGB ROI extraction (step prior to registering false IMC H&E with tolblue-HE registered H&E image)

    par: The path to the parameter file to use for image registration on the full image (In our case the composition file)

    ROI_dict: Dictionary object indicating the name of files and filepaths to the csv Files

    rot1: Any extra rotation that you added before exporting the UMAP nifti image

    ROI_correction: Corrective padding to add onto the width and height of exported ROIs - integer values"""

    #Set your home directory
    tmp_path = Path('..')
    home_dir=tmp_path.cwd()
    #Change our directory to the output directory
    os.chdir(dir)
    trans_dir = tmp_path.cwd()

    #Get the final full image registration size from the composition parameter file\
    par = Path(par_composition)
    #full_size = GetParamFileImageSize(par).....why??

    #Get the ROI ROI_masks
    ROI_masks = ROI_masks_from_tolBlue

    #Fill the array with pixel values
    trans_start = time.time()
    #This loop will loop through all m/z channels in the SpectrumTable object stored in imzML reader class object
    for j in range(0,data.data.SpectrumTable.shape[1]):

        #Block for exporting the full m/z slice using the composition elastix transform parameters
        print('Transformix Working on slice: '+str(j)+'/'+str(data.data.SpectrumTable.shape[1])+'...')
        #Create a blank array to fill for each m/z channel
        im = np.zeros((data.data.imzmldict["max count of pixels y"],\
            data.data.imzmldict["max count of pixels x"]), dtype = np.float32)
        #Fill the image array with the m/z pixel intensities
        for i, (x, y, z) in enumerate(data.data.coordinates):
            im[y - 1, x - 1] = data.data.SpectrumTable.iloc[i,j]
        #Pad the image the same way you did for the dimension reduction image and registration with H&E image
        im2 = np.pad(im,[(data.data.Image_padx,data.data.Image_padx),(data.data.Image_pady,data.data.Image_pady)],mode='constant')
        #Rotate the image and resize the dame way you did for the dimension reduction image with H&E
        im2 = cv2.resize(np.rot90(im2,rot1),data.data.Image_resize_shape)
        #Create nifti image the same way you did the dimension reduction image with H&E
        tmp_nifti = nib.Nifti1Image(np.rot90(im2,data.data.Image_rot), affine=np.eye(4))
        nib.save(tmp_nifti,'MZ_slice'+str(j)+'.nii')
        print('Saved temporary nifti image')
        #Run transformix for this slice
        tmp_imagepath = Path(os.path.join(trans_dir,'MZ_slice'+str(j)+'.nii'))
        print('Running Transformix for slice '+str(j)+'...')
        os.system("transformix -in "+str(tmp_imagepath)+" -out "+str(Path(trans_dir))+" -tp "+str(par))

        #Now read the transformed image and crop the regions that we need
        full_dat = nib.load(os.path.join(str(trans_dir),'result.nii')).get_fdata()
		#Create a prefix for the ROIs so we can track channels
        pref = 'MZ_slice'+str(j)
        #For each slice, export and apply transformix to the corresponding MSI ROI
        ROImaskExport_MSI_Transformix(ROI_masks_from_tolBlue = ROI_masks,final_sizes = final_sizes,\
            full_img = full_dat,flip_horz=flip_horz,parameter_files=parameter_files,flip_vert=flip_vert,prefix=pref)
        #Delete our sliced image
        del full_dat
        os.remove('MZ_slice'+str(j)+'.nii')
        print('Finished slice '+str(j)+'/'+str(data.data.SpectrumTable.shape[1]))

    #Change back to our home directory
    os.chdir(home_dir)
    trans_stop = time.time()
    print('Finished transforming\n'+'Transformix Computation Time: '+str(trans_stop-trans_start)+' sec.')


def ExportMSIArray(data,export_name,rot1):

    #Copy the spectrum table for input to numba
    spec_table =  np.array(data.data.SpectrumTable.copy())
    #Create blank array
    im = np.zeros((data.data.imzmldict["max count of pixels y"],\
        data.data.imzmldict["max count of pixels x"],spec_table.shape[1]), dtype = np.float32)
    #Copy the coordinates of this dataset
    coords = data.data.coordinates.copy()


    #Define the function
    @njit(parallel=True)
    def parallel_array_fill(nd_im,coords,spec_table):
        """Function for quickly filling a large ndarray across z axis"""
        #Fill the array
        for slice_num in prange(spec_table.shape[1]):
            #Fill the image array with the m/z pixel intensities
            for i, (x, y, z) in enumerate(coords):
                nd_im[y - 1, x - 1,slice_num] = spec_table[i,slice_num]
        return(nd_im)

    #Run the function and fill the array
    filled_array = parallel_array_fill(nd_im = im ,coords = coords ,spec_table = spec_table)
    #Pad the image the same way you did for the dimension reduction image and registration with H&E image
    im2 = np.pad(filled_array,[(data.data.Image_padx,data.data.Image_padx),(data.data.Image_pady,data.data.Image_pady),(0,0)],mode='constant')
    #Rotate the image and resize the dame way you did for the dimension reduction image with H&E
    tmp_nifti = nib.Nifti1Image(np.rot90(np.rot90(im2,data.data.Image_rot),rot1), affine=np.eye(4))
    nib.save(tmp_nifti,str(export_name))


def TransformixROIExtractionMSI_slice(nibabel_obj,slice_num,resize_shape,ROI_masks_from_tolBlue,final_sizes,par_composition,parameter_files,dir,flip_horz=False,flip_vert=False):
    #Set your home directory
    tmp_path = Path('..')
    home_dir=tmp_path.cwd()
    #Change our directory to the output directory
    os.chdir(dir)
    trans_dir = tmp_path.cwd()

    #Get the final full image registration size from the composition parameter file\
    par = Path(par_composition)
    #full_size = GetParamFileImageSize(par).....why??

    #Get the ROI ROI_masks
    ROI_masks = ROI_masks_from_tolBlue

    #Get this slice from the exported nifti image from ExportMSIArray function
    im = nibabel_obj[:,:,slice_num]

    #Rotate the image and resize the dame way you did for the dimension reduction image with H&E
    im = cv2.resize(im,resize_shape)
    #Create nifti image the same way you did the dimension reduction image with H&E
    tmp_nifti = nib.Nifti1Image(im, affine=np.eye(4))
    nib.save(tmp_nifti,'MZ_slice'+str(slice_num)+'.nii')
    print('Saved temporary nifti image for slice '+str(slice_num))

    #Run transformix for this slice
    tmp_imagepath = Path(os.path.join(trans_dir,'MZ_slice'+str(slice_num)+'.nii'))
    print('Running Transformix for slice '+str(slice_num)+'...')
    subprocess.call("transformix -in "+str(tmp_imagepath)+" -out "+str(Path(trans_dir))+" -tp "+str(par))

    #Now read the transformed image and crop the regions that we need
    full_dat = nib.load(os.path.join(str(trans_dir),'result.nii')).get_fdata()
    #Create a prefix for the ROIs so we can track channels
    pref = 'MZ_slice'+str(slice_num)
    #For each slice, export and apply transformix to the corresponding MSI ROI
    ROImaskExport_MSI_Transformix(ROI_masks_from_tolBlue = ROI_masks,final_sizes = final_sizes,\
        full_img = full_dat,flip_horz=flip_horz,parameter_files=parameter_files,flip_vert=flip_vert,prefix=pref)
    #Delete our sliced image
    del full_dat
    os.remove('MZ_slice'+str(slice_num)+'.nii')


def TransformixROIExtractionMSI_parallel(data,nibabel_obj,ROI_masks_from_tolBlue,final_sizes,par_composition,parameter_files,dir,flip_horz=False,flip_vert=False,processes=-1):

    resize_shape = data.data.Image_resize_shape
    num_slices = nibabel_obj.shape[2]

    Parallel(n_jobs = processes)(delayed(TransformixROIExtractionMSI_slice)(nibabel_obj,slice_num,resize_shape,ROI_masks_from_tolBlue,\
        final_sizes,par_composition,parameter_files,dir,flip_horz,\
            flip_vert) for slice_num in range(num_slices))
    print('Finished Exporting')



def CompileROIs(home_dir):
    """This function is to be used for reading all m/z images in your final registrations
    folder in order to create a single nifti file containing the registered MSI stack"""
    #Set your working directory
    tmp=Path('..')
    os.chdir(home_dir)
    home_dir = tmp.cwd()
    start = time.time()
    #Get a list of files in the home directory
    list_folders = [os.path.join(tmp.cwd(),f) for f in os.listdir(tmp.cwd()) if os.path.isdir(os.path.join(tmp.cwd(), f))]
    #loop through each folder and concatenate
    for i in list_folders:
        #Switch to this directory
        os.chdir(i)
        #Get all of the nifti images in this directory
        MZ_images = _utils.TraverseDir(ending = '.nii')
        #Have to convert the images to strings for nibabel to concatenate them
        MZ_images = [str(image_path) for image_path in MZ_images]
        #Sort the images so the slices are in order
        MZ_images.sort(key=lambda f: int((str(Path(f).parent).split("MZ_slice")[-1])))
        #Concatenate the images to new nifti object
        print('Compiling '+str(Path(i).stem)+'...')
        tmp_nii = nib.concat_images(MZ_images)
        #Save the object
        print('Saving '+str(Path(i).stem)+'...')
        nib.save(tmp_nii,str(Path(i).stem)+'_result.nii')
        print('Saved '+str(Path(i).stem))
        #Remove our tmp nifti object
        del tmp_nii
        #Change directory back to the home directory
        os.chdir(home_dir)
    stop = time.time()
    print('Finished Compiling Images '+str(stop-start)+' sec. ')







#-----Create class object-----
class Elastix():
	"""Elastix image registration class
	"""

	def __init__(self,fixed,moving,out_dir,p,fp=None,mp=None,fMask=None):
		"""initialize class instance
		"""

		#Create pathlib objects and set class parameters
		self.fixed = Path(fixed)
		self.moving = Path(moving)
		self.out_dir = Path(out_dir)
		self.p = [Path(par_file) for par_file in p]
		self.fp = None if fp is None else self.fp = Path(fp)
		self.mp = None if mp is None else self.mp = Path(fp)
		self.fMask = None if fMask is None else self.fMask = Path(fMask)
		self.command = "elastix"

		#Load the images to check for dimension number
		print('Loading images...')
	    #Load images
	    niiFixed = nib.load(str(self.fixed))
	    niiMoving = nib.load(str(self.moving))
		#Print update
		print('Done loading')

	    #Check to see if there is single channel input (grayscale)
	    if niiFixed.ndim == 2 and niiMoving.ndim == 2:
	        print('Detected single channel input images...')
	        #Add fixed and moving image to the command string
	        self.command = self.command+" -f "+str(self.fixed)+ " -m "+str(self.moving)

	    #Check to see if there is multichannel input
	    else:
	        print('Exporting single channel images for multichannel input...')
	        #Read the images
	        niiFixed = niiFixed.get_fdata()
	        niiMoving = niiMoving.get_fdata()

	        #Set up list of names for the images
	        fixedList = []
	        movingList = []

	        #Export single channel images for each channel
	        for i in range(niiFixed.shape[2]):
	            #Create a filename
	            fname = Path(os.path.join(self.fixed.parent,str(self.fixed.stem+str(i)+self.fixed.suffix)))
	            #Update the list of names for fixed image
	            fixedList.append(fname)
	            #Update the list of names for fixed image
	            self.command = self.command + ' -f' + str(i) + ' ' + str(fname)
	            #Create a nifti image
	            #Check to see if the path exists
	            if not fname.is_file():
	                #Create a nifti image
	                nii_im = nib.Nifti1Image(niiFixed[:,:,i], affine=np.eye(4))
	                nib.save(nii_im,str(fname))

	        for i in range(niiMoving.shape[2]):
	            #Create a filename
	            mname = Path(os.path.join(self.moving.parent,str(self.moving.stem+str(i)+self.moving.suffix)))
	            #Update the list of names for moving image
	            movingList.append(mname)
	            #Update the list of names for moving image
	            self.command = self.command + ' -m' + str(i) + ' ' + str(mname)
	            #Check to see if the path exists
	            if not mname.is_file():
	                #Create a nifti image
	                nii_im = nib.Nifti1Image(niiMoving[:,:,i], affine=np.eye(4))
	                nib.save(nii_im,str(mname))

	    #Add the parameter files
	    self.command = self.command+" -p "+str(self.p[par_file]) for par_file in self.p

	    #Check for corresponding points in registration (must have fixed and moving set)
	    if self.fp and self.mp is not None:
			#Add to the command
	        self.command = self.command +" -fp "+str(self.fp)+" -mp "+str(self.mp)

	    #Check for fixed mask
	    if fMask is not None:
	        #Create pathlib Paths
	        fMask = Path(fMask)
	        self.command = self.command +" -fMask "+str(fMask)

	    #Check for making new directories
	    #if mkdir is True:
	    #    n=0
	    #    while n>=0:
	    #        tmp_name = "elastix"+str(n)
	    #        if not os.path.exists(Path(os.path.join(out_dir,tmp_name))):
	    #            os.mkdir(Path(os.path.join(out_dir,tmp_name)))
	    #            out_dir = Path(os.path.join(out_dir,tmp_name))
	    #            break
	    #        n+=1

	    #Add the output directory to the command
	    self.command = self.command +" -out "+str(self.out_dir)

	#Add main elastix component
	def RunElastix(self,command):
		"""
		Run the elastix registration. You must be able to call elastix
		from your command shell to use this. You must also have your parameter
		text files set before running (see elastix parameter files).

		Currently supports nifti1image format only!
		"""

		#Print command
		print(str(self.command))
		#Print elastix update
		print('Running elastix...')
		#Start timer
		start = time.time()
	    #Send the command to the shell
	    os.system(self.command)
		#Stop timer
		stop = time.time()
		#Print update
		print('Finished -- computation took '+str(stop-start)+'sec.')
	    #Return values
	    return self.command




class Transformix():
	"""Python class for transformix
	"""

	def __init__(self):
		"""initialize class instance
		"""

		#Create pathlib objects



#
