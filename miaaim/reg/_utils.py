# General elastix image registration and transformix utility functions
# Developer: Joshua M. Hess, BSc
# Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

# import modules
import os
from pathlib import Path
import re
import pandas as pd
import numpy as np
import warnings
import skimage.io
import nibabel as nib

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


def TraverseDir(ending=".txt",dir=None):
    """Traverse a directory to search for files that end with the
    specified suffix.

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
    #Traverse the directory to search for files
    full_list = []
    for root, dirs, files in os.walk(dir):
        for f in files:
            if f.endswith(ending):
                full_list.append(Path(os.path.join(root,f)))
    #Return the list
    return full_list



def GetFinalTransformParameters(dir=None):
    """Extract Transform Parameter files from elastix in order.

    Parameters
    ----------
    dir: string (Default: None, will search in current working directory)
        Directory to search for files in.

    Returns
    -------
    full_list: list
        Transform paramter files as pathlib objects in order.
    """

    #If directory is not specified, use the working directory
    if dir is None:
        tmp = Path('..')
        dir = tmp.cwd()

    #Search the directory only for files
    full_list = []
    for file in os.listdir(dir):
        if "TransformParameters" in file:
            full_list.append(Path(os.path.join(dir,file)))
    #Order the list to get the last transform parameter file
    full_list.sort(key=lambda f: int(str(f).split("TransformParameters.")[1].split(".")[0]))
    #Return the list
    return full_list[-1],full_list



def GetFirstTransformParameters(dir=None):
    """Extract Transform Parameter files from elastix in order.

    Parameters
    ----------
    dir: string (Default: None, will search in current working directory)
        Directory to search for files in.

    Returns
    -------
    full_list: list
        Transform paramter files as pathlib objects in order.
    """

    #If directory is not specified, use the working directory
    if dir is None:
        tmp = Path('..')
        dir = tmp.cwd()

    #Search the directory only for files
    full_list = []
    for file in os.listdir(dir):
        if "TransformParameters" in file:
            full_list.append(Path(os.path.join(dir,file)))
    #Order the list to get the last transform parameter file
    full_list.sort(key=lambda f: int(str(f).split("TransformParameters.")[1].split(".")[0]))
    #Return the list
    return full_list[0],full_list



def ParseElastix(input,par):
    """Parse an elastix parameter file or elastix.log file and
    extract a number associated with the given string parameter.

    Parameters
    ----------
    input: string
        Path to input file.

    par: string
        String indicating the parameter in the files to extract.

    Returns
    -------
    number: string
        Orginal string found in the given file. This will be converted to an
        integer or a floating point number.

    num: integer or float
        Number corresponding to the parameter specified.
    """

    #Read the transform parameters
    with open(input, 'r') as file:
        filedata = file.readlines()
    #Add each line to a list with separation
    result=[]
    for x in filedata:
        result.append(x.split('\n')[0])
    #Find the parameter (Add a space for a match)
    lines = [s for s in result if str(par+' ') in s][-1]
    number = re.findall(r"[-+]?\d*\.\d+|\d+", lines)[0]
    #Try to convert to integer, otherwise convert to float
    if number.isdigit():
        num = int(number)
    else:
        num = float(number)
    #Return the number
    return number, num

def CreateInverseTransformFiles(p,outdir=None):
    """Take transformix parameter files and create inverse
    transform parameter file for inverse optimization.

    Parameters
    ----------
    p: string
        indicates parameters files to adapt after inverse transform.

    Returns
    -------
    file_paths: string
        path to inverse transform file created from elastix
    """
    # create pathlib objects
    ps = [Path(par_file) for par_file in p]

    # create list for new files
    out_files = []
    # iterate through parameter files
    for input in ps:
        # get components of filename
        nm = input.stem+'-Inverse.txt'
        # check for output directory
        if outdir is None:
            dir = input.parent
        else:
            dir = outdir
        # create output name
        out_nm = Path.joinpath(dir,nm)
        # update output list
        out_files.append(out_nm)

        #Read the registration parameters
        with open(input, 'r') as file:
            filedata = file.readlines()
        # create copy of filedata
        result_out = filedata.copy()
        #Add each line to a list with separation
        result=[]
        for x in filedata:
            result.append(x.split('\n')[0])

        # first look for metric
        par = "Metric"
        # search for initial transform parameter file
        try:
            # get line which matches this parameter
            lines = [s for s in result if str(par+' ') in s]
            # filter the lines
            for l in lines:
            	# extract first word
            	test = l.split(" ")[0].replace('(',"")
            	# see if first word is a match
            	if test == par:
            		# set line to be the desired string
            		lines = l
            		# break
            		break
            # get the string after space (exclude parenthesis at end)
            value = lines.split(" ")[1:][0][:-1]
            # replace the value in the line with new value
            newline = lines.replace(value,"DisplacementMagnitudePenalty")
            # replace the results with the newline
            result_out=list(map(lambda x: x.replace(lines,newline),result_out))
        # # if not available, add it
        except:
            # get new line with metric information
            newline = '(Metric DisplacementMagnitudePenalty)\n'
            # add new line to results
            result_out.insert(0,newline)

        # replace compose transforms
        par = 'HowToCombineTransforms'
        # search for initial transform parameter file
        try:
            # get line which matches this parameter
            lines = [s for s in result if str(par+' ') in s][-1]
            # get the string after space (exclude parenthesis at end)
            value = lines.split(" ")[1:][0][:-1]
            # replace the value in the line with new value
            newline = lines.replace(value,"Compose")
            # replace the results with the newline
            result_out=list(map(lambda x: x.replace(lines,newline),result_out))
        # # if not available, add it
        except:
            # get new line with metric information
            newline = '(HowToCombineTransforms Compose)\n'
            # add new line to results
            result_out.insert(0,newline)

        # export new file
        with open(out_nm, 'w') as f:
            f.writelines(result_out)
        return out_files

def AlterInverseTransform(input):
    """Alter initial inverse transform parameter to have no initial transform.

    Parameters
    ----------
    input: string
        indicates file to adapt after inverse transform.
    """
    # change to pathlib
    input = Path(input)
    # set path to output file
    output_f = Path(input)
    #Read the registration parameters
    with open(input, 'r') as file:
        filedata = file.readlines()
    # create copy of filedata
    result_out = filedata.copy()
    #Add each line to a list with separation
    result=[]
    for x in filedata:
        result.append(x.split('\n')[0])

    # replace compose transforms
    par = 'InitialTransformParametersFileName'
    # get line which matches this parameter
    lines = [s for s in result if str(par+' ') in s]
    # filter the lines
    for l in lines:
    	# extract first word
    	test = l.split(" ")[0].replace('(',"")
    	# see if first word is a match
    	if test == par:
    		# set line to be the desired string
    		lines = l
    		# break
    		break
    # get the string after space (exclude parenthesis at end)
    value = lines.split(" ")[1:][0][:-1]
    # replace the value in the line with new value
    newline = lines.replace(value,"NoInitialTransform")
    # replace the results with the newline
    result_out=list(map(lambda x: x.replace(lines,newline),result_out))

    # export new file
    with open(output_f, 'w') as f:
        f.writelines(result_out)




# input = Path("/Users/joshuahess/Desktop/aMI_affine.txt")
# outdir = None


def ValidateKNNaMIparams(cF,mF,p,outdir=None):
	"""Validate automatically the KNN alpha MI paramter file for multichannel
	images. Input directions can be easy to miss, so here an automatic
	validation is built in.

	Parameters
	----------
	cF: int
		Indicates number of channels in multichannel fixed image.

	mF: int
		Indicates number of channels in multichannel moving image.

	input: list
		List of paths to parameter files.

	outdir: str (default: input file directory given by None)
		Path to output directory
	"""
	# create pathlib objects
	ps = [Path(par_file) for par_file in p]
	# create list for new files
	out_files = []
	# iterate through parameter files
	for input in ps:
		# create new flag for exporting new text file
		flag = False

		# get components of filename
		nm = input.stem+'-validated.txt'
		# check for output directory
		if outdir is None:
			dir = input.parent
		else:
			dir = outdir

		# create output name
		out_nm = Path.joinpath(dir,nm)

		#Read the registration parameters
		with open(input, 'r') as file:
			filedata = file.readlines()

		# create copy of filedata
		result_out = filedata.copy()
		# add each line to a list with separation
		result=[]
		for x in filedata:
			result.append(x.split('\n')[0])

		# replace compose transforms
		par = 'Metric'
		# get line which matches this parameter
		lines = [s for s in result if str(par+' ') in s]
		# filter the lines
		for l in lines:
			# extract first word
			test = l.split(" ")[0].replace('(',"")
			# see if first word is a match
			if test == par:
				# set line to be the desired string
				lines = l
				# break
				break

        # get the string after space (exclude parenthesis at end)
		value = lines.split(" ")[1:][0][:-1]
        # check for KNN aMI
		if "KNNGraphAlphaMutualInformation" in value:

    		# replace compose transforms
			par = 'FixedImagePyramid'
    		# get line which matches this parameter
			lines = [s for s in result if str(par+' ') in s]
    		# filter the lines
			for l in lines:
    			# extract first word
				test = l.split(" ")[0].replace('(',"")
    			# see if first word is a match
				if test == par:
    				# set line to be the desired string
					lines = l
    				# break
					break
    		# get the length of the line (number of parameters)
			len_values = len(lines.split(" ")[1:])
    		# get the string after space (exclude parenthesis at end)
			value = lines.split(" ")[1:][0]
    		# check to see if number of values equals number of channels
			if len_values != cF:
    			# set flag to true for this file
				flag = True
    			# create new string by duplicating original value
				out = ''
				for i in range(cF):
					if i == 0:
						out += value
					else:
						out += " " + value
    			# raise warning
				warnings.warn(f'FixedImagePyramid parameters {lines} do not match fixed image dimensions. Automatically updating to {out}')
    			# get the actual values for the line
				vs_actual = ''
				for i in range(len_values):
					if i == 0:
						vs_actual += value
					else:
						vs_actual += " " + value
    			# replace the value in the line with new value
				newline = lines.replace(vs_actual,out)
    			# replace the results with the newline
				result_out=list(map(lambda x: x.replace(lines,newline),result_out))

    		# replace compose transforms
			par = 'MovingImagePyramid'
    		# get line which matches this parameter
			lines = [s for s in result if str(par+' ') in s]
    		# filter the lines
			for l in lines:
    			# extract first word
				test = l.split(" ")[0].replace('(',"")
    			# see if first word is a match
				if test == par:
    				# set line to be the desired string
					lines = l
    				# break
					break
    		# get the length of the line (number of parameters)
			len_values = len(lines.split(" ")[1:])
    		# get the string after space (exclude parenthesis at end)
			value = lines.split(" ")[1:][0]
    		# check to see if number of values equals number of channels
			if len_values != mF:
    			# set flag to true for this file
				flag = True
    			# create new string by duplicating original value
				out = ''
				for i in range(mF):
					if i == 0:
						out += value
					else:
						out += " " + value
    			# raise warning
				warnings.warn(f'MovingImagePyramid parameters {lines} do not match fixed image dimensions. Automatically updating to {out}')
    			# get the actual values for the line
				vs_actual = ''
				for i in range(len_values):
					if i == 0:
						vs_actual += value
					else:
						vs_actual += " " + value
    			# replace the value in the line with new value
				newline = lines.replace(vs_actual,out)
    			# replace the results with the newline
				result_out=list(map(lambda x: x.replace(lines,newline),result_out))

    		# replace compose transforms
			par = 'Interpolator'
    		# get line which matches this parameter
			lines = [s for s in result if str(par+' ') in s]
    		# filter the lines
			for l in lines:
    			# extract first word
				test = l.split(" ")[0].replace('(',"")
    			# see if first word is a match
				if test == par:
    				# set line to be the desired string
					lines = l
    				# break
					break
    		# get the length of the line (number of parameters)
			len_values = len(lines.split(" ")[1:])
    		# get the string after space (exclude parenthesis at end)
			value = lines.split(" ")[1:][0]
    		# check to see if number of values equals number of channels
			if len_values != mF:
    			# set flag to true for this file
				flag = True
    			# create new string by duplicating original value
				out = ''
				for i in range(mF):
					if i == 0:
						out += value
					else:
						out += " " + value
    			# raise warning
				warnings.warn(f'Interpolator parameters {lines} do not match fixed image dimensions. Automatically updating to {out}')
    			# get the actual values for the line
				vs_actual = ''
				for i in range(len_values):
					if i == 0:
						vs_actual += value
					else:
						vs_actual += " " + value
    			# replace the value in the line with new value
				newline = lines.replace(vs_actual,out)
    			# replace the results with the newline
				result_out=list(map(lambda x: x.replace(lines,newline),result_out))

		# check for flag
		if flag:
			# update output list
			out_files.append(out_nm)
			# write a new file
			with open(out_nm, 'w') as f:
				f.writelines(result_out)
		# otherwise leave orinal input
		else:
			# leave the parameter file
			out_files.append(input)
	# return the name of all files
	return out_files



# testing the points formatter
# txt_file = "/Users/joshuahess/Desktop/POINTS.txt"
# FormatFijiLandmarkPoints(txt_file)


def FormatFijiLandmarkPoints(txt_file, selection_type):
    """This function will take the text point selection file that you export from
    using control+m in fiji (ImageJ) to the correct text file format to use with
    elastix image registration

    selection_type must be the string 'index' or 'points'"""
    
    # create pathlib object from text file
    txt_file = Path(txt_file)
    #Get the image folder name
    parent=Path(txt_file).parent    
    #Remove the file extension
    outname=parent.joinpath(txt_file.stem+"-validated.txt")
    
    #Read input text file
    data = pd.read_csv(txt_file, sep='\t', names=["X","Y"])

    #Create a new text file
    txt_file = open(outname,"w+")
    #Create first string to write to your file
    str_list = [str(selection_type),str(data.shape[0])]
    txt_file.writelines(i + '\n' for i in str_list)
    #Close and save the txt file
    txt_file.close()
    
    #Get only the data we need for the text file
    point_tab = data[['X','Y']]
    #Now append the data table to the txt file
    point_tab.to_csv(outname, header=False, index=False, sep=' ', mode='a')
    
    # return the new name
    return outname



# testing
#in_dir = Path("/Users/joshuahess/Desktop/test")
#TransformParameters = [ in_dir.joinpath("TransformParameters.0.txt"),
#         in_dir.joinpath("TransformParameters.1.txt") ]
#outdir=in_dir

def InitiateMaskTransformParameters(TransformParameters, outdir):
    """
    Function for copying and altering a TransformParameters.txt elastix file
    for transforming moving mask. Use for transforming a mask prior to
    filtering segmented cells, or for computing registration metrics involving
    masks.

    Parameters
    ----------
    TransformParameters : TYPE
        DESCRIPTION.

    Returns
    -------
    new_name : TYPE
        DESCRIPTION.

    """
	# create pathlib objects
    ps = [Path(par_file)for par_file in TransformParameters]
    outdir = Path(outdir)
    # create list for new files
    out_files = []
	# iterate through parameter files
    for p in ps:
        p = ps[0]
		# create new flag for exporting new text file
        flag = False

		# get components of filename
        nm = "Mask_"+p.name
		# create output name
        out_nm = Path.joinpath(outdir,nm)

		#Read the registration parameters
        with open(p, 'r') as file:
            filedata = file.readlines()

		# create copy of filedata
        result_out = filedata.copy()
		# add each line to a list with separation
        result=[]
        for x in filedata:
            result.append(x.split('\n')[0])

    # replace compose transforms
    par = 'FinalBSplineInterpolationOrder'
    # get line which matches this parameter
    lines = [s for s in result if str(par+' ') in s]
    # filter the lines
    for l in lines:
    	# extract first word
    	test = l.split(" ")[0].replace('(',"")
    	# see if first word is a match
    	if test == par:
    		# set line to be the desired string
    		lines = l
    		# break
    		break
    # get the string after space (exclude parenthesis at end)
    value = lines.split(" ")[1:][0][:-1]
    # replace the value in the line with new value
    newline = lines.replace(value,"0")
    # replace the results with the newline
    result_out=list(map(lambda x: x.replace(lines,newline),result_out))



    #Read the transform parameters
    with open(TransformParameters, 'r') as file:
        filedata = file.read()
    #Replace the FinalBSplineInterpolationOrder with 0 order
    filedata = filedata.replace("(FinalBSplineInterpolationOrder 3)", "(FinalBSplineInterpolationOrder 0)")
    #Replace the file type from nifti to tif
    filedata = filedata.replace('(ResultImageFormat "nii")', '(ResultImageFormat "tif")')
    #Replace the resulting image pixel type from double to 8bit
    filedata = filedata.replace('(ResultImagePixelType "double")', '(ResultImagePixelType "char")')
    #Get the parent directory and filename
    filename = Path(TransformParameters)
    #Create name for the new transform parameter file
    new_name = Path(os.path.join(str(filename.parent),str("Mask_"+filename.stem+filename.suffix)))
    #Write out the new file
    with open(new_name, 'w+') as new_file:
        new_file.write(filedata)
    #Close the files
    file.close()
    new_file.close()
    #Return the path to the TransformParameters
    return new_name


def ConvertMasksToNifti(masks):
    """Since the nifti format is used for image registration, the moving
    mask has to be converted to the nifti format 
    as well due to rotation and center of origin
    """

    #Check if the images are a list or a single file
    if not isinstance(masks, list):
        #if the masks are a single file, convert them to a list
        masks = [masks]
    #Set up a list of export names
    exp_names = []
    #Iterate through the masks list and export to nifti
    for mask in masks:
        #Get the image path and name
        name = Path(mask)
        #Get the new name
        export_name = os.path.join(str(name.parent),name.stem+'.nii')
        #Read the image
        im = skimage.io.imread(name)
        #convert to nifti image (have to flip and rotate to get same image)
        nii_im = nib.Nifti1Image(np.rot90(np.fliplr(im),1), affine=np.eye(4))
        #Save the image
        nib.save(nii_im,export_name)
        #Update the export name list
        exp_names.append(Path(export_name))
    #Return the export path for future reference
    return exp_names







#import skimage.io
#import nibabel as nib
#import matplotlib.pyplot as plt

#name = "/Users/joshuahess/Desktop/ROI014_PROSTATE_TMA005_core_msi_mask.tif"
#im = skimage.io.imread(name)
#convert to nifti image (have to flip and rotate to get same image)
#nii_im = nib.Nifti1Image(np.rot90(np.fliplr(im),1), affine=np.eye(4))
#Save the image
#nib.save(nii_im,"/Users/joshuahess/Desktop/ROI014_PROSTATE_TMA005_core_msi_mask.nii")





#
