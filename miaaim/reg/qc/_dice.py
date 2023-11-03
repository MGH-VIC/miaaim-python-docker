# Dice score quality control module
# Developer: Joshua M. Hess, BSc
# Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

# import modules
import skimage
import SimpleITK as sitk
import numpy as np
import sys
import os
import nibabel as nib
from pathlib import Path
import re
import io
import pandas as pd
from skimage import measure
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import uncertainties as unc
import uncertainties.unumpy as unp
import matplotlib.pyplot as plt
# import custom modules
from miaaim.reg import _utils
import ElastixImageRegistration as ImReg

def itit_elastix_mask_parameters(TransformParameters):
    """Function for copying and altering a TransformParameters.txt elastix file
    for transforming moving mask
    (Use before calculating the dice score for image registration).

    Parameters
    ----------
    TransformParameters: str
        Path to elastix transform parameter file after elastix registration.

    Returns
    -------
    :str: Path to new file for binary mask transformix.
    """

    # read transform parameter file
    with open(TransformParameters, 'r') as file:
        filedata = file.read()
    # replace FinalBSplineInterpolationOrder with 0 order (binary mask)
    filedata = filedata.replace("(FinalBSplineInterpolationOrder 3)", "(FinalBSplineInterpolationOrder 0)")
    # replace file type from nifti to tif
    filedata = filedata.replace('(ResultImageFormat "nii")', '(ResultImageFormat "tif")')
    # replace resulting image pixel type from double to 8bit
    filedata = filedata.replace('(ResultImagePixelType "double")', '(ResultImagePixelType "char")')
    # get parent directory and filename
    filename = Path(TransformParameters)
    # create name for the new transform parameter file
    new_name = Path(os.path.join(str(filename.parent),str(filename.stem+"_mask"+filename.suffix)))
    # write out the new file
    with open(new_name, 'w+') as new_file:
        new_file.write(filedata)
    file.close()
    new_file.close()
    # return new file path
    return new_name

def convert_masks_to_nifti(masks):
    """Since the nifti format is used for image registration, the moving
    mask has to be converted to the nifti format as well for consistent rotation
    and center of origin.

    Parameters
    ----------
    masks: list
        List of paths to masks to initiate

    Returns
    -------
    :list: new mask file paths for transformix.
    """

    # check if input is list or a single file
    if type(masks) is not list:
        # if single file convert to list
        masks = [masks]
    # create new list for new names
    exp_names = []
    # iterate through masks and export to nifti format
    for mask in masks:
        # get mask path
        name = Path(mask)
        # create new name
        export_name = os.path.join(str(name.parent),name.stem+'.nii')
        # read TIFF mask
        im = skimage.io.imread(name)
        # convert to nifti image (have to flip and rotate to get same image)
        nii_im = nib.Nifti1Image(np.rot90(np.fliplr(im),1), affine=np.eye(4))
        # export nifti mask
        nib.save(nii_im,export_name)
        # update list
        exp_names.append(Path(export_name))
    # return new file paths
    return exp_names

def dice_score(fixed,transformed):
    """Function for calculating the dice score from a fixed mask and a transformed
    moving mask. This script assumes that the image is in the nifti format
    with pixel type uint8.

    Parameters
    ----------
    fixed: str
        Path to fixed image mask in the nifti image format (.nii).

    transformed: str
        Path to transformed moving image mask in the nifti image format (.nii).

    Returns
    -------
    :float: dice score.
    """

    # read fixed mask
    fixedImage = sitk.ReadImage(str(fixed), sitk.sitkUInt8)
    # read transformed moving mask
    transformedImage = sitk.ReadImage(str(transformed), sitk.sitkUInt8)
    # initiate
    dice_sim = sitk.LabelOverlapMeasuresImageFilter()
    # compute
    dice_sim.Execute(fixedImage,transformedImage)
    # compute dice score
    sim = dice_sim.GetDiceCoefficient()
    # return value
    return sim

def transformix_dice_scores(moving_masks,fixed_masks,transform_pars):
    """Function for iterating through a list of moving and fixed masks,
    applying transform parameter file, and calculating the dice score from each.
    Make sure that the list of moving and fixed masks line up perfectly! (names).

    Parameters
    ----------
    moving_masks: list
        List of paths to moving masks.

    fixed_masks: list
        List of paths to fixed masks.

    Returns
    -------
    """

    #Get the current directory
    home_dir = Path(os.getcwd())

    #Set up a list to store results
    results = []
    #Iterate through each parameter file
    for par in transform_pars:
        #Create a directory for the mask results if not one
        if not os.path.exists(str(par.parent)):
            os.mkdir(str(par.parent))
        #Change to this directory
        os.chdir(par.parent)
        #iterate through moving masks
        idx = 0
        #Set up list to keep results for this transform parameter
        mask_res = {}
        for m_mask in moving_masks:
            #Get new variable for this path
            out_dir = Path(os.getcwd())
            #Run elastix for this image
            ImReg.TransformixRegistration(input_img = m_mask,output_dir=out_dir,parameter_file=par,conc=False,points=None)
            #Get the results and calculate the dice score from this mask
            mov_res_mask = Path(os.path.join(os.getcwd(),"result.tif"))
            #Rename this file to the correct ROI for dice score
            mov_res_mask.replace(mov_res_mask.with_name(m_mask.stem+"_result.tif"))
            mov_res_mask = mov_res_mask.with_name(m_mask.stem+"_result.tif")
            #Get the dice score for this mask
            tmp_dice = DiceScore(fixed = fixed_masks[idx],moving=mov_res_mask)
            #Add this result to the list
            mask_res.update({m_mask.stem.replace("_msi",""):tmp_dice})
            #Update the index for this fixed image
            idx = idx +1
        #Concatenate the dictionary to pandas dataframe
        tmp_frame = pd.DataFrame([mask_res])
        #Add this frame to the results list
        results.append(tmp_frame)
    #Concatenate the results list
    results = pd.concat(results,axis=0)
    results.reset_index(drop=True, inplace=True)
    #add a column in the dataframe indicating the parameter file
    par_files_table = pd.DataFrame([str(file.parent.stem) for file in transform_pars],columns=["Parameters"])
    #Concatenate par files table and results table for return
    fin_table = pd.concat([par_files_table,results],axis=1)
    #Move back to the home directory
    os.chdir(home_dir)
    #Return the table
    return fin_table


def get_mask_boundary(im,mask,outdir,bbox=None,bbox_color='lime',contour_color='magenta',pad=50,rgb=False,change_type=False):
    """Function for using a binary mask of an image to create a bounding
    region on the original image for viewing purposes"""

    #Change directories
    home_dir = Path(os.getcwd())
    outdir = Path(outdir)

    #Create pathlib object for the image
    im = Path(im)
    #Create pathlib object for the mask
    mask = Path(mask)
    #Check the suffixes
    if im.suffix == '.tif' or im.suffix == '.tiff':
        #Read the image using tifffile
        image = skimage.io.imread(im,plugin='tifffile')
    #Otherwise reading the image as nifit
    elif im.suffix == '.nii':
        #use nibabel
        image = nib.load(str(im)).get_fdata()

    #Check the suffixes
    if mask.suffix == '.tif' or mask.suffix == '.tiff':
        #Read the image using tifffile
        image_mask = skimage.io.imread(mask,plugin='tifffile')
    #Otherwise reading the image as nifit
    elif mask.suffix == '.nii':
        #use nibabel
        image_mask = nib.load(str(mask)).get_fdata()

    #Now check to see if the mask and im match up, if not flip the nifti image
    if im.suffix != mask.suffix:
        #Check if rgb image
        if rgb:
            #Flip the image in each axis
            tmp_im = np.zeros(shape=(image[:,:,0].T.shape[0],image[:,:,0].T.shape[1],3))
            #Iterate through each channel
            for i in range(3):
                #Change the axis and add to the tmp_im
                tmp_im[:,:,i] = image[:,:,i].T.copy()
            #Assign the image
            image = tmp_im.copy()
        #Otherwise the image is grayscale
        else:
            #Flip the image from tifffile
            image = image.T
    #If both mask and image are nifti, transpose to make them easy for viewing
    if im.suffix == mask.suffix:
        #Check next condition
        if im.suffix == '.nii':
            #Check if rgb image
            if rgb:
                #Flip the image in each axis
                tmp_im = np.zeros(shape=(image[:,:,0].T.shape[0],image[:,:,0].T.shape[1],3))
                #Iterate through each channel
                for i in range(3):
                    #Change the axis and add to the tmp_im
                    tmp_im[:,:,i] = image[:,:,i].T.copy()
                #Assign the image
                image = tmp_im.copy()
            else:
                #Flip them both
                image = image.T
            #Flip the mask
            image_mask = image_mask.T

    #Now find the contour of the mask
    contours = measure.find_contours(image_mask, image_mask.min())
    #Get the xy dimensions of bounding box
    if bbox is None:
        #Get the bounding box
        bbox = []
        for contour in contours:
            Xmin = np.min(contour[:,0])
            Xmax = np.max(contour[:,0])
            Ymin = np.min(contour[:,1])
            Ymax = np.max(contour[:,1])
            bbox.append([Xmin, Xmax, Ymin, Ymax])
    else:
        Xmin = bbox[0]
        Xmax = bbox[1]
        Ymin = bbox[2]
        Ymax = bbox[3]
    #Display the image and plot all contours found
    fig, ax = plt.subplots()
    #Check for rgb plotting
    if not rgb:
        #Plot grayscale
        ax.imshow(image, cmap=plt.cm.gray)
    else:
        #Check if change type
        if change_type:
            #Plot rgb
            ax.imshow(np.array(image,np.int32))
        #Otherwise dont
        else:
            ax.imshow(image)
    #Get the length of the X contour
    lenX = Xmax-Xmin
    #Get the length of the X contour
    lenY = Ymax-Ymin
    # Create a Rectangle patch
    rect = patches.Rectangle((Ymin-pad,Xmin-pad),lenY+2*pad,lenX+2*pad,linewidth=2.5,edgecolor=bbox_color,linestyle = ":",facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)
    #Add the contours
    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2,color=contour_color)
    #Set blank ticks
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    #Save the figure
    plt.savefig(Path(os.path.join(str(outdir),mask.stem+"_contour.jpeg")),dpi=600,pad_inches = 0,bbox_inches='tight')

    #Now use the bounding box to crop the original image
    coords = np.array(rect.get_bbox())
    #Crop the orignal image
    crop_im = image[int(coords[0][1]):int(coords[1][1]),int(coords[0][0]):int(coords[1][0])]
    #Crop the mask
    crop_mask = image_mask[int(coords[0][1]):int(coords[1][1]),int(coords[0][0]):int(coords[1][0])]

    #Run the same process above except do not display bounding box
    contours_crop = measure.find_contours(crop_mask, crop_mask.min())
    #Display the image and plot all contours found
    fig, ax = plt.subplots()
    #Check for rgb plotting
    if not rgb:
        #Plot grayscale
        ax.imshow(crop_im, cmap=plt.cm.gray)
    else:
        #Check if change type
        if change_type:
            #Plot rgb
            ax.imshow(np.array(crop_im,np.int32))
        #Otherwise dont
        else:
            ax.imshow(crop_im)
    #Add the contours
    for n, contours_crop in enumerate(contours_crop):
        ax.plot(contours_crop[:, 1], contours_crop[:, 0], linewidth=1.75,color=contour_color)
    #Set blank ticks
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    #Save the figure
    plt.savefig(Path(os.path.join(str(outdir),mask.stem+"_contourCrop.jpeg")),dpi=600,pad_inches = 0,bbox_inches='tight')

    #Return the bounding box object for future use
    return {mask.stem:bbox}


def MultiDimParseElastixOptimization(dir=None,dimensions=range(2,11),iter = range(0,5),method_prefix="UMAP_",subfolder_name="MI_4Resolutions_300GridSpacing"):
    """Function for parsing multiple optimization procedures from image registrations
    using elastix. This function is specifically designed to parse registration
    optimizations used on multiple iterations of a dimension reduction method over
    multiple dimensions of data embedding. Designed for the study of alpha mutual
    information as embedding dimension increases. Default value of dimensions
    2 through 10. method_prefix is a string that will be used that indicates each iteration
    of embedding. Default is to use "umap_iter". For example, our analysis currently uses
    "umap_iter0" and "isomap_iter0" to stand for umap and isomap folders containing single
    result. iter is the range of values used to indicate what run of diimension reduction
    is being used. For example, "umap_iter0" is the first iteration of 5 (umap_iter4).

    Returned will be a dictionary containing 4 keys -- data, transformation type, dimensions and iterations.
    Also returned are two lists containing the keys for all so you can access
    the data easily from the "data" object"""

    #Get the home directory
    home_dir = Path(os.getcwd())
    #If directory is not specified, use the working directory
    if dir is None:
        tmp = Path('..')
        dir = tmp.cwd()

    #Create a dictionary to store all results in
    results = {}

    #Create a list of directories using the dimensions
    dirs = [Path(os.path.join(str(dir),"Dimension"+str(dim))) for dim in dimensions]

    #Create a list to store the affine results
    result_aff = {}
    #Create a list to store the bspline results
    result_bspl = {}
    #Iterate through the list of directories
    for d in dirs:
        #Get the folder name (dimension)
        tmp_dim = d.stem
        #Switch to the new directory
        os.chdir(str(d))
        #Get all of the directories in this directory that match the method
        iternms = [Path(os.path.join(str(d),str(subfolder_name),method_prefix+"iter"+str(i)+"_multi")) for i in iter]

        #Create a temporary dictionary to store results in for a dimension affine
        dim_aff = {}
        #Create a temporary dictionary to store results in for a dimension bspline
        dim_bspl = {}
        #Iterate through each run and parse optimization of metric
        for it in iternms:
            #Get the iteration name
            itnm = it.stem
            #Get the optimization results
            tmp_vals = ParseElastixOptimization(it)
            #Create a dataframe with these results for affine registration
            aff = pd.DataFrame({'Dimension': int(re.findall(r"[-+]?\d*\.\d+|\d+", tmp_dim)[0]), 'Iteration': itnm,'Metric':tmp_vals["0"]["Metric"]})
            #Create a dataframe with these results for nonlinear registration
            bspl = pd.DataFrame({'Dimension': int(re.findall(r"[-+]?\d*\.\d+|\d+", tmp_dim)[0]), 'Iteration': itnm,"Metric":tmp_vals["1"]["Metric"]})
            #Update the affine dictionary
            dim_aff.update({str(itnm):aff})
            #Update the bspline dictionary
            dim_bspl.update({str(itnm):bspl})

        #Add these results for a particular dimension to the results dictionary
        result_aff.update({str(tmp_dim):dim_aff})
        #Add these results for a particular dimension to the results dictionary
        result_bspl.update({str(tmp_dim):dim_bspl})
        #Change back to the home directory
        os.chdir(home_dir)

    #Update the final results to store nonlinear and affine registration
    results.update({"Affine":result_aff,"Bspline":result_bspl})

    #Return the results and keys to access the results dictionary
    return {"data":results, "transforms":["Affine","Bspline"], "dimensions":[d.stem for d in dirs], "iters":[it.stem for it in iternms]}


def Exp(x, a, b, c):
    """Exponential function to use for regression"""
    return a * np.exp(-b * x) + c


def MultiExponentialFitElastix(results_dict, p0=(0, 0.01, 1),export=True):
    """Function for fitting an exponential function to a parsed optimization
    procedure produced from elastix. p0 are initial estimates. Results will simply
    be another list that mirrors the results from MultiDimParseElastixOptimization

    Fit results will be stored in the 'fits' key and can be accessed in the same way
    as the MultiDimParseElastixOptimization return dictionary"""

    #Create a dictionary to store the fit results in
    fit_results = {}

    #Iterate through the top levels (affine vs bspline)
    for t in results_dict["transforms"]:
        #Create dictionary to store results for this transformation
        fit_trans = {}
        #Create a list to store asymptote results affine
        fit_asy = []
        #Iterate through each dimension
        for d in results_dict["dimensions"]:
            #Create dictionary to store results for this dimension
            fit_dim = {}
            #Create a list to store concatenated metric results
            dat_dim = []
            #Create a list to store metric results for export
            exp_dim = []
            #Iterate through each iteration
            for i in results_dict["iters"]:
                #Get the data for this iteration, dimension, and transform
                dat = results_dict["data"][str(t)][str(d)][str(i)]
                #Get the metric values
                met = dat["Metric"].values
                #Get the x axis information
                xdata = np.int64(dat.index.values)
                #Fit the data using exponential function
                popt, pcov = curve_fit(Exp, xdata, met, p0 = p0)
                #Create results dictionary to store the results
                iter_fit = {"popt":popt,"pcov":pcov}

                #Add the iter results to the dimensions results dictionary
                fit_dim.update({str(i):iter_fit})
                #Add data to list of export dimension results
                exp_dim.append(pd.DataFrame({str(i):met,"Iteration":np.arange(1,met.shape[0]+1,dtype='int64')}))
                #Add the data object to the list of dimenson results
                dat_dim.append(pd.DataFrame({"Metric":met,"Iteration":np.arange(1,met.shape[0]+1,dtype='int64'),"Run":str(i)}))
                #Extract the asymptote in c for the fit for future summarization and add to pandas dataframe
                fit_asy.append(pd.DataFrame({"Metric Estimate":[popt[2]],"Iter":[str(i)],"Dimension":[int(re.findall(r"[-+]?\d*\.\d+|\d+", d)[0])]}))

            #Concatenate the results from each iteration for dimension results
            dat_dim = pd.concat(dat_dim,axis=0)
            #Concatenate the results from each iteration for dimension results export
            exp_dim = pd.concat(exp_dim,axis=1)
            #Fit the full data here
            popt, pcov = curve_fit(Exp, dat_dim.Iteration.values, dat_dim.Metric.values, p0 = p0)
            #Add the dimension data to the transformation results dictionary
            fit_dim.update({'Full':{'dat':dat_dim,'export':exp_dim,'fit':{'popt':popt, "pcov":pcov}}})

            #Add the dimension results to the transformation results dictionary
            fit_trans.update({str(d):fit_dim})


        #Concatenate the asymptote values
        fit_asy = pd.concat(fit_asy,axis=0)
        #Add these asymptote values to the dictionary
        fit_trans.update({"Metric Estimates": fit_asy})
        #Add the results for this transformation to the final results
        fit_results.update({str(t):fit_trans})

    #Copy the input dictionary
    return_dict = results_dict.copy()
    #Add the fit results to the same input list
    return_dict.update({"fits":fit_results})

    #Check to see if exporting data
    if export:
        #Export the asymptote estimates
        with pd.ExcelWriter('AsymptoteAlphaMI.xlsx') as writer:
            #Get both transformation types
            for t in list(return_dict["fits"].keys()):
                #Write sheet
                return_dict["fits"][str(t)]["Metric Estimates"].to_excel(writer,sheet_name=str(t))

        #Export the full iteratio summary per dimension
        with pd.ExcelWriter('FullAlphaMI.xlsx') as writer:
            #Get both transformation types
            for t in list(return_dict["fits"].keys()):
                #Create a temporary set of names for the dimensions (we dont want to include one name)
                tmp_nms = list(return_dict["fits"][str(t)].keys())
                tmp_nms.remove("Metric Estimates")
                #Iterate through dimensions
                for d in tmp_nms:
                    #Get data and export sheet
                    return_dict["fits"][str(t)][str(d)]["Full"]['export'].to_excel(writer,sheet_name=str(t)+" "+str(d))
    #Return the new fit and results
    return return_dict




























#
