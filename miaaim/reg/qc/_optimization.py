# Optimization quality control module
# Developer: Joshua M. Hess, BSc
# Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

# import modules
import numpy as np
import pandas as pd
from pathlib import Path
import os

def parse_exact_elastix_metric(elastix_log):
    """Function for parsing the exact metric from elastix.log file.

    Assumes that "ExactMetric" option is set to true in elastix
    registration parameter file.

    Parameters
    ----------
    elastix_log: str
        Path to elastix.log file.

    Returns
    -------
    :float: Final exact metric value.
    """

    # get number of resolutions
    _,NumberOfResolutions = utils.ParseElastix(elastix_log,'NumberOfResolutions')
    # read the transform parameters
    with open(elastix_log, 'r') as file:
        filedata = file.readlines()
    # add each line to a list with separation
    result=[]
    for x in filedata:
        result.append(x.split('\n')[0])
    # find the line number for exact metric
    ExactMetrics = []
    lines = []
    line = 0
    for s in result:
        # check if ExactMetric is in the results
        if 'ExactMetric0' in s:
            # if it is, add to the list of
            ExactMetrics.append(s)
            # also update the list of lines
            lines.append(line)
        # update the counter
        line=line+1
    # find the line number after exact metric for the end of optimization
    EndofOpt = []
    lines_opt = []
    line_opt = 0
    for s in result:
        # check if ExactMetric is in the results
        if str('Time spent in resolution '+str((NumberOfResolutions-1))) in s:
            # if it is, add to list
            EndofOpt.append(s)
            # also update the list of lines
            lines_opt.append(line_opt)
        # update the counter
        line_opt=line_opt+1
    # get a data table with the values for this resolution optimization
    table = pd.read_csv(io.StringIO('\n'.join(result[lines[-1]:lines_opt[-1]])), delimiter='\t')
    # get the final exact metric value
    ExactMetric_fin = table.iloc[-1]['ExactMetric0']
    # return the final metric
    return ExactMetric_fin

def parse_elastix_optimization(dir):
    """Function for reading transform parameter files to track reampled cost
    function value over optimization procedure.

    Parmeters
    ---------
    dir: str
        Path to directory with transform parameter files (IterationInfo files).

    Returns
    -------
    :dict: Dictionary with resulting metric values over optimization.
    """

    # create pathlib object from directory path
    dir = Path(dir)

    # search directory for iteration files
    full_list = []
    for file in os.listdir(str(dir)):
        if "IterationInfo" in file:
            full_list.append(Path(os.path.join(str(dir),file)))
    # order the list to get the optimization procedure in order
    full_list.sort(key=lambda f: (str(f).split("IterationInfo.")[1].split(".")[0],int(re.search(r'\d+', str(f).split("IterationInfo.")[1].split(".")[1]).group())))

    # get the unique registration components
    comps = np.unique([str(f).split("IterationInfo.")[1].split(".")[0] for f in full_list])
    # create dictionary to store the results in
    dict = {}
    for c in comps:
        # add to dictionary
        dict.update({str(c):[]})
    # read each of the optimization files
    for f in full_list:
        # check to see which registration component
        c = str(f).split("IterationInfo.")[1].split(".")[0]
        # read the optimization file
        dat = pd.read_csv(f,sep='\t')
        # get the metric information
        dat = pd.DataFrame(list(dat["2:Metric"]),columns=["Metric"])
        # add this to the list inside the dictionary
        dict[str(c)].append(dat)
    # concatenate all of the dataframes in the list
    for c in dict:
        # concatenate
        dict[str(c)] = pd.concat(dict[str(c)],axis=0)
        # reset the indices
        dict[str(c)].reset_index(drop=True, inplace=True)
    # return the dictionary
    return dict


def ParseMultiMetricMultiResolutionOptimization(dirs):
    """Function for parsing multiple optimization procedures from image registration
    with multiple resolutions"""

    #Create dictionary to store results in
    results = {}
    #Iterate through each directory in dirs
    for d in dirs:
        #Get the resolution level filename
        res = d.stem
        #Parse the elastix optimization procedure
        results.update({res:ParseElastixOptimization(d)})

    #Make sure all the optimizations have same number of transformations
    num_trans = []
    for res, opt in results.items():
        #Get the number of transformations
        num_trans.append(len(opt.keys()))
    #Check if all equal
    if not num_trans.count(num_trans[0]) == len(num_trans):
        #Raise error
        raise ValueError("Number of transformations are not equal")
    #Otherwise proceed
    else:
        #Get again the number of transformations as an int this time
        num_trans = num_trans[0]
        #Create a dictionary to store final results concatenations
        final_dict = {str(trans): {} for trans in range(num_trans)}
        #Extract all resolutions for each transformation
        for res, opt in results.items():
            #Iterate through the transformations
            for trans, iter in opt.items():
                #Extract the transformation and update to the dictionary
                final_dict[trans].update({str(res):iter})
        #Iterate through the final_dict and create dataframes
        for key in final_dict.keys():
            #Create dataframe
            trans_frame = pd.concat([final_dict[key][trans] for trans in final_dict[key].keys()],axis=1)
            #Fix column names
            trans_frame.columns = final_dict[key].keys()
            #Update the final_dict to replace old data with new frame
            final_dict[key] = trans_frame
        #Export the asymptote estimates
        with pd.ExcelWriter('MultiMetricMultiResolutionInfo.xlsx') as writer:
            #Get both transformation types
            for key,value in final_dict.items():
                #Write sheet
                value.to_excel(writer,sheet_name=str(key))
