#Command line implementation for HDIreg module using command line input
#Developer: Joshua M. Hess, BSc
#Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

#Import custom modules
from miaaim.cli.reg import _parse
from miaaim.reg.ilastik import _yaml_reg

#Parse the command line arguments
args = _parse.ParseHDIregistrationYAML()

#Run the elastix registration function
_yaml_reg.RunHDIregistrationYAML(**args)
