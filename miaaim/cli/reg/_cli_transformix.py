#Command line implementation for HDIreg module using command line input
#Developer: Joshua M. Hess, BSc
#Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

#Import custom modules
from miaaim.cli.reg import _parse
from miaaim.reg import _transformix

#Parse the command line arguments
args = _parse.ParseCommandTransformix()

#Run the elastix registration function
_transformix.Transformix(**args)
