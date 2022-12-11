#Command line implementation for HDIreg module using command line input
#Developer: Joshua M. Hess, BSc
#Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

#Import custom modules
from miaaim.cli.reg import _parse
from miaaim.cli.reg import _yaml_elastix
from miaaim.reg import _elastix

#Parse the command line arguments
args = _parse.ParseCommandElastix()

#Run the elastix registration function
_elastix.Elastix(**args)
