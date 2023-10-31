from miaaim.cli.external.mcmicro.quant import _parse
from miaaim.external.mcmicro.quant import _sc_quant

#Parse the command line arguments
args = _parse.ParseInputDataExtract()

#Run the MultiExtractSingleCells function
_sc_quant.MultiExtractSingleCells(**args)
