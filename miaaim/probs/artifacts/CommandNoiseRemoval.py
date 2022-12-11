#Script for parsing command line arguments and running noise removal and segmentation
import ParseInput
import NoiseRemoval

#Parse the command line arguments
args = ParseInput.ParseInputNoiseRemoval()

#Run the MultiIlastikOMEPrep function
NoiseRemoval.MultiRemoveArtifacts(**args)
