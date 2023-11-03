#Script for parsing command line arguments and running noise removal and segmentation
import ParseInput
import WatershedSegmentation

#Parse the command line arguments
args = ParseInput.ParseInputTripletSegmentation()

#Run the MultiTripletPipeline function
WatershedSegmentation.MultiTripletPipeline(**args)
