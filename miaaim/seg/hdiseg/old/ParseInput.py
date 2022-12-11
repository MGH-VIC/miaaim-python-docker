#Functions for parsing command line arguments for noise removal and segmentation
import argparse
import numpy as np

def ParseInputNoiseRemoval():
   """Function for parsing command line arguments for input to noise removal"""

#if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--input',nargs='*')
   parser.add_argument('--output',nargs='*')
   parser.add_argument('--sigma', default=None)
   parser.add_argument('--correction',default=None)
   args = parser.parse_args()
   #Create a dictionary object to pass to the next function
   dict = {'input': args.input, 'output': args.output, 'sigma': args.sigma,\
      'correction':args.correction}
   #Print the dictionary object
   print(dict)
   #Return the dictionary
   return dict


def ParseInputTripletSegmentation():
   """Function for parsing command line arguments for input to watershed segmentation
   pipeline used for triplet melanoma dataset"""

#if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--input',nargs='*', help="enter path to images with spaces between each image (Ex: /path1/image1.ome.tiff /path2/image2.ome.tiff)")
   parser.add_argument('--output',nargs='*')
   parser.add_argument('--nuc_channel',type=int)
   parser.add_argument('--sigma',type=int)
   parser.add_argument('--correction',default=None)
   parser.add_argument('--area_threshold',type=int)
   parser.add_argument('--min_size',type=int)
   parser.add_argument('--footprint',type=int)
   args = parser.parse_args()
   #Create a dictionary object to pass to the next function
   dict = {'input': args.input, 'output': args.output,\
        'nuc_channel':args.nuc_channel, 'sigma':args.sigma,\
        'correction':args.correction, 'area_threshold':args.area_threshold,\
        'min_size':args.min_size, 'footprint':np.ones((args.footprint,args.footprint))}
   #Print the dictionary object
   print(dict)
   #Return the dictionary
   return dict
