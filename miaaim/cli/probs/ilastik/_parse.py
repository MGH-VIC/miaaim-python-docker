#Functions for parsing command line arguments for ome ilastik prep
import argparse


def ParseIlastikYAML():
   """Function for parsing command line arguments for input to ilastik YAML
   prep"""

#if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--pars')
   parser.add_argument('--force_export', action='store_true',default = False)

   args = parser.parse_args()

   #Adjustment to account for user-facing 1-based indexing and the 0-based Python implementation
   # if args.nuclei_index != None:
   #    nuc_idx = args.nuclei_index-1
   # else:
   #    nuc_idx = None
   # if args.channelIDs != None:
   #    chIDs = [x-1 for x in args.channelIDs]
   # else:
   #    chIDs = None

   # create dictionary
   dict = {'pars': args.pars, 'force_export': args.force_export}
   print(dict)
   return dict
