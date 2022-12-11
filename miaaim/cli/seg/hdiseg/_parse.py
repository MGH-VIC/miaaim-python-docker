#Functions for parsing command line arguments for ome ilastik prep
import argparse


def ParseHDIsegYAML():
   """Function for parsing command line arguments for input to ilastik YAML
   prep"""

#if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--pars')

   args = parser.parse_args()

   # create dictionary
   dict = {'pars': args.pars}
   print(dict)
   return dict
