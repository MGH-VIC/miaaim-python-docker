#Functions for parsing command line arguments for quantificaton
import argparse


def ParseHDIquantYAML():
   """Function for parsing command line arguments for input to quantification"""

#if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--pars')

   args = parser.parse_args()

   # create dictionary
   dict = {'pars': args.pars}
   print(dict)
   return dict
