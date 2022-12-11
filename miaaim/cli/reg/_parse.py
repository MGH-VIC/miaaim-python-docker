#Functions for parsing command line arguments elastix for HDIreg
#Developer: Joshua M. Hess, BSc
#Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

#Import external modules
import argparse


def ParseCommandElastix():
   """Function for parsing command line arguments for input to elastix
   """

#if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--fixed')
   parser.add_argument('--moving')
   parser.add_argument('--out_dir')
   parser.add_argument('--p', nargs='*')
   parser.add_argument('--fp')
   parser.add_argument('--mp')
   parser.add_argument('--fMask')
   args = parser.parse_args()
   #Create a dictionary object to pass to the next function
   dict = {'fixed': args.fixed, 'moving': args.moving, 'out_dir': args.out_dir,\
   'p': args.p, 'fp': args.fp, 'mp': args.mp, 'fMask': args.fMask}
   #Print the dictionary object
   print(dict)
   #Return the dictionary
   return dict

def ParseCommandTransformix():
   """Function for parsing command line arguments for input to transformix
   """

#if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--in_im')
   parser.add_argument('--out_dir')
   parser.add_argument('--tps', nargs='*')
   parser.add_argument('--target_size', type=int, nargs='*')
   parser.add_argument('--pad', type=int, nargs='*')
   parser.add_argument('--trim', type=int)
   parser.add_argument('--crops')
   parser.add_argument('--out_ext')
   args = parser.parse_args()
   #Create a dictionary object to pass to the next function
   dict = {'in_im': args.in_im, 'out_dir': args.out_dir, 'tps': args.tps,\
   'target_size': args.target_size, 'pad': args.pad, 'trim': args.trim,\
   'crops': args.crops, 'out_ext': args.out_ext}
   #Print the dictionary object
   print(dict)
   #Return the dictionary
   return dict


def ParseCommandElastixYAML():
    """Function for parsing command line arguments for input to YAML HDIprep"""

    # if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed")
    parser.add_argument("--moving")
    parser.add_argument("--path_to_yaml")
    parser.add_argument("--out_dir")
    args = parser.parse_args()
    # Create a dictionary object to pass to the next function
    dict = {'fixed': args.fixed, 'moving': args.moving,\
    "path_to_yaml": args.path_to_yaml, "out_dir": args.out_dir}
    # Print the dictionary object
    print(dict)
    # Return the dictionary
    return dict
