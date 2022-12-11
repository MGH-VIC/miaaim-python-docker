# Module for high-dimensional image registration
# Developer: Joshua M. Hess, BSc
# Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital


__author__ = ", ".join(["Joshua Hess"])
__maintainer__ = ", ".join(["Joshua Hess"])
__version__ = "0.0.2"
__email__ = "jmh4003@med.cornell.edu.org"

try:
    from importlib_metadata import version  # Python < 3.8
except ImportError:
    from importlib.metadata import version  # Python = 3.8

from packaging.version import parse

try:
    __full_version__ = parse(version(__name__))
    __full_version__ = (
        f"{__version__}+{__full_version__.local}"
        if __full_version__.local
        else __version__
    )
except ImportError:
    __full_version__ = __version__

del version, parse
