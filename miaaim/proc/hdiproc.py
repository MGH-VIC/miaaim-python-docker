# Class for merging data within a modality
# Developer: Joshua M. Hess, BSc
# Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

# Import external moduless
from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
import scipy.sparse
import skimage
import seaborn as sns
from sklearn.utils import check_random_state, extmath, check_array
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import MiniBatchKMeans
from scipy.optimize import curve_fit
from ast import literal_eval
from operator import itemgetter
import uncertainties.unumpy as unp
import uncertainties as unc
import yaml
import logging
import warnings
logging.captureWarnings(True)

import copy

# Import miaaim module
import miaaim
from miaaim.io.imread import _import, _prep_bridger
from miaaim.io.imread import _imzml_reader
from miaaim.io.imwrite import _export

# Import custom modules
from miaaim.proc._fuzzy_ops import FuzzySetCrossEntropy
from miaaim.proc._morph import MedFilter, Opening, Closing, NonzeroSlice, Thresholding, MorphFill, ConvexHull, MaskBoundary
from miaaim.proc._utils import Exp, CreateHyperspectralImage, CreateHyperspectralImageRectangular, ExportNifti, Export1
# for logging purposes
from miaaim.cli.proc import _parse

def find_ab_params(spread, min_dist):

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]

def check_base_object(base):

    # Handle all the optional plotting arguments, setting default
    if base.a is None or base.b is None:
        base._a, base._b = find_ab_params(base.spread, base.min_dist)
    else:
        base._a = base.a
        base._b = base.b
    if isinstance(base.init, np.ndarray):
        init = check_array(base.init, dtype=np.float32, accept_sparse=False)
    else:
        init = base.init
    base._initial_alpha = base.learning_rate
    base._validate_parameters()
    # return checked umap object
    return base

def simplicial_set_embedding_HDIprep(base):

    alt_embed,_ = umap.umap_.simplicial_set_embedding(
        data=base._raw_data,
        graph=base.graph_,
        n_components=base.n_components,
        initial_alpha=base._initial_alpha,
        a=base._a,
        b=base._b,
        gamma=base.repulsion_strength,
        negative_sample_rate=base.negative_sample_rate,  # Default umap behavior is n_epochs None -- converts to 0
        n_epochs=200,
        init=base.init,
        random_state=check_random_state(base.random_state),
        metric=base._input_distance_func,
        metric_kwds=base._metric_kwds,
        densmap=False,
        densmap_kwds={},
        output_dens=False,
        output_metric=base._output_distance_func,
        output_metric_kwds=base._output_metric_kwds,
        euclidean_output=base.output_metric in ("euclidean", "l2"),
        parallel=base.random_state is None,
        verbose=base.verbose,
    )
    # return embedding
    return alt_embed


class DeepCopyHDIreader(_import.HDIreader):
    """Helper class to deep copy HDIreader object for subsampled
    dimension reduction and projection on new data.

    Deep copy used to initialize HDIreader object using PrepBridge
    without having to write intermediate image and re-read.
    """
    def __init__(self, source=None):
        if source is not None:
            self.__dict__.update(copy.deepcopy(source.__dict__))


# Create a class for storing multiple datasets for a single modality
class IntraModalityDataset:
    """Merge HDIreader classes storing imaging datasets.

    Parameters
    ----------
    list_of_HDIimports: list of length (n_samples)
        Merges input HDIreader objects to be merged into single class.

    Returns
    -------
    Initialized class objects:

    * self.set_dict: dictionary
        Dictionary containing each input samples filename as the key.

    * self.umap_object: object of class UMAP
        Stores any UMAP class objects after running UMAP.

    * self.umap_embeddings: dictionary
        Dictionary storing UMAP embeddings for each input sample.

    * self.umap_optimal_dim: integer
        Specifies steady state embedding dimensionality for UMAP.

    * self.processed_images_export: None or dictionary after ``ExportNifti1``
        Dictionary that links input file names with new export file names.

    * self.landmarks: integer
        Specifies number of landmarks to use for steady state embedding dimensionality
        estimation.
    """

    # Create initialization
    def __init__(self, list_of_HDIimports, qc=True):

        # Create objects
        self.set_dict = {}
        self.umap_object = None
        self.umap_embeddings = {}
        self.umap_optimal_dim = None
        self.processed_images_export = {}
        self.masks_export = {}
        self.landmarks = None
        self.qc = qc
        self.yaml_log = {}

        # create list to store filenames
        filenames = []
        # Iterate through the list of HDIimports and add them to the set dictionary
        for dat in list_of_HDIimports:
            # Update the dictionary with keys being filenames
            self.set_dict.update({dat.hdi.data.filename: dat})
            # get named information from the hdi data object for import option
            updater_ = dat.import_kwargs
            # update filenames
            filenames.append(str(dat.hdi.data.filename.parent))

        # update logger with version number of miaaim
        self.yaml_log.update({"VERSION":miaaim.__version__})

        # update yaml logger
        self.yaml_log.update({"ImportOptions":{'paths':filenames,
                                                'qc':qc}})
        self.yaml_log["ImportOptions"].update(updater_)

        # update yaml_log with processing steps (initialize to empty)
        self.yaml_log.update({'ProcessingSteps':[]})

    def _reduce_reload(self,subsample,method, **kwargs):
        """Helper function to reload data using fully prescribed keyword arguments.
        """
        # Iterate through the set dictionary
        for f, hdi_imp in self.set_dict.items():
            #### TEMPORARY ####
            # check if there is any processed image at all
            if hdi_imp.hdi.data.processed_image is not None:
                # reload data using processed mask
                bridger = _import.HDIreader(data=hdi_imp.hdi.data,
                        image=hdi_imp.hdi.data.processed_image,
                        mask=hdi_imp.hdi.data.processed_mask,
                        channels=hdi_imp.hdi.data.channels,
                        flatten=True,   # set flatten to true for reload
                        filename=hdi_imp.hdi.data.filename,
                        subsample=subsample,
                        method=method,
                        **kwargs
    
                )
                # update the set dictionary with new data loaded
                self.set_dict[f] = bridger
            else:
                # leave the same as it was
                self.set_dict[f] = self.set_dict[f]

    def _validate_reduce(self,subsample=True,method='default',**kwargs):
        # list of potential steps
        red_commands = ["RunUMAP",
                        "RunOptimalUMAP",
                        "RunParametricUMAP",
                        "RunOptimalParametricUMAP"
                        ]
        # reload
        logging.info('Reloading using processed mask for dimension reduction')
        self._reduce_reload(subsample=subsample,method=method,**kwargs)

    # Create dimension reduction method with UMAP
    def RunUMAP(self, import_args={'subsample':True,'method':'default'},channels=None, **kwargs):
        """Creates an embedding of high-dimensional imaging data. Each
        pixel will be represented by its coordinates in the UMAP projection
        space.

        Parameters
        ----------
        kwargs: arguments passed to UMAP.
            Important arguments:

            * n_neighbors: integer
                Specifies number of nearest neighbors.

            * random_state: integer
                Specifies random state for reproducibility.

        Returns
        -------
        self.umap_embeddings: dictionary
            Stores umap coordinates for each input file as the dictionary key.
        """

        # update yaml logger
        self.yaml_log['ProcessingSteps'].append({"RunUMAP":{'import_args':{**import_args},
                                        'channels':channels,
                                        **kwargs}})

        # validate reload commands
        self._validate_reduce(**import_args)
        
        # check for deep slicing of images (down to pixel table level)
        if channels is not None:
            # deep slice
            self.DeepSlice(channels=channels)

        # Create a dictionary to store indices in
        file_idx = {}
        # Create a counter
        idx = 0
        # Create a list to store data tables in
        pixel_list = []
        # Create a blank frame
        tmp_frame = pd.DataFrame()

        # Iterate through the set dictionary
        for f, hdi_imp in self.set_dict.items():
            # Get the number of rows in the spectrum table
            nrows = hdi_imp.hdi.data.pixel_table.shape[0]
            # update the list of concatenation indices with filename
            file_idx.update({f: (idx, idx + nrows)})
            # Update the index
            idx = idx + nrows

            # Get the spectrum
            tmp_frame = pd.concat([tmp_frame, hdi_imp.hdi.data.pixel_table])
            # Clear the old pixel table from memory
            # hdi_imp.hdi.data.pixel_table = None

        # Set up UMAP parameters
        base = umap.UMAP(transform_mode="graph",**kwargs).fit(tmp_frame)

        # Handle all the optional plotting arguments, setting default
        base = check_base_object(base)

        # Print update for this dimension
        logging.info("Embedding in dimension " + str(base.n_components))
        # Use previous simplicial set and embedding components to embed in higher dimension
        alt_embed = simplicial_set_embedding_HDIprep(base)
        # update embedding
        base.embedding_ = alt_embed
        # Unravel the UMAP embedding for each sample
        for f, tup in file_idx.items():

            # Check to see if file has subsampling
            if self.set_dict[f].hdi.data.sub_coordinates is not None:
                # Extract the corresponding index from  UMAP embedding with subsample coordinates
                self.umap_embeddings.update(
                    {
                        f: pd.DataFrame(
                            base.embedding_[tup[0] : tup[1], :],
                            index=self.set_dict[f].hdi.data.sub_coordinates,
                        )
                    }
                )
            else:
                # Otherwise use the full coordinates list
                self.umap_embeddings.update(
                    {
                        f: pd.DataFrame(
                            base.embedding_[tup[0] : tup[1], :],
                            index=self.set_dict[f].hdi.data.coordinates,
                        )
                    }
                )
                # Here, ensure that the appropriate order for the embedding is given (c-style...imzml parser is fortran)
                self.umap_embeddings[f] = self.umap_embeddings[f].reindex(
                    sorted(list(self.umap_embeddings[f].index), key=itemgetter(1, 0))
                )

        # Update the transform mode
        base.transform_mode = "embedding"
        # Add the umap object to the class
        self.umap_object = base

    # Create dimension reduction method with UMAP
    def RunParametricUMAP(self, import_args={'subsample':True,'method':'default'}, channels=None, **kwargs):
        """Creates an embedding of high-dimensional imaging data using
        UMAP parametrized by neural network. Each
        pixel will be represented by its coordinates in the UMAP projection
        space.

        Parameters
        ----------
        kwargs: key word arguments passed to UMAP.
            Important arguments:

            * n_neighbors: integer
                Specifies number of nearest neighbors.

            * random_state: integer
                Specifies random state for reproducibility.

        Returns
        -------
        self.umap_embeddings: dictionary
            Stores umap coordinates for each input file as the dictionary key.
        """

        # update yaml logger
        self.yaml_log['ProcessingSteps'].append({"RunParametricUMAP":{'import_args':{**import_args},
                                        'channels':channels,
                                        **kwargs}})

        # validate reload commands
        self._validate_reduce(**import_args)
        
        # check for deep slicing of images (down to pixel table level)
        if channels is not None:
            # deep slice
            self.DeepSlice(channels=channels)

        # Create a dictionary to store indices in
        file_idx = {}
        # Create a counter
        idx = 0
        # Create a list to store data tables in
        pixel_list = []
        # Create a blank frame
        tmp_frame = pd.DataFrame()

        # Iterate through the set dictionary
        for f, hdi_imp in self.set_dict.items():
            # Get the number of rows in the spectrum table
            nrows = hdi_imp.hdi.data.pixel_table.shape[0]
            # update the list of concatenation indices with filename
            file_idx.update({f: (idx, idx + nrows)})
            # Update the index
            idx = idx + nrows

            # Get the spectrum
            tmp_frame = pd.concat([tmp_frame, hdi_imp.hdi.data.pixel_table])

        # run parametric umap with no spectral landmark selection
        base = umap.parametric_umap.ParametricUMAP(transform_mode="embedding",**kwargs).fit(tmp_frame)

        # Unravel the UMAP embedding for each sample
        for f, tup in file_idx.items():

            # Check to see if file has subsampling
            if self.set_dict[f].hdi.data.sub_coordinates is not None:
                # Extract the corresponding index from  UMAP embedding with subsample coordinates
                self.umap_embeddings.update(
                    {
                        f: pd.DataFrame(
                            base.embedding_[tup[0] : tup[1], :],
                            index=self.set_dict[f].hdi.data.sub_coordinates,
                        )
                    }
                )
            else:
                # Otherwise use the full coordinates list
                self.umap_embeddings.update(
                    {
                        f: pd.DataFrame(
                            base.embedding_[tup[0] : tup[1], :],
                            index=self.set_dict[f].hdi.data.coordinates,
                        )
                    }
                )
                # Here, ensure that the appropriate order for the embedding is given (c-style...imzml parser is fortran)
                self.umap_embeddings[f] = self.umap_embeddings[f].reindex(
                    sorted(list(self.umap_embeddings[f].index), key=itemgetter(1, 0))
                )

        # Add the umap object to the class
        self.umap_object = base


    def RunOptimalUMAP(
        self,
        import_args={'subsample':True,'method':'default'},
        dim_range=(1,11),
        landmarks=3000,
        export_diagnostics=False,
        output_dir=None,
        n_jobs=1,
        channels=None,
        **kwargs
    ):
        """Run UMAP over a range of dimensions to choose steady state embedding
        by fitting an exponential regression model to the fuzzy set cross entropy
        curve.

        Parameters
        ----------
        dim_range: tuple (low_dim, high_dim; Default: (1,11))
            Indicates a range of embedding dimensions.

        landmarks: integer (Default: 3000)
            Specifies number of landmarks to use for steady state embedding dimensionality
            estimation.

        export_diagnostics: Bool (Default: False)
            Indicates whether or not to export a csv file and jpeg image showing
            steady state embedding dimensionality reports. These report the
            normalized (0-1 range) fuzzy set cross entropy across the range
            of indicated dimensionalities.

        output_dir: string (Default: None)
            Path to export data to if exporting diagnostic images and plots.

        n_jobs: integer (Default: 1)
            Path to export data to if exporting diagnostic images and plots.

        kwargs: key word arguments passed to UMAP.
            Important arguments:

            * n_neighbors: integer
                Specifies number of nearest neighbors.

            * random_state: integer
                Specifies random state for reproducibility.
        """

        # update yaml logger
        self.yaml_log['ProcessingSteps'].append({"RunOptimalUMAP":{'import_args':{**import_args},
                                        'dim_range':str(dim_range),
                                        'landmarks':landmarks,
                                        'export_diagnostics':export_diagnostics,
                                        'output_dir':str(output_dir),
                                        'n_jobs':n_jobs,
                                        'channels':channels,
                                        **kwargs}})

        # validate reload commands
        self._validate_reduce(**import_args)

        # check for deep slicing of images (down to pixel table level)
        if channels is not None:
            # deep slice
            self.DeepSlice(channels=channels)

        # convert to tuple and back to pathlib after logging
        if isinstance(dim_range,str):
            dim_range = literal_eval(dim_range)

        output_dir = Path(output_dir) if output_dir is not None else None

        # check for landmarks
        self.landmarks = landmarks

        # Create a dictionary to store indices in
        file_idx = {}
        # Create a counter
        idx = 0
        # Create a list to store data tables in
        pixel_list = []
        # Create a blank frame
        tmp_frame = pd.DataFrame()

        # Iterate through the set dictionary
        for f, hdi_imp in self.set_dict.items():
            # Get the number of rows in the spectrum table
            nrows = hdi_imp.hdi.data.pixel_table.shape[0]
            # update the list of concatenation indices with filename
            file_idx.update({f: (idx, idx + nrows)})
            # Update the index
            idx = idx + nrows

            # Get the spectrum
            tmp_frame = pd.concat([tmp_frame, hdi_imp.hdi.data.pixel_table])

        # Create list to store the results in
        ce_res = {}
        # Create a dictionary to store the embeddings in
        embed_dict = {}

        # Check to see if the dim_range is a string
        if isinstance(dim_range, str):
            dim_range = literal_eval(dim_range)

        # Set up the dimension range for UMAP
        dim_range = range(dim_range[0], dim_range[1])

        # Print update
        logging.info(f"Computing UMAP simplicial set on {tmp_frame.shape[1]} image channels...")

        # Run UMAP on the first iteration -- we will skip simplicial set construction in next iterations
        base = umap.UMAP(transform_mode="graph",**kwargs).fit(tmp_frame)

        # Check for landmark subsampling
        if self.landmarks is not None:
                # Print update
                logging.info("Computing "+str(landmarks)+" spectral landmarks...")
                # Calculate singular value decomposition
                a, b, VT = extmath.randomized_svd(base.graph_,n_components=100,random_state=0)

                # Calculate spectral clustering
                kmeans = MiniBatchKMeans(self.landmarks,init_size=3 * self.landmarks,batch_size=10000,random_state=0)
                #Get kmeans labels using the singular value decomposition and minibatch k means
                kmean_lab = kmeans.fit_predict(base.graph_.dot(VT.T))
                # Get  mean values from clustering to define spectral centroids
                means = pd.concat([tmp_frame, pd.DataFrame(kmean_lab,columns=["ClusterID"], index=tmp_frame.index)],axis=1)
                # Get mean values from dataframe
                tmp_centroids = means.groupby("ClusterID").mean().values

                # Create simplicial set from centroided data
                base_centroids = umap.UMAP(transform_mode="graph",**kwargs).fit(tmp_centroids)

                # Handle all the optional plotting arguments, setting default
                base_centroids = check_base_object(base_centroids)

                # Iterate through each subsequent embedding dimension -- add +1 because we have already used range
                for dim in range(dim_range[0], dim_range[-1] + 1):
                    # adjust base number of components
                    base_centroids.n_components = dim
                    # Print update for this dimension
                    logging.info("Embedding in dimension " + str(dim))
                    # Use previous simplicial set and embedding components to embed in higher dimension
                    alt_embed = simplicial_set_embedding_HDIprep(base_centroids)
                    # Print update
                    logging.info("Finished embedding")

                    # Update the embedding dictionary
                    embed_dict.update({dim: alt_embed})

                    # Compute the fuzzy set cross entropy
                    cs = FuzzySetCrossEntropy(
                        alt_embed, base_centroids.graph_, base_centroids.min_dist, n_jobs
                    )
                    # Update list for now
                    ce_res.update({dim: cs})

                # Construct a dataframe from the dictionary of results
                ce_res = pd.DataFrame(ce_res, index=["Cross-Entropy"]).T
                # set base centroids to 0 to save memory
                base_centroids = 0

        else:
                # Iterate through each subsequent embedding dimension -- add +1 because we have already used range
                for dim in range(dim_range[0], dim_range[-1] + 1):
                    # adjust base number of components
                    base.n_components = dim
                    # Print update for this dimension
                    logging.info("Embedding in dimension " + str(dim))
                    # Use previous simplicial set and embedding components to embed in higher dimension
                    alt_embed = simplicial_set_embedding_HDIprep(base)
                    # Print update
                    logging.info("Finished embedding")

                    # Update the embedding dictionary
                    embed_dict.update({dim: alt_embed})

                    # Compute the fuzzy set cross entropy
                    cs = FuzzySetCrossEntropy(
                        alt_embed, base.graph_, base.min_dist, n_jobs
                    )
                    # Update list for now
                    ce_res.update({dim: cs})

                # Construct a dataframe from the dictionary of results
                ce_res = pd.DataFrame(ce_res, index=["Cross-Entropy"]).T

        # Print update
        logging.info("Finding optimal embedding dimension through exponential fit...")
        # Calculate the min-max normalized cross-entropy
        ce_res_norm = MinMaxScaler().fit_transform(ce_res)
        # Convert back to pandas dataframe
        ce_res_norm = pd.DataFrame(
            ce_res_norm, columns=["Scaled Cross-Entropy"], index=[x for x in dim_range]
        )

        # Get the metric values
        met = ce_res_norm["Scaled Cross-Entropy"].values
        # Get the x axis information
        xdata = np.int64(ce_res_norm.index.values)
        # Fit the data using exponential function
        popt, pcov = curve_fit(Exp, xdata, met, p0=(0, 0.01, 1))

        # create parameters from scipy fit
        a, b, c = unc.correlated_values(popt, pcov)

        # Create a tuple indicating the 95% interval containing the asymptote in c
        asympt = (c.n - c.s, c.n + c.s)

        # create equally spaced samples between range of dimensions
        px = np.linspace(dim_range[0], dim_range[-1] + 1, 100000)

        # use unumpy.exp to create samples
        py = a * unp.exp(-b * px) + c
        # extract expected values
        nom = unp.nominal_values(py)
        # extract stds
        std = unp.std_devs(py)

        # Iterate through samples to find the instance that value falls in 95% c value
        for val in range(len(py)):
            # Extract the nominal value
            tmp_nom = py[val].n
            # check if nominal value falls within 95% CI for asymptote
            if asympt[0] <= tmp_nom <= asympt[1]:
                # break the loop
                break
        # Extract the nominal value at this index -- round up (any float value lower is not observed -- dimensions are int)
        opt_dim = int(np.ceil(px[val]))
        # Print update
        logging.info("Optimal UMAP embedding dimension is " + str(opt_dim))

        # Check to see if exporting plot
        if export_diagnostics:
            # Ensure that an output directory is entered
            if output_dir is None:
                # Raise and error if no output
                raise (ValueError("Please add an output directory -- none identified"))
            # Create a path based on the output directory
            else:
                # Create image path
                im_path = Path(os.path.join(output_dir, "OptimalUMAP.jpeg"))
                # Create csv path
                csv_path = Path(os.path.join(output_dir, "OptimalUMAP.csv"))

            # Plot figure and save results
            fig, axs = plt.subplots()
            # plot the fit value
            axs.plot(px, nom, c="r", label="Fitted Curve", linewidth=3)
            # add 2 sigma uncertainty lines
            axs.plot(px, nom - 2 * std, c="c", label="95% CI", alpha=0.6, linewidth=3)
            axs.plot(px, nom + 2 * std, c="c", alpha=0.6, linewidth=3)
            # plot the observed values
            axs.plot(xdata, met, "ko", label="Observed Data", markersize=8)
            # Change axis names
            axs.set_xlabel("Dimension")
            axs.set_ylabel("Min-Max Scaled Cross-Entropy")
            fig.suptitle("Optimal Dimension Estimation", fontsize=12)
            axs.legend()
            # plt.show()
            plt.savefig(im_path, dpi=600)

            # Export the metric values to csv
            ce_res_norm.to_csv(csv_path)

        # set base component dimensionality
        base.n_components = opt_dim
        # check if landmarks
        if self.landmarks is not None:
            # implement umap on the tmp frame -- faster than centroids method
            base = check_base_object(base)
            # Use the optimal UMAP embedding to add to the class object
            base.embedding_ = simplicial_set_embedding_HDIprep(base)
        # otherwise update the embedding with the original
        else:
            base.embedding_ = embed_dict[opt_dim]
        # Update the transform mode
        base.transform_mode = "embedding"

        # Unravel the UMAP embedding for each sample
        for f, tup in file_idx.items():
            # Check to see if file has subsampling
            if self.set_dict[f].hdi.data.sub_coordinates is not None:
                # Extract the corresponding index from  UMAP embedding with subsample coordinates
                self.umap_embeddings.update(
                    {
                        f: pd.DataFrame(
                            base.embedding_[tup[0] : tup[1], :],
                            index=self.set_dict[f].hdi.data.sub_coordinates,
                        )
                    }
                )
            else:
                # Otherwise use the full coordinates list
                self.umap_embeddings.update(
                    {
                        f: pd.DataFrame(
                            base.embedding_[tup[0] : tup[1], :],
                            index=self.set_dict[f].hdi.data.coordinates,
                        )
                    }
                )
                # Here, ensure that the appropriate order for the embedding is given (c-style...imzml parser is fortran)
                self.umap_embeddings[f] = self.umap_embeddings[f].reindex(
                    sorted(list(self.umap_embeddings[f].index), key=itemgetter(1, 0))
                )

        # Add the umap object to the class
        self.umap_object = base
        # Update the optimal dimensionality
        self.umap_optimal_dim = opt_dim

    def RunOptimalParametricUMAP(
        self,
        import_args={'subsample':True,'method':'default'},
        dim_range=(1,11),
        landmarks=3000,
        export_diagnostics=False,
        output_dir=None,
        n_jobs=1,
        channels=None,
        **kwargs
    ):
        """Run parametric UMAP over a range of dimensions to choose steady state embedding
        by fitting an exponential regression model to the fuzzy set cross entropy
        curve.

        Parameters
        ----------
        dim_range: tuple (low_dim, high_dim; Default: (1,11))
            Indicates a range of embedding dimensions.

        landmarks: integer (Default: 3000)
            Specifies number of landmarks to use for steady state embedding dimensionality
            estimation.

        export_diagnostics: Bool (Default: False)
            Indicates whether or not to export a csv file and jpeg image showing
            steady state embedding dimensionality reports. These report the
            normalized (0-1 range) fuzzy set cross entropy across the range
            of indicated dimensionalities.

        output_dir: string (Default: None)
            Path to export data to if exporting diagnostic images and plots.

        n_jobs: integer (Default: 1)
            Path to export data to if exporting diagnostic images and plots.

        kwargs: key word arguments passed to UMAP.
            Important arguments:

            * n_neighbors: integer
                Specifies number of nearest neighbors.

            * random_state: integer
                Specifies random state for reproducibility.
        """

        # update yaml logger
        self.yaml_log['ProcessingSteps'].append({"RunOptimalParametricUMAP":{'import_args':{**import_args},
                                        'dim_range':str(dim_range),
                                        'landmarks':landmarks,
                                        'export_diagnostics':export_diagnostics,
                                        'output_dir':str(output_dir),
                                        'n_jobs':n_jobs,
                                        'channels':channels,
                                        **kwargs}})

        # validate reload commands
        self._validate_reduce(**import_args)

        # check for deep slicing of images (down to pixel table level)
        if channels is not None:
            # deep slice
            self.DeepSlice(channels=channels)

        # convert to tuple and back to pathlib after logging
        if isinstance(dim_range,str):
            dim_range = literal_eval(dim_range)

        output_dir = Path(output_dir) if output_dir is not None else None

        # check for landmarks
        self.landmarks = landmarks

        # Create a dictionary to store indices in
        file_idx = {}
        # Create a counter
        idx = 0
        # Create a list to store data tables in
        pixel_list = []
        # Create a blank frame
        tmp_frame = pd.DataFrame()

        # Iterate through the set dictionary
        for f, hdi_imp in self.set_dict.items():
            # Get the number of rows in the spectrum table
            nrows = hdi_imp.hdi.data.pixel_table.shape[0]
            # update the list of concatenation indices with filename
            file_idx.update({f: (idx, idx + nrows)})
            # Update the index
            idx = idx + nrows

            # Get the spectrum
            tmp_frame = pd.concat([tmp_frame, hdi_imp.hdi.data.pixel_table])

        # Create list to store the results in
        ce_res = {}
        # Create a dictionary to store the neural network models in
        model_dict = {}

        # Check to see if the dim_range is a string
        if isinstance(dim_range, str):
            dim_range = literal_eval(dim_range)

        # Set up the dimension range for UMAP
        dim_range = range(dim_range[0], dim_range[1])
        # Print update
        logging.info(f"Computing UMAP simplicial set on {tmp_frame.shape[1]} image channels...")

        # Check for landmark subsampling
        if self.landmarks is not None:
                # Run UMAP on the first iteration -- we will skip simplicial set construction in next iterations
                base = umap.parametric_umap.ParametricUMAP(transform_mode="graph",**kwargs).fit(tmp_frame)
                # Print update
                logging.info("Computing "+str(landmarks)+" spectral landmarks...")
                # Calculate singular value decomposition
                a, b, VT = extmath.randomized_svd(base.graph_,n_components=100,random_state=0)

                # Calculate spectral clustering
                kmeans = MiniBatchKMeans(self.landmarks,init_size=3 * self.landmarks,batch_size=10000,random_state=0)
                #Get kmeans labels using the singular value decomposition and minibatch k means
                kmean_lab = kmeans.fit_predict(base.graph_.dot(VT.T))
                # Get  mean values from clustering to define spectral centroids
                means = pd.concat([tmp_frame, pd.DataFrame(kmean_lab,columns=["ClusterID"], index=tmp_frame.index)],axis=1)
                # Get mean values from dataframe
                tmp_centroids = means.groupby("ClusterID").mean().values

                # Create simplicial set from centroided data
                base_centroids = umap.UMAP(transform_mode="graph",**kwargs).fit(tmp_centroids)

                # Handle all the optional plotting arguments, setting default
                base_centroids = check_base_object(base_centroids)

                # Iterate through each subsequent embedding dimension -- add +1 because we have already used range
                for dim in range(dim_range[0], dim_range[-1] + 1):
                    # adjust base number of components
                    base_centroids.n_components = dim
                    # Print update for this dimension
                    logging.info("Embedding in dimension " + str(dim))
                    # Use previous simplicial set and embedding components to embed in higher dimension
                    alt_embed = simplicial_set_embedding_HDIprep(base_centroids)
                    # Print update
                    logging.info("Finished embedding")

                    # Compute the fuzzy set cross entropy
                    cs = FuzzySetCrossEntropy(
                        alt_embed, base_centroids.graph_, base_centroids.min_dist, n_jobs
                    )
                    # Update list for now
                    ce_res.update({dim: cs})

                # Construct a dataframe from the dictionary of results
                ce_res = pd.DataFrame(ce_res, index=["Cross-Entropy"]).T
        else:
                # Iterate through each subsequent embedding dimension -- add +1 because we have already used range
                for dim in range(dim_range[0], dim_range[-1] + 1):

                    # Print update for this dimension
                    logging.info("Embedding in dimension " + str(dim))
                    # Use previous simplicial set and embedding components to embed in higher dimension
                    base = umap.parametric_umap.ParametricUMAP(transform_mode="embedding",n_components=dim,**kwargs).fit(tmp_frame)
                    # Print update
                    logging.info("Finished embedding")

                    # Compute the fuzzy set cross entropy
                    cs = FuzzySetCrossEntropy(
                        base.embedding_, base.graph_, base.min_dist, n_jobs
                    )
                    # Update list for now
                    ce_res.update({dim: cs})
                    #update the model dictionary
                    model_dict.update({dim: base})

                # Construct a dataframe from the dictionary of results
                ce_res = pd.DataFrame(ce_res, index=["Cross-Entropy"]).T

        # Print update
        logging.info("Finding optimal embedding dimension through exponential fit...")
        # Calculate the min-max normalized cross-entropy
        ce_res_norm = MinMaxScaler().fit_transform(ce_res)
        # Convert back to pandas dataframe
        ce_res_norm = pd.DataFrame(
            ce_res_norm, columns=["Scaled Cross-Entropy"], index=[x for x in dim_range]
        )

        # Get the metric values
        met = ce_res_norm["Scaled Cross-Entropy"].values
        # Get the x axis information
        xdata = np.int64(ce_res_norm.index.values)
        # Fit the data using exponential function
        popt, pcov = curve_fit(Exp, xdata, met, p0=(0, 0.01, 1))

        # create parameters from scipy fit
        a, b, c = unc.correlated_values(popt, pcov)

        # Create a tuple indicating the 95% interval containing the asymptote in c
        asympt = (c.n - c.s, c.n + c.s)

        # create equally spaced samples between range of dimensions
        px = np.linspace(dim_range[0], dim_range[-1] + 1, 100000)

        # use unumpy.exp to create samples
        py = a * unp.exp(-b * px) + c
        # extract expected values
        nom = unp.nominal_values(py)
        # extract stds
        std = unp.std_devs(py)

        # Iterate through samples to find the instance that value falls in 95% c value
        for val in range(len(py)):
            # Extract the nominal value
            tmp_nom = py[val].n
            # check if nominal value falls within 95% CI for asymptote
            if asympt[0] <= tmp_nom <= asympt[1]:
                # break the loop
                break
        # Extract the nominal value at this index -- round up (any float value lower is not observed -- dimensions are int)
        opt_dim = int(np.ceil(px[val]))
        # Print update
        logging.info("Optimal UMAP embedding dimension is " + str(opt_dim))

        # Check to see if exporting plot
        if export_diagnostics:
            # Ensure that an output directory is entered
            if output_dir is None:
                # Raise and error if no output
                raise (ValueError("Please add an output directory -- none identified"))
            # Create a path based on the output directory
            else:
                # Create image path
                im_path = Path(os.path.join(output_dir, "OptimalUMAP.jpeg"))
                # Create csv path
                csv_path = Path(os.path.join(output_dir, "OptimalUMAP.csv"))

            # Plot figure and save results
            fig, axs = plt.subplots()
            # plot the fit value
            axs.plot(px, nom, c="r", label="Fitted Curve", linewidth=3)
            # add 2 sigma uncertainty lines
            axs.plot(px, nom - 2 * std, c="c", label="95% CI", alpha=0.6, linewidth=3)
            axs.plot(px, nom + 2 * std, c="c", alpha=0.6, linewidth=3)
            # plot the observed values
            axs.plot(xdata, met, "ko", label="Observed Data", markersize=8)
            # Change axis names
            axs.set_xlabel("Dimension")
            axs.set_ylabel("Min-Max Scaled Cross-Entropy")
            fig.suptitle("Optimal Dimension Estimation", fontsize=12)
            axs.legend()
            # plt.show()
            plt.savefig(im_path, dpi=600)

            # Export the metric values to csv
            ce_res_norm.to_csv(csv_path)

        #check if landmarks
        if self.landmarks is not None:
            # implement parametric umap on optimal dimensionality
            base = umap.parametric_umap.ParametricUMAP(transform_mode="embedding",n_components=opt_dim,**kwargs).fit(tmp_frame)
        # otherwise fill in with existing model
        else:
            # Use the optimal UMAP embedding to add to the class object
            base = model_dict[opt_dim]

        # Unravel the UMAP embedding for each sample
        for f, tup in file_idx.items():

            # Check to see if file has subsampling
            if self.set_dict[f].hdi.data.sub_coordinates is not None:
                # Extract the corresponding index from  UMAP embedding with subsample coordinates
                self.umap_embeddings.update(
                    {
                        f: pd.DataFrame(
                            base.embedding_[tup[0] : tup[1], :],
                            index=self.set_dict[f].hdi.data.sub_coordinates,
                        )
                    }
                )
            else:
                # Otherwise use the full coordinates list
                self.umap_embeddings.update(
                    {
                        f: pd.DataFrame(
                            base.embedding_[tup[0] : tup[1], :],
                            index=self.set_dict[f].hdi.data.coordinates,
                        )
                    }
                )
                # Here, ensure that the appropriate order for the embedding is given (c-style...imzml parser is fortran)
                self.umap_embeddings[f] = self.umap_embeddings[f].reindex(
                    sorted(list(self.umap_embeddings[f].index), key=itemgetter(1, 0))
                )

        # Add the umap object to the class
        self.umap_object = base
        # Update the optimal dimensionality
        self.umap_optimal_dim = opt_dim

    # Add function for creating hyperspectral image from UMAP
    def SpatiallyMapUMAP(self,method="rectangular",save_mem=True):
        """Map UMAP projections into the spatial domain (2-dimensional) using
        each pixel's original XY positions.

        Parameters
        ----------
        method: string (Default: "rectangular")
            Type of mapping to use for reconstructing an image from the UMAP
            embeddings.

            Options include:

            * "rectangular"
                Use for images that do not have an associated mask with them. This
                is the fastest option for spatial reconstruction.

            * "coordinate"
                Use each pixel's XY coordinate to fill an array one pixel at a
                time. This must be used for images that contain masks or are
                not stored as rectangular arrays.

        save_mem: bool (Default: True)
            Save memory by deleting reserves of full images and intermediate steps.
        """

        # update yaml logger
        self.yaml_log['ProcessingSteps'].append({"SpatiallyMapUMAP":{'method':method,
                                             'save_mem':save_mem}})

        # Check to make sure that UMAP object in class is not empty
        if self.umap_object is None:
            # Raise an error
            raise ValueError(
                "Spatially mapping an embedding is not possible yet! Please run UMAP first."
            )

        # For now, create a dictionary to store the results in
        results_dict = {}

        # Run through each object in the set dictionary
        for f, locs in self.umap_embeddings.items():

            logging.info("working on " + str(f) + "...")

            # Check to see if there is subsampling
            if self.set_dict[f].hdi.data.sub_coordinates is not None:

                # Get the inverse pixels
                inv_pix = list(
                    set(self.set_dict[f].hdi.data.coordinates).difference(
                        set(list(locs.index))
                    )
                )

                # check for saving memory
                if save_mem:
                    # remove pixel unncessary portions of stored image
                    self.set_dict[f].hdi.data.pixel_table = None
                    self.set_dict[f].hdi.data.coordinates = None
                    self.set_dict[f].hdi.data.sub_coordinates = None

                # Create a mask based off array size and current UMAP data points
                data = np.ones(len(inv_pix), dtype=np.bool_)
                # Create row data for scipy coo matrix (-1 index for 0-based python)
                row = np.array([inv_pix[c][1] - 1 for c in range(len(inv_pix))])
                # Create row data for scipy coo matrix (-1 index for 0-based python)
                col = np.array([inv_pix[c][0] - 1 for c in range(len(inv_pix))])

                # Create a sparse mask from data and row column indices
                sub_mask = scipy.sparse.coo_matrix(
                    (data, (row, col)), shape=self.set_dict[f].hdi.data.array_size
                )

                # Remove the other objects used to create the mask to save memory
                data, row, col, inv_pix = None, None, None, None

                # Read the file and use the mask to create complementary set of pixels
                # new_data = _import.HDIreader(
                #     path_to_data=f,
                #     path_to_markers=None,
                #     flatten=True,
                #     subsample=None,
                #     mask=sub_mask,
                #     save_mem=True
                # )

                # get corresponding hdi import
                hdi_imp=DeepCopyHDIreader(self.set_dict[f])
                # create temporary new data
                new_data = _import.HDIreader(
                    data=hdi_imp.hdi.data,
                    image=hdi_imp.hdi.data.image,
                    mask=sub_mask,
                    channels=hdi_imp.hdi.data.channels,
                    flatten=True,   # set flatten to true for reload
                    filename=hdi_imp.hdi.data.filename,
                    subsample=None,
                    method=None,
                    save_mem=True
                )

                # Remove the mask to save memory
                sub_mask = None

                # print update
                logging.info("Transforming pixels into existing UMAP embedding of subsampled pixels...")
                # Run the new pixel table through umap transformer
                embedding_projection = self.umap_object.transform(
                    new_data.hdi.data.pixel_table
                )
                # Add the projection to dataframe and coerce with existing embedding
                embedding_projection = pd.DataFrame(
                    embedding_projection,
                    index=list(new_data.hdi.data.pixel_table.index),
                )

                # Remove the new data to save memory
                new_data = None

                # Concatenate with existing UMAP object
                self.umap_embeddings[f] = pd.concat([locs, embedding_projection])

                # save memory do not store embedding twice
                embedding_projection = None

                # Reindex data frame to row major orientation
                self.umap_embeddings[f] = self.umap_embeddings[f].reindex(
                    sorted(list(self.umap_embeddings[f].index), key=itemgetter(1, 0))
                )

            # print update
            logging.info ('Reconstructing image...')
            # check for mask to use in reconstruction
            if method=="rectangular":
                # Use the new embedding to map coordinates to the image
                hyper_im = CreateHyperspectralImageRectangular(
                    embedding=self.umap_embeddings[f],
                    array_size=self.set_dict[f].hdi.data.array_size,
                    coordinates=list(self.umap_embeddings[f].index),
                )
            elif method=="coordinate":
                # use array reshaping (faster)
                hyper_im = CreateHyperspectralImage(
                    embedding=self.umap_embeddings[f],
                    array_size=self.set_dict[f].hdi.data.array_size,
                    coordinates=list(self.umap_embeddings[f].index),
                )
            else:
                raise(ValueError("Spatial reconstruction method not supported."))

            # Update list
            results_dict.update({f: hyper_im})

            # add this hyperspectral image to the hdi_import object as processed_image
            self.set_dict[f].hdi.data.processed_image = hyper_im

        # print update
        logging.info("Finished spatial mapping")

        # Return the resulting images
        # return results_dict

    # Create definition for image filtering and processing
    def ApplyManualMask(self):
        """Apply input mask to image. This function is
        primarily used on histology images and images that do not need dimension
        reduction. Dimension reduction with a mask will by default zero all other pixels
        in the image outside of the mask, but do not use this function if
        performing dimension reduction.
        """
        # update yaml logger
        self.yaml_log['ProcessingSteps'].append("ApplyManualMask")

        # Iterate through the set dictionary
        for f, hdi_imp in self.set_dict.items():
            # Ensure that the mask is not none
            if hdi_imp.hdi.data.mask is None:
                # Skip this image if there is no mask
                continue
            # Ensure that the image itself is not none
            if hdi_imp.hdi.data.image is None:
                # Skip this image if there is no mask
                continue

            # Check to see if the preprocessed is initiated
            if hdi_imp.hdi.data.processed_image is None:
                # If not, use the original image
                hdi_imp.hdi.data.processed_image = hdi_imp.hdi.data.image.copy()
                # Use the mask on the image
                hdi_imp.hdi.data.processed_image[~hdi_imp.hdi.data.mask.toarray()] = 0
                # go ahead and mask the image as well
                hdi_imp.hdi.data.image[~hdi_imp.hdi.data.mask.toarray()] = 0
            # Otherwise the processed image exists and now check the data type
            else:
                # Proceed to process the processed image as an array
                if isinstance(
                    hdi_imp.hdi.data.processed_image, scipy.sparse.coo_matrix
                ):
                    # Convert to array
                    hdi_imp.hdi.data.processed_image = (
                        hdi_imp.hdi.data.processed_image.toarray()
                    )

                    # Use the mask on the image
                    hdi_imp.hdi.data.processed_image[
                        ~hdi_imp.hdi.data.mask.toarray()
                    ] = 0
                    # Turn the processed mask back to sparse matrix
                    hdi_imp.hdi.data.processed_image = scipy.sparse.coo_matrix(
                        hdi_imp.hdi.data.processed_image, dtype=np.bool_
                    )

    def InvertMask(self):
        """Helper to invert a boolean mask
        """

        # update yaml logger
        self.yaml_log['ProcessingSteps'].append('InvertMask')

        # Iterate through the set dictionary
        for f, hdi_imp in self.set_dict.items():
            # Use the mask on the image
            hdi_imp.hdi.data.processed_image = ~hdi_imp.hdi.data.processed_image.toarray()
            # Turn the processed mask back to sparse matrix
            hdi_imp.hdi.data.processed_image = scipy.sparse.coo_matrix(
                hdi_imp.hdi.data.processed_image, dtype=np.bool_
            )

    def MedianFilter(self, filter_size, parallel=False):
        """Median filtering of images to remove salt and pepper noise.
        A circular disk is used for the filtering. Images that are not single channel
        are automatically converted to grayscale prior to filtering.

        Parameters
        ----------
        filter_size: integer
            Size of disk to use for the median filter.

        parallel: Bool (Default: False)
            Use parallel processing with all available CPUs.
        """

        # update yaml logger
        self.yaml_log['ProcessingSteps'].append({"MedianFilter":{'filter_size':filter_size,
                                         'parallel':parallel}})

        # Iterate through the set dictionary
        for f, hdi_imp in self.set_dict.items():
            # Ensure that the mask is not none
            if hdi_imp.hdi.data.image is None:
                # Skip this image if there is no mask
                continue

            # Check to see if the preprocessed is initiated
            if hdi_imp.hdi.data.processed_image is None:
                # If not, use the original image
                hdi_imp.hdi.data.processed_image = MedFilter(
                    hdi_imp.hdi.data.image, filter_size, parallel
                )

            # Use the initiated image
            else:
                # Use the processed image
                hdi_imp.hdi.data.processed_image = MedFilter(
                    hdi_imp.hdi.data.processed_image, filter_size, parallel
                )

    def Threshold(self, type="otsu", channel=None, thresh_value=None, correction=1.0):
        """Threshold grayscale images. Produces a sparse boolean
        mask.

        Parameters
        ----------
        type: string (Default: "otsu")
            Type of thresholding to use.

            Options include:

            * "otsu"
                Otsu automated thresholding.

            * "manual"
                Set manual threshold value.

        thresh_value: float (Default: None)
            Manual threshold to use if ``type`` is set to "manual"

        correction: float (Default: 1.0)
            Correct factor to multiply threshold by for more stringent thresholding.
        """

        # update yaml logger
        self.yaml_log['ProcessingSteps'].append({"Threshold":{'type':type,
                                     'channel':channel,
                                     'thresh_value':thresh_value,
                                     'correction':correction}})

        # Iterate through the set dictionary
        for f, hdi_imp in self.set_dict.items():
            # Ensure that the mask is not none
            if hdi_imp.hdi.data.image is None:
                # Skip this image if there is no mask
                continue

            # Check to see if the preprocessed is initiated
            if hdi_imp.hdi.data.processed_image is None:
                # If not, use the original image
                hdi_imp.hdi.data.processed_image = Thresholding(
                    hdi_imp.hdi.data.image, type, channel, thresh_value, correction
                )

            # Use the initiated image
            else:
                # Use the processed image
                hdi_imp.hdi.data.processed_image = Thresholding(
                    hdi_imp.hdi.data.processed_image, type, channel, thresh_value, correction
                )

    def Open(self, disk_size, parallel=False):
        """Morphological opening on boolean array (i.e., a mask).
        A circular disk is used for the filtering.

        Parameters
        ----------
        filter_size: integer
            Size of disk to use for the median filter.

        parallel: Bool (Default: False)
            Use parallel processing with all available CPUs.
        """

        # update yaml logger
        self.yaml_log['ProcessingSteps'].append({"Open":{'disk_size':disk_size,
                                'parallel':parallel}})

        # Iterate through the set dictionary
        for f, hdi_imp in self.set_dict.items():
            # Ensure that the mask is not none
            if hdi_imp.hdi.data.image is None:
                # Skip this image if there is no mask
                continue

            # Check to see if the preprocessed is initiated
            if hdi_imp.hdi.data.processed_image is None:
                # If not, use the original image
                hdi_imp.hdi.data.processed_image = Opening(
                    hdi_imp.hdi.data.image, disk_size, parallel
                )

            # Use the initiated image
            else:
                # Use the processed image
                hdi_imp.hdi.data.processed_image = Opening(
                    hdi_imp.hdi.data.processed_image, disk_size, parallel
                )

    def Slice(self,channels):
        """Subset channels of ndarray for further processing

        Parameters
        ----------
        channels: integer or list of integers
            Indicates which indices to extract from c axis of image
        """

        # update yaml logger
        self.yaml_log['ProcessingSteps'].append({"Slice":{'channels':channels}})

        # Iterate through the set dictionary
        for f, hdi_imp in self.set_dict.items():
            # Ensure that the mask is not none
            if hdi_imp.hdi.data.image is None:
                # Skip this image if there is no mask
                continue

            # Check to see if the preprocessed is initiated
            if hdi_imp.hdi.data.processed_image is None:
                # If not, use the original image
                hdi_imp.hdi.data.processed_image = hdi_imp.hdi.data.image[:,:,channels]

            # Use the initiated image
            else:
                # Use the processed image
                hdi_imp.hdi.data.processed_image = hdi_imp.hdi.data.processed_image[:,:,channels]

    def DeepSlice(self,channels):
        """Subset channels of ndarray for further processing

        Parameters
        ----------
        channels: integer or list of integers
            Indicates which indices to extract from c axis of image
        """
        # logger
        logging.info("DeepSlice: slicing all data for images")
        # update yaml logger
        # self.yaml_log['ProcessingSteps'].append({"DeepSlice":{'channels':channels}})

        # Iterate through the set dictionary
        for f, hdi_imp in self.set_dict.items():
            # at the hdi import level slice all the way from image, to pixel table etc.
            hdi_imp.hdi.Slice(channels=channels)

            #print(hdi_imp.hdi)

    def Close(self, disk_size, parallel=False):
        """Morphological closing on boolean array (i.e., a mask).
        A circular disk is used for the filtering.

        Parameters
        ----------
        filter_size: integer
            Size of disk to use for the median filter.

        parallel: Bool (Default: False)
            Use parallel processing with all available CPUs.
        """

        # logger
        logging.info("Close: filling holes of processed mask")
        # update yaml logger
        self.yaml_log['ProcessingSteps'].append({"Close":{'disk_size':disk_size,
                                'parallel':parallel}})

        # Iterate through the set dictionary
        for f, hdi_imp in self.set_dict.items():
            # Ensure that the mask is not none
            if hdi_imp.hdi.data.image is None:
                # Skip this image if there is no mask
                continue

            # Check to see if the preprocessed is initiated
            if hdi_imp.hdi.data.processed_image is None:
                # If not, use the original image
                hdi_imp.hdi.data.processed_image = Closing(
                    hdi_imp.hdi.data.image, disk_size, parallel
                )

            # Use the initiated image
            else:
                # Use the processed image
                hdi_imp.hdi.data.processed_image = Closing(
                    hdi_imp.hdi.data.processed_image, disk_size, parallel
                )

    def Fill(self):
        """Morphological filling on a binary mask. Fills holes in the given mask.
        """

        # logger
        logging.info("Fill: filling holes of processed mask")
        # update yaml logger
        self.yaml_log['ProcessingSteps'].append("Fill")

        # Iterate through the set dictionary
        for f, hdi_imp in self.set_dict.items():
            # Ensure that the mask is not none
            if hdi_imp.hdi.data.image is None:
                # Skip this image if there is no mask
                continue

            # Check to see if the preprocessed is initiated
            if hdi_imp.hdi.data.processed_image is None:
                # If not, use the original image
                hdi_imp.hdi.data.processed_image = MorphFill(
                    hdi_imp.hdi.data.image
                )

            # Use the initiated image
            else:
                # Use the processed image
                hdi_imp.hdi.data.processed_image = MorphFill(
                    hdi_imp.hdi.data.processed_image
                )

    def MaskConvexHull(self):
        """Create convex hull from a given mask.
        """

        # logger
        logging.info("MaskConvexHull: applying convex hull to processed mask")
        # update yaml logger
        self.yaml_log['ProcessingSteps'].append('MaskConvexHull')

        # Iterate through the set dictionary
        for f, hdi_imp in self.set_dict.items():
            # Ensure that the mask is not none
            if hdi_imp.hdi.data.image is None:
                # Skip this image if there is no mask
                continue

            # Check to see if the preprocessed is initiated
            if hdi_imp.hdi.data.processed_image is None:
                # If not, use the original image
                hdi_imp.hdi.data.processed_image = ConvexHull(
                    hdi_imp.hdi.data.image
                )

            # Use the initiated image
            else:
                # Use the processed image
                hdi_imp.hdi.data.processed_image = ConvexHull(
                    hdi_imp.hdi.data.processed_image
                )

    def NonzeroBox(self):
        """Use a nonzero indices of a binary mask to create a bounding box for
        the mask itself and for the original image. This isused so that
        a controlled amount of padding can be added to the edges of the images in
        a consistent manner.
        """

        # logger
        logging.info("NonzeroBox: applying nonzero box to processed image")
        # update yaml logger
        self.yaml_log['ProcessingSteps'].append('NonzeroBox')

        # Iterate through the set dictionary
        for f, hdi_imp in self.set_dict.items():
            # Ensure that the mask is not none
            if hdi_imp.hdi.data.image is None:
                # Skip this image if there is no mask
                continue

            # Check to see if the preprocessed is initiated
            if hdi_imp.hdi.data.processed_image is None:
                # Skip this iteration because the processed image must be present
                continue

            # If all conditions are satisfied, use the slicing on the images
            (
                hdi_imp.hdi.data.image,
                hdi_imp.hdi.data.processed_image,
            ) = NonzeroSlice(
                hdi_imp.hdi.data.processed_image, hdi_imp.hdi.data.image
            )

    # Create definition for image filtering and processing
    def ApplyMask(self, invert=False):
        """Apply mask to image. This function is
        primarily used on histology images and images that do not need dimension
        reduction. Should be used after a series of morphological
        operations. This applies the resulting mask of thresholding and
        morphological operations.

        Parameters
        ----------
        invert: bool
            Invert mask
        """

        # logger
        logging.info("ApplyMask: applying processed mask to image")
        # update yaml logger
        self.yaml_log['ProcessingSteps'].append({"ApplyMask":{'invert':invert}})

        # Iterate through the set dictionary
        for f, hdi_imp in self.set_dict.items():
            # Ensure that the mask is not none
            if hdi_imp.hdi.data.processed_image is None:
                # Skip this image if there is no mask
                continue

            # Create a temporary image based on the current image
            tmp_im = hdi_imp.hdi.data.image.copy()
            # check for invert
            if invert:
                # set mask object
                hdi_imp.hdi.data.processed_mask = ~hdi_imp.hdi.data.processed_image.copy().toarray()
                # Use the mask on the image and replace the image with the masked image
                tmp_im[hdi_imp.hdi.data.processed_image.toarray()] = 0
            else:
                # set mask object
                hdi_imp.hdi.data.processed_mask = hdi_imp.hdi.data.processed_image.copy().toarray()
                # Use the mask on the image and replace the image with the masked image
                tmp_im[~hdi_imp.hdi.data.processed_image.toarray()] = 0
            # Set the processed image as the masked array
            hdi_imp.hdi.data.processed_image = tmp_im

    def GetMaskBoundaryQC(self):
        """Use processed image mask to create boundary for pc purposes.
        """
        # create dictionary to store boundary masks in
        bound_masks = {}
        # print
        logging.info('Extracting countour from processed masks')
        # Iterate through the set dictionary
        for f, hdi_imp in self.set_dict.items():
            # check if processed mask exists
            if hdi_imp.hdi.data.processed_mask is None:
                # check if there is manual mask
                if hdi_imp.hdi.data.mask is not None:
                    # if yes just use that
                    hdi_imp.hdi.data.processed_mask = hdi_imp.hdi.data.mask
                    # run get boundaries
                    bound = MaskBoundary(hdi_imp.hdi.data.processed_mask)
                    # update dictionary
                    bound_masks.update({f:bound})
                else:
                    # update dictionary to none
                    bound_masks.update({f:None})
            else:
                # run get boundaries
                bound = MaskBoundary(hdi_imp.hdi.data.processed_mask)
                # update dictionary
                bound_masks.update({f:bound})
        # return boundary masks
        return bound_masks

    def GetSubsampleMaskQC(self):
        """Use subsampled image mask to create boundary for pc purposes.
        """
        # create dictionary to store subsampled masks in
        sub_masks = {}
        # print
        logging.info('Extracting subsampled masks if used for dimension reduction')
        # Iterate through the set dictionary
        for f, hdi_imp in self.set_dict.items():
            # check if processed mask exists
            if hdi_imp.hdi.data.subsampled_mask is None:
                # add nothing
                sub_masks.update({f:None})
            else:
                # update with mask
                sub_masks.update({f:hdi_imp.hdi.data.subsampled_mask})
        # return boundary masks
        return sub_masks

    # Add function for exporting UMAP nifti image
    def ExportNifti1(self, output_dir, padding=None, target_size=None):
        """Export processed images resulting from UMAP and
        spatially mapping UMAP, or exporting processed histology images.

        Parameters
        ----------
        output_dir: string
            Path to output directory to store processed nifti image.

        padding: string of tuple of type integer (padx,pady; Default: None)
            Indicates height and length padding to add to the image before exporting.

        target_size: string of tuple of type integer (sizex,sizey; Default: None)
            Resize image using bilinear interpolation before exporting.
        """

        # update yaml logger
        self.yaml_log['ProcessingSteps'].append({"ExportNifti1":{'output_dir':str(output_dir),
                                        'padding':str(padding),
                                        'target_size':str(target_size)}})

        # create dictionary with connected file names
        connect_dict = {}

        # Iterate through the set dictionary
        for f, hdi_imp in self.set_dict.items():
            # Create an image name -- remove .ome in the name if it exists and add umap suffix
            im_name = Path(
                os.path.join(
                    str(output_dir),
                    f.stem.replace(".ome", "")
                    + "_processed.nii",
                )
            )

            # Ensure that the mask is not none
            if hdi_imp.hdi.data.processed_image is None:
                # Make sure the image exists
                if hdi_imp.hdi.data.image is None:
                    continue
                # Otherwise export the image
                else:
                    # Export the original image
                    ExportNifti(hdi_imp.hdi.data.image, im_name, padding, target_size)
            # Otherwise export the processed image
            else:
                # Use utils export nifti function
                ExportNifti(hdi_imp.hdi.data.processed_image, im_name, padding, target_size)
            # Add exported file names to class object -- connect input file name with the exported name
            connect_dict.update({f: im_name})
            # update the object padding and target size
            hdi_imp.padding = padding
            hdi_imp.target_size = target_size

        # Add the connecting dictionary to the class object
        self.processed_images_export.update(connect_dict)

        # return the dictionary of input names to output names
        # return connect_dict

    # Add function for exporting UMAP nifti image
    def Export(self, output_dir, suffix="_processed.nii", padding=None, target_size=None,grayscale=False):
        """Export processed images resulting from UMAP and
        spatially mapping UMAP, or exporting processed histology images.

        Parameters
        ----------
        output_dir: string
            Path to output directory to store processed image.

        padding: string of tuple of type integer (padx,pady; Default: None)
            Indicates height and length padding to add to the image before exporting.

        target_size: string of tuple of type integer (sizex,sizey; Default: None)
            Resize image using bilinear interpolation before exporting.

        suffix: string
            Indicates suffix to add to exported image (including file type).
        """

        # update yaml logger
        self.yaml_log['ProcessingSteps'].append({"Export":{'output_dir':str(output_dir),
                                    'suffix':suffix,
                                    'padding':str(padding),
                                    'target_size':str(target_size),
                                    'grayscale':grayscale}})

        # Create dictionary with connected file names
        connect_dict = {}

        # Iterate through the set dictionary
        for f, hdi_imp in self.set_dict.items():
            # Create an image name -- remove .ome in the name if it exists and add umap suffix
            im_name = Path(
                os.path.join(
                    str(output_dir),
                    f.stem.replace(".ome", "")
                    + str(suffix),
                )
            )

            # Ensure that the mask is not none
            if hdi_imp.hdi.data.processed_image is None:
                # Make sure the image exists
                if hdi_imp.hdi.data.image is None:
                    continue
                # Otherwise export the image
                else:
                    # Export the original image
                    Export1(hdi_imp.hdi.data.image, im_name, padding, target_size,grayscale)
            # Otherwise export the processed image
            else:
                # Use utils export nifti function
                Export1(hdi_imp.hdi.data.processed_image, im_name, padding, target_size,grayscale)
            # Add exported file names to class object -- connect input file name with the exported name
            connect_dict.update({f: im_name})
            # update the object padding and target size
            hdi_imp.padding = padding
            hdi_imp.target_size = target_size

        # Add the connecting dictionary to the class object
        self.processed_images_export.update(connect_dict)

        # return the dictionary of input names to output names
        # return connect_dict

    # Add function for exporting UMAP nifti image
    def ExportMask(self, output_dir, suffix="_processed_mask.tiff", padding=None, target_size=None):
        """Export processed mask after ApplyMask function.

        Parameters
        ----------
        output_dir: string
            Path to output directory to store processed image.

        padding: string of tuple of type integer (padx,pady; Default: None)
            Indicates height and length padding to add to the image before exporting.

        target_size: string of tuple of type integer (sizex,sizey; Default: None)
            Resize image using bilinear interpolation before exporting.

        suffix: string
            Indicates suffix to add to exported image (including file type).
        """

        # update yaml logger
        self.yaml_log['ProcessingSteps'].append({"ExportMask":{'output_dir':str(output_dir),
                                    'suffix':suffix,
                                    'padding':str(padding),
                                    'target_size':str(target_size)}})


        # Create dictionary with connected file names
        connect_dict = {}

        # Iterate through the set dictionary
        for f, hdi_imp in self.set_dict.items():
            # Create an image name -- remove .ome in the name if it exists and add umap suffix
            im_name = Path(
                os.path.join(
                    str(output_dir),
                    f.stem.replace(".ome", "")
                    + str(suffix),
                )
            )

            if hdi_imp.hdi.data.processed_mask is None:
                if hdi_imp.hdi.data.mask is not None:
                    # Export mask instead of processed mask (convert from scipy to array)
                    Export1(hdi_imp.hdi.data.mask.toarray(), im_name, padding, target_size)
                else:
                    continue
            # Otherwise export the image
            else:
                if isinstance(hdi_imp.hdi.data.processed_mask, scipy.sparse.coo_matrix):
                    # convert to array
                    hdi_imp.hdi.data.processed_mask = hdi_imp.hdi.data.processed_mask.toarray()
                # Export the original image
                Export1(hdi_imp.hdi.data.processed_mask, im_name, padding, target_size)

            connect_dict.update({f: im_name})

        # Add the connecting dictionary to the class object
        self.masks_export.update(connect_dict)

    def ExportIMZMLCoordinateMask(self,output_dir,padding,target_size):
        """
        Create mask indicating coordinates of MSI data acquisition and export
        as tiff file.
        """

        # update yaml logger
        self.yaml_log['ProcessingSteps'].append({"ExportIMZMLCoordinateMask":{'output_dir':str(output_dir),
                                    'padding':str(padding),
                                    'target_size':str(target_size)}})


        # Create dictionary with connected file names
        connect_dict = {}

        # Iterate through the set dictionary
        for f, hdi_imp in self.set_dict.items():
            # check the type of imported data to make sure is imzML
            if not isinstance(hdi_imp,_imzml_reader.imzMLreader):
                pass

            # get corresponding hdi import
            hdi_imp=DeepCopyHDIreader(self.set_dict[f])
            # create temporary new data
            new_data = _import.HDIreader(
                data=hdi_imp.hdi.data,
                image=hdi_imp.hdi.data.image,
                mask=None,
                channels=hdi_imp.hdi.data.channels,
                flatten=True,   # set flatten to true for reload
                filename=hdi_imp.hdi.data.filename,
                subsample=None,
                method=None,
                save_mem=True
            )

            # Create an image name -- remove .ome in the name if it exists and add umap suffix
            im_name = Path(
                os.path.join(
                    str(output_dir),
                    f.stem
                    + "_mask.tiff",
                )
            )

            # create the mask
            im = new_data.CreateCoordinateMask()
            # export (is already np array instead of sparse matrix)
            Export1(im, im_name, padding, target_size)
            # update export dictionary
            connect_dict.update({f: im_name})

        # Add the connecting dictionary to the class object
        self.masks_export.update({connect_dict})

		# return results
        return connect_dict

    def IMZMLtoNIFTI(self,output_dir,padding,target_size):
        """
        Export imzml file as nifti while preserving metadata.
        """

        # update yaml logger
        self.yaml_log['ProcessingSteps'].append({"IMZMLtoNIFTI":{'output_dir':str(output_dir),
                                    'padding':str(padding),
                                    'target_size':str(target_size)}})


        # Create dictionary with connected file names
        connect_dict = {}

        # Iterate through the set dictionary
        for f, hdi_imp in self.set_dict.items():
            # check the type of imported data to make sure is imzML
            if not isinstance(hdi_imp,_imzml_reader.imzMLreader):
                pass

            # get corresponding hdi import
            hdi_imp=DeepCopyHDIreader(self.set_dict[f])
            # create temporary new data
            new_data = _import.HDIreader(
                data=hdi_imp.hdi.data,
                image=hdi_imp.hdi.data.image,
                mask=None,
                channels=hdi_imp.hdi.data.channels,
                flatten=True,   # set flatten to true for reload
                filename=hdi_imp.hdi.data.filename,
                subsample=None,
                method=None,
                save_mem=True
            )

            # Create an image name -- remove .ome in the name if it exists and add umap suffix
            im_name = Path(
                os.path.join(
                    str(output_dir),
                    f.stem
                    + ".nii",
                )
            )

            # create the mask
            im = new_data.ConstructImage()
            # export (is already np array instead of sparse matrix)
            Export1(im, im_name, padding, target_size)
            # update export dictionary
            connect_dict.update({f: im_name})

        # Add the connecting dictionary to the class object
        self.processed_images_export.update(connect_dict)

		# return results
        return connect_dict


    def IMZMLtoHDF5(self,output_dir,padding,target_size):
        """
        Export imzml file as hdf5 while preserving metadata.
        """

        # update yaml logger
        self.yaml_log['ProcessingSteps'].append({"IMZMLtoNIFTI":{'output_dir':str(output_dir),
                                    'padding':str(padding),
                                    'target_size':str(target_size)}})

        # Create dictionary with connected file names
        connect_dict = {}
        # convert the padding and target size to tuple if present
        if padding is not None:
            # check for string or integer
            if isinstance(padding,str):
                padding = literal_eval(padding)
        if target_size is not None:
            # check for string or integer
            if isinstance(target_size,str):
                target_size = literal_eval(target_size)

        # Iterate through the set dictionary
        for f, hdi_imp in self.set_dict.items():
            # check the type of imported data to make sure is imzML
            if not isinstance(hdi_imp,_imzml_reader.imzMLreader):
                pass

            # get corresponding hdi import
            hdi_imp=DeepCopyHDIreader(self.set_dict[f])
            # create temporary new data
            new_data = _import.HDIreader(
                data=hdi_imp.hdi.data,
                image=hdi_imp.hdi.data.image,
                mask=None,
                channels=hdi_imp.hdi.data.channels,
                flatten=True,   # set flatten to true for reload
                filename=hdi_imp.hdi.data.filename,
                subsample=None,
                method=None,
                save_mem=True
            )

            # Create an image name -- remove .ome in the name if it exists and add umap suffix
            im_name = Path(
                os.path.join(
                    str(output_dir),
                    f.stem
                    + ".hdf5",
                )
            )

            # iterate through channels (saves memory instead of loading full image)
            for c in len(self.data.pixel_table.columns):
                # create image
                im = self.CreateSingleChannelArray(idx=c)
                #Add a color axis when reshaping instead
                im = im.reshape((1,im.shape[0],im.shape[1],1))

                # Check to see if padding
                if padding is not None:
                    # check dimensions of the image
                    if im.ndim > 2:
                        im = np.pad(
                            im,
                            [(padding[0], padding[0]), (padding[1], padding[1]), (0, 0)],
                            mode="constant",
                        )
                    elif im.ndim == 2:
                        im = np.pad(
                            im,
                            [(padding[0], padding[0]), (padding[1], padding[1])],
                            mode="constant",
                        )
                    else:
                        raise ValueError(f'Image with {im.ndim} not supported.')
                # Check to see if resizing
                if target_size is not None:
                    im = resize(im,target_size)

                #Create an hdf5 dataset if idx is 0 plane
                if c == 0:
                    #Create hdf5
                    h5 = h5py.File(im_name, "w")
                    h5.create_dataset(str(im_stem), data=image,chunks=True,maxshape=(1,None,None,None))
                    # copy over the imzml information to metadata in h5 file
                    h5['imzml metadata'] = self.data.imzmldict
                    h5.close()
                else:
                    #Append hdf5 dataset
                    h5 = h5py.File(im_name, "a")
                    #Add step size to the z axis
                    h5[str(im_stem)].resize((c+1), axis = 3)
                    #Add the image to the new channels
                    h5[str(im_stem)][:,:,:,c:c+1] = image
                    h5.close()

            # update export dictionary
            connect_dict.update({f: im_name})

        # Add the connecting dictionary to the class object
        self.processed_images_export.update(connect_dict)

    def PlotProcessedImage(self):
        # for plotting purposes, extract the key of the data set
        key = list(self.set_dict.keys())[0]
        if isinstance(self.set_dict[key].hdi.data.processed_image, np.ndarray):
            plt.imshow(self.set_dict[key].hdi.data.processed_image)
        else:
            # check processed mask
            plt.imshow(self.set_dict[key].hdi.data.processed_image.toarray())

    def PlotInputImage(self,channel):
        # for plotting purposes, extract the key of the data set
        key = list(self.set_dict.keys())[0]
        # check ruthenium stain
        plt.imshow(self.set_dict[key].hdi.data.image[:,:,channel])


class HDIpreprocessing(IntraModalityDataset):
    """MIAAIM preprocesing module with logging and quality control
    capabilities.

    Wraps IntraModalityDataset class for reading and logging purposes.
    """

    def __init__(self,
                paths,
                qc=True,
                mask=None,
                root_folder=None,
                **kwargs):

        self.name = None
        self.root_folder = None
        self.qc_dir = None
        self.qc_preprocess_dir = None
        self.qc_preprocess_name_dir = None
        self.docs_dir = None
        self.preprocess_dir = None
        self.processed_dir = None
        self.log_name = None
        self.logger = None
        self.sh_name = None
        self.yaml_name = None

        # create logger format
        FORMAT = '%(asctime)s | [%(pathname)s:%(lineno)s - %(funcName)s() ] | %(message)s'

        # check to see if paths are list or single image
        if not isinstance(paths,list):
            # create list
            paths = [Path(paths)]
            # get the name of the paths
            name = paths[0].name
            self.name = name
        else:
            # create pathlib objects for list
            paths=[Path(p) for p in paths]
            # get the name of the paths
            name = paths[0].name
            self.name = name


        # root folder will be created
        if root_folder is None:
            # create one
            # print(os.path.join(Path(paths[0]).parent,"miaaim-proc"))
            root_folder = Path(paths[0].parent.parent)
            preprocess_dir = root_folder.joinpath("preprocessing")
            if not preprocess_dir.exists():
                preprocess_dir.mkdir()
        # create folder for the specific modality processing
        processed_dir = preprocess_dir.joinpath(f'{self.name}')
        if not processed_dir.exists():
            processed_dir.mkdir()

        # create logger if qc true
        if qc:

            # create a docs directory
            docs_dir = Path(root_folder).joinpath("docs")

            # check if exists already
            if not docs_dir.exists():
                # make it
                docs_dir.mkdir()

            # create a parameters directory
            pars_dir = docs_dir.joinpath("parameters")
            # check if exists already
            if not pars_dir.exists():
                # make it
                pars_dir.mkdir()
            # create name of shell command
            yaml_name = os.path.join(Path(pars_dir),"miaaim-preprocessing"+f'-{self.name}'+".yaml")

            # create a qc directory
            qc_dir = docs_dir.joinpath("qc")
            # check if exists already
            if not qc_dir.exists():
                # make it
                qc_dir.mkdir()

            # create direectory specific for process
            qc_preprocess_dir = qc_dir.joinpath("preprocessing")
            # check if exists already
            if not qc_preprocess_dir.exists():
                # make it
                qc_preprocess_dir.mkdir()

            qc_preprocess_name_dir = qc_preprocess_dir.joinpath(f'{self.name}')
            # check if exists already
            if not qc_preprocess_name_dir.exists():
                # make it
                qc_preprocess_name_dir.mkdir()

            # create a qc directory
            prov_dir = docs_dir.joinpath("provenance")
            # check if exists already
            if not prov_dir.exists():
                # make it
                prov_dir.mkdir()
            # create name of logger
            log_name = os.path.join(Path(prov_dir),"miaaim-preprocessing"+f'-{self.name}'+".log")

            # check if it exists already
            if Path(log_name).exists():
                # remove it
                Path(log_name).unlink()
            # configure log
            logging.basicConfig(filename=log_name,
                                    encoding='utf-8',
                                    level=logging.INFO,
                                    format=FORMAT,
                                    force=True)

            # get logger
            logger = logging.getLogger()
            # writing to stdout
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.DEBUG)
            # handler.setFormatter(FORMAT)
            logger.addHandler(handler)
            # command to capture print functions to log
            print = logger.info

            # create name of shell command
            sh_name = os.path.join(Path(prov_dir),"miaaim-preprocessing"+f'-{self.name}'+".sh")

            # update attributes
            self.root_folder = root_folder
            self.processed_dir = processed_dir
            self.preprocess_dir = preprocess_dir
            self.docs_dir = docs_dir
            self.qc_dir = qc_dir
            self.qc_preprocess_dir = qc_preprocess_dir
            self.qc_preprocess_name_dir = qc_preprocess_name_dir
            self.prov_dir = prov_dir
            self.log_name = log_name
            self.logger = logger
            self.sh_name = sh_name
            self.yaml_name = yaml_name

            # print first log
            logging.info("MIAAIM PREPROCESSING")
            logging.info(f'MIAAIM VERSION {miaaim.__version__}')
            logging.info(f'METHOD: HDIpreprocessing')
            logging.info(f'ROOT FOLDER: {self.root_folder}')
            logging.info(f'RESULTS FOLDER: {self.processed_dir}')
            logging.info(f'PROVENANCE FOLDER: {self.prov_dir}')
            logging.info(f'QC FOLDER: {self.qc_preprocess_name_dir} \n')

        logging.info(f'IMPORTING DATA')
        # Create a list to store the _import sets in
        data = []
        # Iterate through each path
        for i in range(len(paths)):
            # Ensure that it is a pathlib object
            p = Path(paths[i])
            # Read the data using _import
            p_dat = _import.HDIreader(path_to_data=p, mask=mask, **kwargs)
            # Append this p_dat to the data list
            data.append(p_dat)
        # initialize IntraModalityDataset
        IntraModalityDataset.__init__(self,
                                    list_of_HDIimports=data,
                                    qc=qc)

        # update yaml file
        # update yaml file
        self.yaml_log.update({'MODULE':"Preprocessing"})
        self.yaml_log.update({'METHOD':"HDIpreprocessing"})
        # update logger
        logging.info(f'\n')
        logging.info("PROCESSING DATA")
        
    def RunUMAP(
        self, import_args={'subsample':True,'method':'default'}, channels=None, **kwargs
    ):
        logging.info("RunUMAP: computing UMAP embedding")

        # run super method
        super().RunUMAP(import_args=import_args, channels=channels, **kwargs)        
    
    def RunParametricUMAP(
        self, import_args={'subsample':True,'method':'default'}, channels=None, **kwargs
    ):
        logging.info("RunParametricUMAP: computing parametric UMAP embedding")

        # run super method
        super().RunParametricUMAP(import_args=import_args, channels=channels, **kwargs) 

    def RunOptimalUMAP(
        self, import_args={'subsample':True,'method':'default'}, dim_range='(1,11)', landmarks=3000, export_diagnostics=True, output_dir=None, n_jobs=1, **kwargs
    ):
        logging.info("RunOptimalUMAP: computing optimal UMAP embedding")
        # check for output_dir
        if output_dir is None:
            # default to QC directory
            output_dir = self.qc_preprocess_name_dir

        # check for qc
        if self.qc:
            # override
            export_diagnostics=True

        # run super method
        super().RunOptimalUMAP(import_args=import_args,
                        dim_range=dim_range,
                        landmarks=landmarks,
                        export_diagnostics=export_diagnostics,
                        output_dir=output_dir,
                        n_jobs=n_jobs,
                         **kwargs)

    def RunOptimalParametricUMAP(
        self, import_args={'subsample':True,'method':'default'}, dim_range=(1,11), landmarks=3000, export_diagnostics=True, output_dir=None, n_jobs=1, **kwargs
    ):
        logging.info("RunOptimalParametricUMAP: computing optimal parametric UMAP embedding")
        # check for output_dir
        if output_dir is None:
            # default to QC directory
            output_dir = self.qc_preprocess_name_dir

        # check for qc
        if self.qc:
            # override
            export_diagnostics=True

        # run super method
        super().RunOptimalParametricUMAP(import_args=import_args,
                        dim_range=dim_range,
                        landmarks=landmarks,
                        export_diagnostics=export_diagnostics,
                        output_dir=output_dir,
                        n_jobs=n_jobs,
                         **kwargs)


    def ExportNifti1(self, output_dir=None, padding=None, target_size=None,grayscale=False):

        logging.info("ExportNifti1: exporting processed image to nifti format")
        # check for output_dir
        if output_dir is None:
            # create it
            output_dir = self.processed_dir
            if not output_dir.exists():
                output_dir.mkdir()

        # run super method
        super().ExportNifti1(output_dir=output_dir,
                            padding=padding,
                            target_size=target_size,
                            grayscale=grayscale)

    def Export(self, output_dir=None, suffix="_processed.nii", padding=None, target_size=None,grayscale=False):

        logging.info("Export: exporting processed image")
        # check for output_dir
        if output_dir is None:
            # create it
            output_dir = self.processed_dir
            if not output_dir.exists():
                output_dir.mkdir()

        # run super method
        super().Export(output_dir=output_dir,
                        suffix=suffix,
                        padding=padding,
                        target_size=target_size,
                        grayscale=grayscale)

    def ExportMask(self, output_dir=None, suffix="_processed_mask.tiff", padding=None, target_size=None):

        logging.info("ExportMask: exporting processed mask")
        # check for output_dir
        if output_dir is None:
            # create it
            output_dir = self.processed_dir
            if not output_dir.exists():
                output_dir.mkdir()

        # run super method
        super().ExportMask(output_dir=output_dir,
                        suffix=suffix,
                        padding=padding,
                        target_size=target_size)

    def ExportQCMask(self):
        # get dictionary of processed masks
        qc_masks = self.GetMaskBoundaryQC()
        # Iterate through the set dictionary
        for f, hdi_imp in self.set_dict.items():
            # extract the corresponding boundary mask
            m = qc_masks[f]
            # check if none
            if m is not None:
                # create new name and export
                qc_name = Path(
                    os.path.join(
                        str(self.qc_preprocess_name_dir),
                        f.stem.replace(".ome", "")
                        + str("_qcMask.tiff")
                    )
                )
                # check that padding and target size are not none
                if hdi_imp.padding is None:
                    # raise warning
                    warnings.warn(f'{f} padding is set to none')

                # Use utils export nifti function
                Export1(m, qc_name, hdi_imp.padding, hdi_imp.target_size)

    def ExportSubsampleQCMask(self):
        # get dictionary of subsampled masks
        sub_masks = self.GetSubsampleMaskQC()
        # Iterate through the set dictionary
        for f, hdi_imp in self.set_dict.items():
            # extract the corresponding boundary mask
            m = sub_masks[f]
            # check if none
            if m is not None:
                # create new name and export
                qc_name = Path(
                    os.path.join(
                        str(self.qc_preprocess_name_dir),
                        f.stem.replace(".ome", "")
                        + str("_qcSubMask.tiff")
                    )
                )
                # check that padding and target size are not none
                if hdi_imp.padding is None:
                    # raise warning
                    warnings.warn(f'{f} padding is set to none')

                # Use utils export nifti function
                Export1(m, qc_name, hdi_imp.padding, hdi_imp.target_size)

    def ExportIMZMLCoordinateMask(self,output_dir=None,padding=None,target_size=None):
        """
        Create mask indicating coordinates of MSI data acquisition and export
        as tiff file.
        """

        logging.info("ExportIMZMLCoordinateMask: exporting coordinate mask from imzML data")
        # check for output_dir
        if output_dir is None:
            # create it
            output_dir = self.processed_dir
            if not output_dir.exists():
                output_dir.mkdir()

        # run super method
        super().ExportIMZMLCoordinateMask(output_dir=output_dir,
                                        padding=padding,
                                        target_size=target_size)

    def IMZMLtoNIFTI(self,output_dir=None,padding=None,target_size=None):
        """
        Export imzml file as nifti while preserving metadata.
        """

        logging.info("IMZMLtoNIFTI: exporting imzML data to nifti format")
        # check for output_dir
        if output_dir is None:
            # create it
            output_dir = self.processed_dir
            if not output_dir.exists():
                output_dir.mkdir()

        # run super method
        super().IMZMLtoNIFTI(output_dir=output_dir,
                            padding=padding,
                            target_size=target_size)

    def IMZMLtoHDF5(self,output_dir=None,padding=None,target_size=None):
        """
        Export imzml file as hdf5 while preserving metadata.
        """

        logging.info("IMZMLtoHDF5: exporting imzML data to hdf5 format")
        # check for output_dir
        if output_dir is None:
            # create it
            output_dir = self.processed_dir
            if not output_dir.exists():
                output_dir.mkdir()

        # run super method
        super().IMZMLtoHDF5(output_dir=output_dir,
                            padding=padding,
                            target_size=target_size)

    def _exportYAML(self):
        """Function to export yaml log to file for documentation
        """
        logging.info(f'Exporting {self.yaml_name}')
        # open file and export
        with open(self.yaml_name, 'w') as outfile:
            yaml.dump(self.yaml_log, outfile, default_flow_style=False,sort_keys=False)

    def _exportSH(self):
        """Function to export sh command to file for documentation
        """
        logging.info(f'Exporting {self.sh_name}')
        # get name of the python path and cli file
        proc_fname = os.path.join(Path(_parse.__file__).parent,"_cli_proc.py")
        # get path to python executable
        # create shell command script
        with open (self.sh_name, 'w') as rsh:
            rsh.write(f'''\
        #! /bin/bash
        {sys.executable} {proc_fname} --pars {self.yaml_name}
        ''')

    def QC(self):
        """Function to export QC metrics to file for documentation
        """
        # log
        logging.info("QC: extracting quality control information")
        self.yaml_log['ProcessingSteps'].append("QC")
        # export QC information
        # check for qc
        if self.qc:
            # export processed masks
            self.ExportQCMask()
            # export subsampled masks
            self.ExportSubsampleQCMask()

        # provenance
        self._exportYAML()
        self._exportSH()
        # close the logger
        self.logger.handlers.clear()












#
