# Developer: Joshua M. Hess, BSc
# Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

#Import modules
from sklearn.base import BaseEstimator
from umap import umap_
import itertools
import warnings
import numpy as np
import scipy.sparse
from sklearn.utils import check_random_state, check_array


def transform(UMAP, X):
    """Transform X into the existing embedded space and return that
    transformed output.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        New data to be transformed.

    Returns
    -------
    X_new : array, shape (n_samples, n_components)
        Embedding of the new data in low-dimensional space.
    """
    # If we fit just a single instance then error
    if self._raw_data.shape[0] == 1:
        raise ValueError(
            "Transform unavailable when model was fit with only a single data sample."
        )
    # If we just have the original input then short circuit things
    X = check_array(X, dtype=np.float32, accept_sparse="csr", order="C")
    x_hash = joblib.hash(X)
    if x_hash == UMAP._input_hash:
        if self.transform_mode == "embedding":
            return UMAP.embedding_
        elif UMAP.transform_mode == "graph":
            return UMAP.graph_
        else:
            raise ValueError(
                "Unrecognized transform mode {}; should be one of 'embedding' or 'graph'".format(
                    self.transform_mode
                )
            )
    if UMAP.densmap:
        raise NotImplementedError(
            "Transforming data into an existing embedding not supported for densMAP."
        )

    if UMAP.metric == "precomputed":
        raise ValueError(
            "Transform  of new data not available for precomputed metric."
        )

    # X = check_array(X, dtype=np.float32, order="C", accept_sparse="csr")
    random_state = check_random_state(UMAP.transform_seed)
    rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

    if UMAP._small_data:
        try:
            # sklearn pairwise_distances fails for callable metric on sparse data
            _m = self.metric if self._sparse_data else self._input_distance_func
            dmat = pairwise_distances(
                X, UMAP._raw_data, metric=_m, **self._metric_kwds
            )
        except (TypeError, ValueError):
            dmat = dist.pairwise_special_metric(
                X,
                self._raw_data,
                metric=self._input_distance_func,
                kwds=self._metric_kwds,
            )
        indices = np.argpartition(dmat, self._n_neighbors)[:, : self._n_neighbors]
        dmat_shortened = submatrix(dmat, indices, self._n_neighbors)
        indices_sorted = np.argsort(dmat_shortened)
        indices = submatrix(indices, indices_sorted, self._n_neighbors)
        dists = submatrix(dmat_shortened, indices_sorted, self._n_neighbors)
    else:
        epsilon = 0.24 if self._knn_search_index._angular_trees else 0.12
        indices, dists = self._knn_search_index.query(
            X, self.n_neighbors, epsilon=epsilon
        )

    dists = dists.astype(np.float32, order="C")
    # Remove any nearest neighbours who's distances are greater than our disconnection_distance
    indices[dists >= self._disconnection_distance] = -1
    adjusted_local_connectivity = max(0.0, self.local_connectivity - 1.0)
    sigmas, rhos = smooth_knn_dist(
        dists,
        float(self._n_neighbors),
        local_connectivity=float(adjusted_local_connectivity),
    )

    rows, cols, vals, dists = compute_membership_strengths(
        indices, dists, sigmas, rhos, bipartite=True
    )

    graph = scipy.sparse.coo_matrix(
        (vals, (rows, cols)), shape=(X.shape[0], self._raw_data.shape[0])
    )

    if self.transform_mode == "graph":
        return graph






def compute_cobordism(X_list, type="intersection", **kwargs):
    """Patch multiple umap models (simplicial sets) to create a cobordism.

    Parameters
    ----------
    X_list: list of arrays
        Objects to be passed to umap.fit(). As of now, all of these arrays
        must have the same number of variables.

    Returns
    -------
    stacked: sparse graph
        Stitched together simplicial set that represents a cobordism.
    """

    #get index elements
    sz = np.arange(0,len(X_list))
    #Create indexer
    idx_bs = 0
    #Create dictionary for the base UMAPs (intra datasets)
    bases = {}
    #Iterate through each element in the list and run umap base
    for X in X_list:
        #Run the base UMAP
        b = umap_.UMAP(transform_mode = 'graph', **kwargs).fit(X)
        #Add the UMAP object to the dictionary
        bases.update({idx_bs:b})
        #update the index
        idx_bs = idx_bs +1

    combs = list(itertools.combinations(sz, 2))
    #Create indexer
    idx = 0
    #Create overarching dictionary to store all results
    coreograph = {}

    if type == "intersection":

        #Iterate through each data set in the input
        for X in combs:
            #Use the simplicial sets to find mutual nearest neighbors (intersection to patch manifolds)
            r_l = bases[X[0]].transform(X_list[X[1]])
            l_r = bases[X[1]].transform(X_list[X[0]])
            #Mix simplicial sets to get the intersection patch in between train and test (consistent coordinate system)
            mix = umap_.general_simplicial_set_intersection(r_l.T, l_r, 0.5)
            #mix = general_simplicial_set_union(r_l.T, l_r)

            #Update the indexer
            coreograph.update({idx:mix})
            #Update the indexer
            idx = idx + 1

    elif type == "union":

        #Iterate through each data set in the input
        for X in combs:
            #Use the simplicial sets to find mutual nearest neighbors (intersection to patch manifolds)
            r_l = bases[X[0]].transform(X_list[X[1]])
            l_r = bases[X[1]].transform(X_list[X[0]])
            #Mix simplicial sets to get the intersection patch in between train and test (consistent coordinate system)
            mix = umap_.general_simplicial_set_union(r_l.T, l_r)

            #Update the indexer
            coreograph.update({idx:mix})
            #Update the indexer
            idx = idx + 1

    #Get the length of the data
    n = len(X_list)
    #Create row blocks (left to right)
    row_block = []
    #Iterate through the blocks and append rows first
    for i in sz:
        #Get the shape of the base block
        shapes = bases[i].graph_.shape
        #Reset the start to be false
        start = False
        #Set the block to none
        block = None
        #Iterate through blocks
        for j in sz:
            #Check if i and j are the same -- signifies a base block
            if i == j:
                #Create the first block
                chunk = bases[i].graph_
            #Otherwise
            else:
                #Get the correct index from the coreograph dictionary
                v = [x for x, y in enumerate(combs) if ((y[0] == i and y[1] == j) or (y[1] == i and y[0] == j))][0]
                #Get the pairwise combination
                chunk = coreograph[v]

                #Check if we need to transpose outcome
                if chunk.shape[0] != bases[i].graph_.shape[0]:
                    #Flip
                    chunk = chunk.T

            #Check to see if there exists a start block
            if not start:
                #Create a start chunk if not one
                block = chunk
                #Set start to true now
                start = True
            #Otherwise update the block
            else:
                #Append the block
                block = scipy.sparse.hstack([block,chunk])
        #Update the row blocks
        row_block.append(block)
    #Stack vertically all row blocks
    stacked = scipy.sparse.vstack(row_block)
    #Reset the connectivity
    stacked = umap_.reset_local_connectivity(stacked)
    #stacked[stacked<0.1] = 0
    #return the connected and stacked UMAP
    return stacked

def get_batch_indices(X_list):
    """Get start and stop indices of each array in data list.

    Parameters
    ----------
    X_list: list of arrays
        Objects to be passed to umap.fit(). As of now, all of these arrays
        must have the same number of variables.

    Returns
    -------
    indices: list
        Stores start and stop indices of each array in the input list.

    counts_list: list
        Stores the absolute count of rows in each input array.
    """

    #Create dictionary to store the indices in
    indices = []
    #Create a counter for starting row
    count = 0
    #Create a list of counts
    counts_list = []
    #Iterate through the list
    for x in range(len(X_list)):
        #Get shape of rows
        rows = X_list[x].shape[0]
        #Get the shape of the rows
        indices.append((count,count+rows))
        #Updat the counter
        count = count + rows
        #Update the counts list
        counts_list.append(count)
    #Return the indices
    return indices, counts_list


def embed_cobordism(X_list,
                    patched_simplicial_set,
                    n_components = 2,
                    n_epochs = 200,
                    gamma = 1,
                    initial_alpha = 1.0,
                    **kwargs
                    ):
    """Embed cobordism.

    Parameters
    ----------
    X_list: list of arrays
        Objects to be passed to umap.fit(). As of now, all of these arrays
        must have the same number of variables.

    patched_simplicial_set: sparse graph
        Stitched together simplicial set that represents a cobordism.

    n_components: integer (Default: 2)
        Number of embedding components.

    Returns
    -------
    embed: array
        Embedding array.
    """

    #Combine the data
    X = np.vstack(X_list)
    #Run UMAP on the first iteration -- we will skip simplicial set construction in next iterations
    base = umap_.UMAP(n_components,n_epochs, **kwargs)

    X = check_array(X, dtype=np.float32, accept_sparse="csr", order="C")
    base._raw_data = X

    # Handle all the optional arguments, setting default
    if base.a is None or base.b is None:
        base._a, base._b = umap_.find_ab_params(base.spread, base.min_dist)
    else:
        base._a = base.a
        base._b = base.b

    if isinstance(base.init, np.ndarray):
        init = check_array(base.init, dtype=np.float32, accept_sparse=False)
    else:
        init = base.init

    base._initial_alpha = base.learning_rate

    base._validate_parameters()

    #Embed the data
    embed, _ = umap_.simplicial_set_embedding(
        data = X,
        graph = patched_simplicial_set,
        n_components = n_components,
        initial_alpha = base._initial_alpha,
        a = base._a,
        b = base._b,
        gamma = gamma,
        negative_sample_rate = base.negative_sample_rate,
        n_epochs = n_epochs,
        init = base.init,
        random_state = check_random_state(base.random_state),
        metric = base._input_distance_func,
        metric_kwds = base._metric_kwds,
        densmap = False,
        densmap_kwds = {},
        output_dens = False,
        output_metric = base._output_distance_func,
        output_metric_kwds = base._output_metric_kwds,
        euclidean_output=True,
        parallel = base.random_state is None,
        verbose = base.verbose,
    )
    #return the embedding
    return embed

def compute_i_patchmap_cobordism(dat,new_dat,random_state,n_neighbors,**kwargs):
    """Create cobordism between reference and query data for information transfer.

    Parameters
    ----------
    dat: array
        Reference data set.

    new_dat: array
        Query data set.

    random_state: integer
        Random state to set the seed for reproducibility.

    n_neighbors: integer
        Number of neighbors to use for constructing the cobordism.

    kwargs: key word arguments
        Passes to UMAP.

    Returns
    -------
    out: sparse graph
        Output cobordism.

    lt_ind: tuple
        Specifies the indices of the reference data set in the resulting cobordism.

    rt_ind: tuple
        Specifies the indices of the query data set in the resulting cobordism.
    """
    #Get the shape of the train data for indices
    lt_ind = (0,dat.shape[0])
    #Get indices for the right data set
    rt_ind = (dat.shape[0],dat.shape[0]+new_dat.shape[0])
    #Create separate fuzzy simplicial sets for train and test data
    g1 = umap.UMAP(
                n_neighbors = n_neighbors,
                random_state = random_state,
                transform_mode = 'graph',
                **kwargs).fit(dat)
    g2 = umap.UMAP(
                n_neighbors = n_neighbors,
                random_state = random_state,
                transform_mode = 'graph',
                **kwargs).fit(new_dat)

    #Use the simplicial sets to find mutual nearest neighbors (intersection to patch manifolds)
    g2_g1 = g1.transform(new_dat)
    g1_g2 = g2.transform(dat)

    #Mix simplicial sets to get the intersection patch in between train and test (consistent coordinate system)
    mix = umap_.general_simplicial_set_intersection(g2_g1.T, g1_g2, 0.5)
    #Stack horizontally the left with the mix to get that side
    a = scipy.sparse.hstack([g1.graph_,mix])
    #Stack vertically the mix with the right side
    b = scipy.sparse.vstack([mix,g2.graph_]).T
    #Combine the two to get the full dataset
    c = scipy.sparse.vstack([a,b])
    #Reset the local connectivity of the combined umap
    out = umap_.reset_local_connectivity(c)
    #return the output, left indices, and right indices
    return out, lt_ind, rt_ind

def multimodal_project_patched(out,overlay,lt_ind, rt_ind, prune=None):
    """Use cobordism to transfer data from reference data set to query data set.

    Parameters
    ----------
    out: sparse graph
        Output cobordism

    overlay: array (n_samples,n_features)
        Array representing reference data that will be interpolated to the query data.

    lt_ind: tuple
        Specifies the indices of the reference data set in the resulting cobordism.

    rt_ind: tuple
        Specifies the indices of the query data set in the resulting cobordism.

    prune: float (Default: None)
        Value in the interval [0-1] that specifies a threshold for edge weights
        in the cobordism. Any values lower than the set threshold will be pruned.

    Returns
    -------
    projections: sparse array
        Interpolated measures in the query data set from the reference overlay data.

    rt: sparse graph
        L1 normalized graph used to transfer data from reference to query data.
    """
    #use the indices of the patched UMAP to get only the right side
    rt = out[rt_ind[0]:rt_ind[1],lt_ind[0]:lt_ind[1]]
    #Optional to prune edges with < 0.05 probability
    if prune is not None:
        #Prune edges
        rt[rt<prune] = 0
    #Check to see if there are value lower than the probability of 5%
    if rt.max() < 0.05:
        #Raise a warning
        warnings.warn('Maximum connection strength across cobordism for some points is < 0.05.')
    #Use the right side test data and create normalized matrix to train data only
    rt = normalize(rt,'l1')
    #Create scipy matrix from overlay data
    overlay = scipy.sparse.csr_matrix(overlay)
    #Extract linear combination of values from the pseudo transition matrix
    projections = overlay.T.dot(rt.T).T
    #Return the new labels
    return projections, rt

def multimodal_project_patched_markov(out,overlay,lt_ind, rt_ind, prune=None):
    """Use cobordism to transfer data from reference data set to query data set by
    using L1 normalization across all cobordism structure, not just the reference
    to query geodesics.

    Parameters
    ----------
    out: sparse graph
        Output cobordism

    overlay: array (n_samples,n_features)
        Array representing reference data that will be interpolated to the query data.

    lt_ind: tuple
        Specifies the indices of the reference data set in the resulting cobordism.

    rt_ind: tuple
        Specifies the indices of the query data set in the resulting cobordism.

    prune: float (Default: None)
        Value in the interval [0-1] that specifies a threshold for edge weights
        in the cobordism. Any values lower than the set threshold will be pruned.

    Returns
    -------
    projections: sparse array
        Interpolated measures in the query data set from the reference overlay data.

    rt: sparse graph
        L1 normalized graph used to transfer data from reference to query data.
    """

    #Use simplicial set to normalize
    out = normalize(out,'l1')
    #use the indices of the patched UMAP to get only the right side
    rt = out[rt_ind[0]:rt_ind[1],lt_ind[0]:lt_ind[1]]
    #Optional to prune edges with < 0.05 probability
    if prune is not None:
        #Prune edges
        rt[rt<prune] = 0
    #Check to see if there are value lower than the probability of 5%
    #if rt.max() < 0.05:
        #Raise a warning
    #    warnings.warn('Maximum probability of connection for some points < 0.05.')
    #Create scipy matrix from overlay data
    overlay = scipy.sparse.csr_matrix(overlay)
    #Extract linear combination of values from the pseudo transition matrix
    projections = overlay.T.dot(rt.T).T
    #Return the new labels
    return projections, rt

def i_patchmap(dat,overlay,new_dat,random_state,n_neighbors,prune=None,all_markov = False, **kwargs):
    """i-PatchMAP workflow for transferring data across cobordism geodesics.

    Parameters
    ----------
    dat: array
        Reference data set.

    overlay: array (n_samples,n_features)
        Array representing reference data that will be interpolated to the query data.

    new_dat: array
        Query data set.

    random_state: integer
        Random state to set the seed for reproducibility.

    n_neighbors: integer
        Number of neighbors to use for constructing the cobordism.

    prune: float (Default: None)
        Value in the interval [0-1] that specifies a threshold for edge weights
        in the cobordism. Any values lower than the set threshold will be pruned.

    all_markov: Bool
        Compute L1 normalization across all cobordism instead of reference to query data.

    kwargs: key word arguments to pass to UMAP.

    Returns
    -------
    projections: sparse array
        Interpolated measures in the query data set from the reference overlay data.

    rt: sparse graph
        L1 normalized graph used to transfer data from reference to query data.

    out: sparse graph
        Output cobordism.
    """
    #Print update
    print('Computing Cobordism and Projecting Data...')
    #Run the composite manifolds
    out, lt_ind, rt_ind = compute_i_patchmap_cobordism(dat,new_dat,random_state,n_neighbors,**kwargs)

    #Check for all markov
    if all_markov:
        #Get the projections
        projections, rt = multimodal_project_patched_markov(out,overlay,lt_ind, rt_ind,prune)
    else:
        #Get the projections
        projections, rt = multimodal_project_patched(out,overlay,lt_ind, rt_ind,prune)
    #Print update
    print('Finished')
    #Return the predictions
    return projections, rt, out



def compute_batch_transforms(X_list, **kwargs):
    """
    X_list: list of data objects to be passed to umap.fit(). Must be same # variables
    """

    #Reorder the list to match input
    #in_list = [X_list[i] for i in order]
    #Stack the data for later
    stacked_data = np.vstack(X_list)
    #Compute the indices for batch
    indxs,counts = get_batch_indices(X_list)

    #Compute patched simplicial set
    cob = compute_cobordism(X_list,metric='cosine',**kwargs)

    #get size of list
    sz = len(X_list)
    #Create a return list
    return_list = []
    # iterate through batches
    for i in range(sz):
        idx = indxs[i]
        # ugly way to mask array
        mask_1 = np.zeros(cob.shape[0], dtype=bool)
        mask_1[idx[0]:idx[1]] = True
        mask_2 = np.zeros(cob.shape[0], dtype=bool)
        mask_2[0:idx[0]] = True
        mask_2[idx[1]:] = True
        #use the indices of the patched UMAP to get only the right side
        rt = cob[mask_1][:, mask_2]

        #Use the right side test data and create normalized matrix to train data only
        rt = normalize(rt,'l1')
        #Create scipy matrix from overlay data
        overlay = scipy.sparse.csr_matrix(stacked_data[mask_2])
        #Extract linear combination of values from the pseudo transition matrix
        translation_vects = overlay.T.dot(rt.T).T
        # subtract translation vectors from data
        new_dat = scipy.sparse.csr_matrix(stacked_data[mask_1]) + translation_vects
        # add to return list
        return_list.append(new_dat.toarray())
    #Return the transformed data
    return return_list, cob
