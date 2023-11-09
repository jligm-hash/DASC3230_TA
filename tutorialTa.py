
# mnist reshape



def mnist_reshape(tmp):
    '''
    input: pandas series for one row
    output: np.array of (28, 28)
    '''
    return(tmp.to_numpy().reshape(28,28))


def mnist_imshow(tmp, cmap = 'Greys'):
    '''
    input: pandas series
    output: img for one digit
    '''
    import matplotlib.pyplot as plt
    
    return(plt.imshow(mnist_reshape(tmp), interpolation = "none", cmap = cmap))


def plotLabelDistribution4sc_tsne(labelArr, X_embedded, colorArr):
    '''
    input: label arr, x_embedded
    output: tsne
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    
    # colorArr = ["#9e0142", "#d53e4f", "#f46d43", "#fdae61", "#fee08b", "#e6f598", "#abdda4", "#66c2a5", "#3288bd", "#5e4fa2"]
    
    plt.figure()
    for tmpIndex, tmpLabel in enumerate(np.unique(labelArr)):
        plt.scatter(X_embedded[labelArr == tmpLabel, 0], 
                    X_embedded[labelArr == tmpLabel, 1],
                    c = colorArr[tmpIndex], 
                    label = tmpLabel)
    plt.xlabel('tSNE1')
    plt.ylabel('tSNE2')
    plt.legend()
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    # plt.axis('equal')
    
    
def plot_dendrogram(model, **kwargs):
    '''
    source: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py
    '''
    # Create linkage matrix and then plot the dendrogram
    
    import numpy as np
    from scipy.cluster.hierarchy import dendrogram
    
    
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    
    
def plotLabelDistribution4moon(tmpLabelArr, feature4moon):
    '''
    input: label arr, x_embedded
    output: tsne
    '''
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure()
    for tmpIndex, tmpLabel in enumerate(np.unique(tmpLabelArr)):
        plt.scatter(feature4moon[tmpLabelArr == tmpLabel, 0], 
                    feature4moon[tmpLabelArr == tmpLabel, 1],
                    # c = colorArr[tmpIndex], 
                    label = tmpLabel)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    
    
    
def make_moons2(n_samples=100, *, shuffle=True, noise=None, random_state=None, setDist=0.5):
    """
    source: https://github.com/scikit-learn/scikit-learn/blob/093e0cf14/sklearn/datasets/_samples_generator.py#L786
    
    Make two interleaving half circles.

    A simple toy dataset to visualize clustering and classification
    algorithms. Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int or tuple of shape (2,), dtype=int, default=100
        If int, the total number of points generated.
        If two-element tuple, number of points in each of two moons.

        .. versionchanged:: 0.23
           Added two-element tuple.

    shuffle : bool, default=True
        Whether to shuffle the samples.

    noise : float, default=None
        Standard deviation of Gaussian noise added to the data.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset shuffling and noise.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
        The generated samples.

    y : ndarray of shape (n_samples,)
        The integer labels (0 or 1) for class membership of each sample.
    """

    # if isinstance(n_samples, numbers.Integral):
    #     n_samples_out = n_samples // 2
    #     n_samples_in = n_samples - n_samples_out
    # else:
    #     try:
    #         n_samples_out, n_samples_in = n_samples
    #     except ValueError as e:
    #         raise ValueError(
    #             "`n_samples` can be either an int or a two-element tuple."
    #         ) from e
    
#     import array
    import numbers
#     import warnings
#     from collections.abc import Iterable
#     from numbers import Integral, Real

    import numpy as np
#     import scipy.sparse as sp
#     from scipy import linalg

#     from ..preprocessing import MultiLabelBinarizer
#     from ..utils import check_array, check_random_state
#     from ..utils import shuffle as util_shuffle
#     from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
#     from ..utils.random import sample_without_replacement



    n_samples_out, n_samples_in = n_samples, n_samples
    generator = check_random_state(random_state)
    
    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - setDist

    X = np.vstack(
        [np.append(outer_circ_x, inner_circ_x), np.append(outer_circ_y, inner_circ_y)]
    ).T
    y = np.hstack(
        [np.zeros(n_samples_out, dtype=np.intp), np.ones(n_samples_in, dtype=np.intp)]
    )

    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

    if noise is not None:
        X += generator.normal(scale=noise, size=X.shape)

    return X, y


def check_random_state(seed):
    """
    source: https://github.com/scikit-learn/scikit-learn/blob/714c50092b1bfafad2b568a07d635d5db1301d58/sklearn/utils/validation.py#L1262
    
    Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    :class:`numpy:numpy.random.RandomState`
        The random state object based on `seed` parameter.
    """
    import numpy as np
    import numbers
        
        
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )


    
def util_shuffle(*arrays, random_state=None, n_samples=None):
    """
    Source: https://github.com/scikit-learn/scikit-learn/blob/714c50092b1bfafad2b568a07d635d5db1301d58/sklearn/utils/__init__.py#L633
    Shuffle arrays or sparse matrices in a consistent way.

    This is a convenience alias to ``resample(*arrays, replace=False)`` to do
    random permutations of the collections.

    Parameters
    ----------
    *arrays : sequence of indexable data-structures
        Indexable data-structures can be arrays, lists, dataframes or scipy
        sparse matrices with consistent first dimension.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for shuffling
        the data.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    n_samples : int, default=None
        Number of samples to generate. If left to None this is
        automatically set to the first dimension of the arrays.  It should
        not be larger than the length of arrays.

    Returns
    -------
    shuffled_arrays : sequence of indexable data-structures
        Sequence of shuffled copies of the collections. The original arrays
        are not impacted.

    See Also
    --------
    resample : Resample arrays or sparse matrices in a consistent way.

    Examples
    --------
    It is possible to mix sparse and dense arrays in the same run::

      >>> import numpy as np
      >>> X = np.array([[1., 0.], [2., 1.], [0., 0.]])
      >>> y = np.array([0, 1, 2])

      >>> from scipy.sparse import coo_matrix
      >>> X_sparse = coo_matrix(X)

      >>> from sklearn.utils import shuffle
      >>> X, X_sparse, y = shuffle(X, X_sparse, y, random_state=0)
      >>> X
      array([[0., 0.],
             [2., 1.],
             [1., 0.]])

      >>> X_sparse
      <3x2 sparse matrix of type '<... 'numpy.float64'>'
          with 3 stored elements in Compressed Sparse Row format>

      >>> X_sparse.toarray()
      array([[0., 0.],
             [2., 1.],
             [1., 0.]])

      >>> y
      array([2, 1, 0])

      >>> shuffle(y, n_samples=2, random_state=0)
      array([0, 1])
    """
    return resample(
        *arrays, replace=False, n_samples=n_samples, random_state=random_state
    )

def resample(*arrays, replace=True, n_samples=None, random_state=None, stratify=None):
    """
    Source: https://github.com/scikit-learn/scikit-learn/blob/714c50092b1bfafad2b568a07d635d5db1301d58/sklearn/utils/__init__.py#L483
    Resample arrays or sparse matrices in a consistent way.

    The default strategy implements one step of the bootstrapping
    procedure.

    Parameters
    ----------
    *arrays : sequence of array-like of shape (n_samples,) or \
            (n_samples, n_outputs)
        Indexable data-structures can be arrays, lists, dataframes or scipy
        sparse matrices with consistent first dimension.

    replace : bool, default=True
        Implements resampling with replacement. If False, this will implement
        (sliced) random permutations.

    n_samples : int, default=None
        Number of samples to generate. If left to None this is
        automatically set to the first dimension of the arrays.
        If replace is False it should not be larger than the length of
        arrays.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for shuffling
        the data.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    stratify : array-like of shape (n_samples,) or (n_samples, n_outputs), \
            default=None
        If not None, data is split in a stratified fashion, using this as
        the class labels.

    """
    import numpy as np
    
    max_n_samples = n_samples
    random_state = check_random_state(random_state)

    if len(arrays) == 0:
        return None

    first = arrays[0]
    n_samples = first.shape[0] if hasattr(first, "shape") else len(first)

    if max_n_samples is None:
        max_n_samples = n_samples
    elif (max_n_samples > n_samples) and (not replace):
        raise ValueError(
            "Cannot sample %d out of arrays with dim %d when replace is False"
            % (max_n_samples, n_samples)
        )

    check_consistent_length(*arrays)

    if stratify is None:
        if replace:
            indices = random_state.randint(0, n_samples, size=(max_n_samples,))
        else:
            indices = np.arange(n_samples)
            random_state.shuffle(indices)
            indices = indices[:max_n_samples]
    else:
        # Code adapted from StratifiedShuffleSplit()
        y = check_array(stratify, ensure_2d=False, dtype=None)
        if y.ndim == 2:
            # for multi-label y, map each distinct row to a string repr
            # using join because str(row) uses an ellipsis if len(row) > 1000
            y = np.array([" ".join(row.astype("str")) for row in y])

        classes, y_indices = np.unique(y, return_inverse=True)
        n_classes = classes.shape[0]

        class_counts = np.bincount(y_indices)

        # Find the sorted list of instances for each class:
        # (np.unique above performs a sort, so code is O(n logn) already)
        class_indices = np.split(
            np.argsort(y_indices, kind="mergesort"), np.cumsum(class_counts)[:-1]
        )

        n_i = _approximate_mode(class_counts, max_n_samples, random_state)

        indices = []

        for i in range(n_classes):
            indices_i = random_state.choice(class_indices[i], n_i[i], replace=replace)
            indices.extend(indices_i)

        indices = random_state.permutation(indices)

    # convert sparse matrices to CSR for row-based indexing
    arrays = [a.tocsr() if issparse(a) else a for a in arrays]
    resampled_arrays = [_safe_indexing(a, indices) for a in arrays]
    if len(resampled_arrays) == 1:
        # syntactic sugar for the unit argument case
        return resampled_arrays[0]
    else:
        return resampled_arrays
    
    
    
def check_consistent_length(*arrays):
    """
    Source: https://github.com/scikit-learn/scikit-learn/blob/714c50092b1bfafad2b568a07d635d5db1301d58/sklearn/utils/validation.py#L397
    Check that all arrays have consistent first dimensions.

    Checks whether all objects in arrays have the same shape or length.

    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """
    
    import numpy as np

    lengths = [_num_samples(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError(
            "Found input variables with inconsistent numbers of samples: %r"
            % [int(l) for l in lengths]
        )