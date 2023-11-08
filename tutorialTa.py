
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