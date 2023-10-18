
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


