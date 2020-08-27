import os

def log_file(directory, meta, **kwargs):
    sigma = kwargs['sigma']
    min_distance = kwargs['min_distance']
    eps = kwargs['eps']
    min_samples = kwargs['min_samples']

    with open(directory+'/'+meta["Name"],'w') as file:
        file.write('Size of the Sigma used for the bluring of the image : {}\n'\
                    'Distance minimum between 2 peaks (nucleus): {}\n'\
                    'Minimum distance between cells within a cluster: {}\n'\
                    'Minimum amount of cells to form a cluster: {}\n'\
                    .format(str(sigma),str(min_distance),
                    str(eps), str(min_samples)))
