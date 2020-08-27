import javabridge
import bioformats
import scripts.io as io
import scripts.processing as processing
import scripts.analysis as analysis
import os
import pandas as pd

def batch_analysis(path, output_path, sigma, pixel_density, min_samples, **kwargs):

    """Go through evry image files in the directory (path).
    Parameters
    ----------
    path : str
    kwargs : dict
        Additional keyword-argument to be pass to the function:
         - imageformat
    """


    imageformat= kwargs.get('imageformat')
    imfilelist=[os.path.join(path,f) for f in os.listdir(path) if f.endswith(imageformat)]
    
    for file in imfilelist:
        
        print(file)
        javabridge.start_vm(class_path=bioformats.JARS)
        _, series = io._metadata(file)
        for serie in range(series):

            mip, directory, meta = io.load_TIFF(file, output_path, serie = serie)
            
            local_maxi, labels, gauss = processing.wide_clusters(mip, sigma=sigma, 
                                                pixel_density = pixel_density,
                                                min_samples = min_samples,
                                                plot=False)
            del mip
            ganglion_prop = processing.segmentation(gauss, local_maxi, labels, meta,
                                                    directory, save = True)
            del gauss
            analysis.create_dataframe(ganglion_prop, labels, local_maxi, meta, directory, save=True)
