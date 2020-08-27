import bioformats
import numpy as np
import os
import time


def _metadata(path, serie = 0):
    xml = bioformats.get_omexml_metadata(path)
    md = bioformats.omexml.OMEXML(xml)

    series = md.image_count

    meta={'AcquisitionDate': md.image(serie).AcquisitionDate}
    meta['Name']=md.image(serie).Name.replace(' ', '_')
    meta['SizeC']=md.image(serie).Pixels.SizeC
    meta['SizeT']=md.image(serie).Pixels.SizeT
    meta['SizeX']=md.image(serie).Pixels.SizeX
    meta['SizeY']=md.image(serie).Pixels.SizeY
    meta['SizeZ']=md.image(serie).Pixels.SizeZ
    meta['DimensionOrder']=md.image(serie).Pixels.DimensionOrder
    meta['PhysicalSizeX'] = md.image(serie).Pixels.PhysicalSizeX
    meta['PhysicalSizeY'] = md.image(serie).Pixels.PhysicalSizeY
    meta['PhysicalSizeZ'] = md.image(serie).Pixels.PhysicalSizeZ
    
    return(meta, series)


def load_bioformats(path, serie = 0):
    meta, _ = _metadata(path, serie = serie)
    image = np.empty((meta['SizeT'], meta['SizeZ'], meta['SizeX'], meta['SizeY'], meta['SizeC']))
    
    
    with bioformats.ImageReader(path) as rdr:
        for t in range(0, meta['SizeT']):
            for z in range(0, meta['SizeZ']):
                for c in  range(0, meta['SizeC']):
                    image[t,z,:,:,c]=rdr.read(c=c, z=z, t=t, series=serie,
                                                 index=None, rescale=False, wants_max_intensity=False,
                                                 channel_names=None)
    img = np.squeeze(image)
    if img.ndim == 3:
        mip = np.amax(img, 0)
    elif img.ndim == 4:
        mip = np.amax(img[:,:,:,1], 0)
    else:
        print("need to correct for ndim > 3")

    directory = _new_directory(path, meta)

    return(img, directory, mip, meta)

def load_TIFF(path, opath, serie = 0):
    meta, _ = _metadata(path, serie = serie)
    image = np.empty((meta['SizeX'], meta['SizeY'], meta['SizeC']))

    #with bioformats.ImageReader(path) as rdr:
    #    for t in range(0, meta['SizeT']):
    #        for z in range(0, meta['SizeZ']):
    #            for c in  range(0, meta['SizeC']):
    #                image[:,:,c]=rdr.read(c=c, z=z, t=t, series=serie,
    #                                      index=None, rescale=False, wants_max_intensity=False,
    #                                      channel_names=None)
    
    img=bioformats.load_image(path, c=0)

    directory = _new_directory(opath, meta)

    return(img, directory, meta)

def _new_directory(path, meta):

    directory = str(path)+"/"+"result"+'_'+meta["Name"]+'_'+ time.strftime('%m'+'_'+'%d'+'_'+'%Y')
    if os.path.exists(directory):
        expand = 0
        while True:
            expand += 1
            new_directory = directory+"_"+str(expand)
            if os.path.exists(new_directory):
                continue
            else:
                directory = new_directory
                os.makedirs(directory)
                break
    else:
        os.makedirs(directory)
    return(directory)
