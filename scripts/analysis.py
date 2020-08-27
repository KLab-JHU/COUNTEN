import pandas as pd
import numpy as np

def create_dataframe(ganglion_prop, labels, local_maxi, meta, directory, save=False):
    if not ganglion_prop:
        df = pd.DataFrame(np.zeros((1,6)), columns = ['ganglion', 'Nbr of neurons',
                                           "surface ganglion", "major axis length",
                                          "minor axis length", "orientation"])
    else:
        result = []
        intergang = 0
        label_new = np.copy(labels)
        label_new[labels == 0] = max(labels)+1

        for prop in ganglion_prop:
            label = prop.label
            number_of_neurons = len(np.where(label_new == label)[0])
            
            result.append((label, number_of_neurons))
            intergang = intergang + number_of_neurons
            
        result = np.asarray(result)
        surface_gang = [prop.area for prop in ganglion_prop]
        major_gang = [prop.major_axis_length for prop in ganglion_prop]
        minor_gang = [prop.minor_axis_length for prop in ganglion_prop]
        orientation_gang = [prop.orientation for prop in ganglion_prop]

        df = pd.DataFrame(result, columns = ['ganglion', 'Nbr of neurons'])
        
        df["surface area"] = surface_gang
        df["major axis length"] = major_gang
        df["minor axis length"] = minor_gang
        df["orientation"] = orientation_gang
        df.loc[:, 'major axis length']*=(meta['PhysicalSizeX'])
        df.loc[:, 'minor axis length']*=(meta['PhysicalSizeX'])
        df = df.rename(columns={"minor axis length": "minor axis length in um"})
        df = df.rename(columns={"major axis length": "major axis length in um"})
        df = df.rename(columns={"surface area": "surface area in um2"})
        
        dist = df.hist(column = ['Nbr of neurons'], bins=20)
        
        
        df = df.replace({"ganglion" : 0},"total in field")

        df.loc[-1] =  np.nan
        df.index = df.index +1
        df = df.sort_index()
        df['ganglion'][0] = "total in field"
        df.loc[0,"Nbr of neurons"] = len(local_maxi)
        
        extragang = df.loc[0,"Nbr of neurons"] - intergang
        df = df.append({'ganglion' : "intra-ganglionic total", 'Nbr of neurons' : intergang}, ignore_index=True)
        df = df.append({'ganglion' : "extra-ganglionic total", 'Nbr of neurons' : extragang}, ignore_index=True)
    
    if save == True:
        try:
            df.to_csv(directory+'/'+'{}.csv'.format(meta["Name"]))
        except FileNotFoundError:
            df.to_csv('{}.csv'.format(meta["Name"]))
            
    
    return(df, dist)
