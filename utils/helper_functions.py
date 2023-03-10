import csv
import numpy as np



def read_data(filename) :
    
    with open(filename, "r") as f:
        heading = next(f)
        reader_obj = csv.reader(f)
        data = np.array([raw for raw in reader_obj])   
        
    fdata = []
    for _, d in enumerate(data) :
        fdata.append([float(value) for _, value in enumerate(d) ])
    
    fdata = np.array(fdata)

    return fdata



