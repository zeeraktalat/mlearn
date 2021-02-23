import numpy as np
cimport numpy as np
import torch
cimport cython

def npTOtorch(self,encoded):
    encoded = torch.from_numpy(encoded)
    return [encoded] # this forms a 2d array where the second dimension is just 1

@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef np.ndarray[np.int64_t, ndim=1] encode_doc_TEST( data_object , text):
    cdef int ix
    cdef int dim2 = len(text)
    
    encoded = np.zeros(dim2, dtype=np.dtype("i"))
    for ix in range(dim2):
        if text[ix] == '<pad>':
            break
        encoded[ix] = data_object.stoi.get(text[ix],data_object.unk_tok)

    #|||| 
    
    # encoded.from_numpy() # this makes a 1-D tensor from pytorch TODO: Fails for now fix it lmao

    return encoded

    # Want to make the data structure TEnsor that is the tensor, then we can fill it in with the encoded line

    #Is to get rid of the if else conditions 

# Write a method in dataset claass that takes this output and turns it into a tensor 

# make sure that the padding token is always the first token or like the first index in the array, right now it is the last index so in
# dataset.py in mlearn swap lines (320,321) and then move them to 309/310, and then hardcode padding token as 0 and unk token as 1
# TODO: DONE

    