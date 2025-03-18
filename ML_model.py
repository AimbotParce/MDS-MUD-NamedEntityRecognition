
from CRF import *

class ML_model:

    ## --------------------------------------------------
    ## Constructor: Load model from file
    ## --------------------------------------------------

    def __init__(self, datafile):

        if datafile[-4:]==".crf" :
            # load CRF model
            self._model = CRF(datafile)

        else:
            print("Unknown model type",datafile[-3:])
            exit()

    ## --------------------------------------------------
    ## Call predictor on a sequence
    ## --------------------------------------------------
            
    def predict(self, xseq) :
        return self._model.predict(xseq)


