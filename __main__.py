import numpy as np

if __name__ == "__main__":
    # import the necessary class 
    from Binless_DHAM import BinlessDHAM

    data = np.load('coorData.npy').T
    force = np.full(data.shape[0], 10.0)
    constr = np.linspace(-38, 38, data.shape[0])

    # initialize the Binless DHAM model
    dham = BinlessDHAM(data, force, constr)
    relax_results = dham.analyze_matrix()
