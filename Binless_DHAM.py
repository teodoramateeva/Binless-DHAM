import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

class BinlessDHAM:
    def __init__(self, data, force, constr, lagtime=1000, T=303.15, R=0.00198720425864083):
        self.data = data
        self.force = force
        self.constr = constr
        self.lagtime = lagtime
        self.T = T
        self.R = R
        self.Nimage, self.Nwindow = data.shape
        self.kBT = 0.6024209229
        self.qspace, self.bin_width, self.bin_center = self.define_RC()
        self.ncount = np.array([np.histogram(data[k, :], bins=self.qspace)[0] for k in range(self.Nimage)])

    def define_RC(self):
        v_min, v_max = self.data.min() - 0.000001, self.data.max() + 0.000001
        # define the number of bins here
        numbins_q = 1000
        # create a dense grid to discretize the reaction coordinate
        qspace = np.linspace(v_min, v_max, numbins_q + 1)
        # define the bin edges
        bin_width = qspace[1] - qspace[0]
        bin_center = qspace[:-1]
        return qspace, bin_width, bin_center

    def compute_transition_matrices(self):
        # call the optimized numba-accelerated function
        Mnobin, Mbin, avg_deltaq = compute_transition_matrices_numba(
            self.data,
            self.qspace,
            self.bin_center,
            self.bin_width,
            self.ncount,
            self.force,
            self.constr,
            self.lagtime,
            self.kBT,
            self.Nimage,
            self.Nwindow
        )
        return Mnobin, Mbin, avg_deltaq

    def analyze_matrix(self):
        # get the transition matrix with and without binning corrections and the average displacement
        Mnobin, Mbin, avg_deltaq = self.compute_transition_matrices()

        # normalize the matrix
        MM = Mbin.copy()
        msum = MM.sum(axis=1)
        MM = np.divide(MM.T, msum, where=msum > 0).T
        
        # perform eigenvalue calculation
        eigvals, eigvecs = np.linalg.eig(MM.T)
        sorted_indices = np.argsort(eigvals)
        second_largest_eigenvalue = eigvals[sorted_indices[-2]]
        print(f"Second-largest eigenvalue: {second_largest_eigenvalue}")
        
        tau = -1 / np.log(eigvals[sorted_indices[-2]])
        print(f"tau is: {tau}")

        # relaxation time 𝜏 is inversely proportional to the rate constant 𝑘
        rate = 1 / tau
        print(f"the rate is: {rate}")

        R = 0.00198720425864083  # kcal/mol·K
        T = 303.15  
        kB = 0.001987

        # calculate deltaG based on the provided relationships
        # deltaG = -R * T * np.log(1 / (kB * T * tau))
        # print(f"deltaG is: {deltaG}")

        # find the equilibrium populations
        mpeq = eigvecs[:, sorted_indices[-1]].real
        mpeq /= mpeq.sum()
        mU2 = -self.kBT * np.log(mpeq)
        mU2 = np.nan_to_num(mU2, nan=0.0)
        mUbin = mU2 - mU2[1]

        # adjust the free energy values & normalize to zero
        mUbin -= mUbin[0]
        self.plot_free_energy(mUbin, avg_deltaq)

    def filter_and_adjust_data(self, mUbin):
        x_values = self.bin_center + self.bin_width / 2
        mask = (x_values > -35) & (x_values < 35)
        filtered_x = x_values[mask]
        filtered_mUbin = mUbin[mask]
        filtered_mUbin -= filtered_mUbin[0]
        return filtered_x, filtered_mUbin

    def plot_free_energy(self, mUbin, avg_deltaq):
        filtered_x, filtered_mUbin = self.filter_and_adjust_data(mUbin)

        plt.figure(figsize=(10, 6))
        plt.plot(filtered_x, filtered_mUbin, 'r-o', label='DHAM', linewidth=2)
        plt.xlabel('x (Å)', fontsize=14)
        plt.ylabel('Free Energy (kcal/mol)', fontsize=14)
        plt.ylim([np.min(filtered_mUbin) - 1, np.max(filtered_mUbin) + 1])
        plt.legend()
        plt.title(
            f'lagtime={self.lagtime} numbin={len(self.bin_center)}\n'
            f'<|Δx|>={avg_deltaq:.4f} binsize={self.bin_width:.4f}', fontsize=12
        )
        plt.show()

# numba-accelerated function
@njit(parallel=True)
def compute_transition_matrices_numba(data, qspace, bin_center, bin_width, ncount, force, constr, lagtime, kBT, Nimage, Nwindow):
    num_bins = len(qspace) - 1
    
    # initialize the transition matrices
    # Mnobin = transition matrix without binning corrections
    # Mbin = transition matrix with binning corrections
    
    Mnobin = np.zeros((num_bins, num_bins))
    Mbin = np.zeros((num_bins, num_bins))
    deltaq = 0.0
    
    # the shape depends on Nimage, Nwindow
    bin_indices = np.digitize(data, qspace) - 1  

    for k in prange(Nimage):
        bi_lag_indices = bin_indices[k, :-lagtime]
        bi_indices = bin_indices[k, lagtime:]

        for i in range(len(bi_lag_indices)):
            bi_lag = bi_lag_indices[i]
            bi = bi_indices[i]
            
            deltaq += abs(data[k, lagtime + i] - data[k, i])

            msum = 0.0
            msumbin = 0.0

            # vectorized computation over all images
            for l in range(Nimage):
                nc = ncount[l, bi_lag]
                if nc > 0:
                    fc = 0.5 * force[l] / kBT
                    ui = fc * (constr[l] - data[k, i])**2
                    uj = fc * (constr[l] - data[k, lagtime + i])**2
                    uibin = fc * (constr[l] - (bin_center[bi_lag] + bin_width / 2))**2
                    ujbin = fc * (bin_center[bi] + bin_width / 2 - constr[l])**2

                    exp_factor = np.exp(-(uj - ui) / 2)
                    exp_factor_bin = np.exp(-(ujbin - uibin) / 2)

                    msum += nc * exp_factor
                    msumbin += nc * exp_factor_bin

            if msum > 0:
                Mnobin[bi_lag, bi] += 1 / msum
            if msumbin > 0:
                Mbin[bi_lag, bi] += 1 / msumbin

    # calculate the average deltaq (displacement)
    avg_deltaq = deltaq / (Nimage * (Nwindow - lagtime))
    return Mnobin, Mbin, avg_deltaq
