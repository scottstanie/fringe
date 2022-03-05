#!/usr/bin/env python3

from array import array
import datetime
import glob
import os
import re
import numpy as np
from osgeo import gdal

def simulate_noise(corr_matrix: np.array) -> np.array:
    N = corr_matrix.shape[0]

    w, v = np.linalg.eigh(corr_matrix)
    w[w < 1e-3] = 0.0

    C = np.dot(v, np.dot(np.diag(np.sqrt(w)), np.matrix.getH(v)))
    Z = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)
    slc = np.dot(C, Z)

    return slc

def simulate_neighborhood_stack(corr_matrix: np.array, neighbor_samples: int = 200) -> np.array:

    nslc = corr_matrix.shape[0]
    # A 2D matrix for a neighborhood over time.
    # Each column is the neighborhood complex data for each acquisition date
    neighbor_stack = np.zeros((nslc, neighbor_samples), dtype=np.complex64)
    for ii in range(neighbor_samples):
        slcs = simulate_noise(corr_matrix)
        # To ensure that the neighborhood is homogeneous,
        # we set the amplitude of all SLCs to one
        neighbor_stack[:, ii] = np.exp(1J*np.angle(slcs))

    return neighbor_stack


def simulate_coherence_matrix(t, gamma0, gamma_inf, Tau0, ph):
    length = t.shape[0]
    C = np.ones((length, length), dtype=np.complex64)
    for ii in range(length):
        for jj in range(ii + 1, length):
            gamma = (gamma0 - gamma_inf) * np.exp((t[ii] - t[jj]) / Tau0) + gamma_inf
            C[ii, jj] = gamma * np.exp(1j * (ph[ii] - ph[jj]))
            C[jj, ii] = np.conj(C[ii, jj])

    return C


def simulate_phase_timeSeries(
        time_series_length: int = 365, acquisition_interval: int = 12, signal_rate: float = 1.0, std_random: float = 0, k: int = 1
        ):
    # time_series_length: length of time-series in days
    # acquisition_interval: time-differense between subsequent acquisitions (days)
    # signal_rate: linear rate of the signal (rad/year)
    # k: seasonal parameter, 1 for annaual and  2 for semi-annual
    t = np.arange(0, time_series_length, acquisition_interval)
    signal_phase = signal_rate * (t - t[0]) / 365.0
    if k > 0:
        seasonal = np.sin(2 * np.pi * k * t / 365.0) + np.cos(2 * np.pi * k * t / 365.0)
        signal_phase = signal_phase + seasonal

    # adding random temporal signal (which simulates atmosphere + DEM error + ...)
    signal_phase = signal_phase + std_random * np.random.randn(len(t))
    signal_phase = signal_phase - signal_phase[0]
    # wrap the phase to -pi to p
    signal_phase = np.angle(np.exp(1j * signal_phase))

    return signal_phase, t


def covariance(c1, c2):

    a1 = np.sum(np.abs(c1) ** 2)
    a2 = np.sum(np.abs(c2) ** 2)

    cov = np.sum(c1 * np.conjugate(c2)) / (np.sqrt(a1) * np.sqrt(a2))
    return cov


def compute_covariance_matrix(neighbor_stack):
    nslc = neighbor_stack.shape[0]
    cov_mat = np.zeros((nslc, nslc), dtype=np.complex64)
    for ti in range(nslc):
        for tj in range(ti + 1, nslc):
            cov = covariance(neighbor_stack[ti, :], neighbor_stack[tj, :])
            cov_mat[ti, tj] = cov
            cov_mat[tj, ti] = np.conjugate(cov)
        cov_mat[ti, ti] = 1.0

    return cov_mat


def estimate_evd(cov_mat):

    # estimate the wrapped phase based on the eigen value decomposition of the covariance matrix
    w, v = np.linalg.eigh(cov_mat)

    # the last eignevalue is the maximum eigenvalue
    # However let's check to make sure
    ind_max = np.argmax(w)

    # the eignevector corresponding to the largest eigenvalue
    # of the covariance matrix is the solution
    evd_estimate = v[:, ind_max]

    # refernce to the first acquisition
    evd_estimate = evd_estimate * np.conjugate(evd_estimate[0])

    return evd_estimate


def simulate_bit_mask(ny, nx, filename="neighborhood_map"):

    # number of uint32 bytes needed to store weights
    number_of_bytes = np.ceil((ny * nx) / 32)

    flags = np.ones((ny, nx), dtype=np.bool_)
    flag_bits = np.zeros((ny, nx), dtype=np.uint8)

    for ii in range(ny):
        for jj in range(nx):
            flag_bits[ii, jj] = flags[ii, jj].astype(np.uint8)

    # create the weight dataset for 1 neighborhood
    cols = nx
    rows = ny
    n_bands = int(number_of_bytes)
    drv = gdal.GetDriverByName("ENVI")
    options = ["INTERLEAVE=BIP"]
    ds = drv.Create(filename, cols, rows, n_bands, gdal.GDT_UInt32, options)

    half_window_y = int(ny / 2)
    half_window_x = int(nx / 2)

    ds.SetMetadata(
        {"HALFWINDOWX": str(half_window_x), "HALFWINDOWY": str(half_window_y)}
    )
    ds = None

    # above we created the ENVI hdr. Now let's write some data into the binary file

    # Let's assume in the neighborhood of nx*ny all pixels are
    # similar to the center pixel
    s = "1" * (nx * ny * n_bands * 4 * 8)

    bin_array = array("B")
    bits = s.ljust(n_bands * nx * ny * 4 * 8, "0")  # pad it to length n_bands*32

    for octect in re.findall(r"\d{8}", bits):  # split it in 4 octects
        bin_array.append(int(octect[::-1], 2))  # reverse them and append it

    with open(filename, "wb") as f:
        f.write(bytes(bin_array))

    return None


class BitMask:
    def __init__(self, ny, nx):
        """A BitMask class

        Parameters
        ----------
        ny: Number of lines
        nx: Number of pixels
        """
        self.ny = ny
        self.nx = nx

    def getbit(self, mask, ii, jj):
        flat = (ii + self.ny) * (2 * self.nx + 1) + jj + self.nx
        num = flat // 8
        bit = flat % 8
        return (mask[num] >> bit) & 1


def test_bit_mask(filename):
    """Load relevant data for a pixel."""
    ds = gdal.Open(filename, gdal.GA_ReadOnly)
    nx = int(ds.GetMetadataItem("HALFWINDOWX"))
    ny = int(ds.GetMetadataItem("HALFWINDOWY"))
    width = ds.RasterXSize
    bands = ds.RasterCount
    ds = None

    # we have only one neighborhood around one pixel
    line = 5
    pixel = 5

    fid = open(filename, "rb")
    fid.seek((line * width + pixel) * bands * 4)
    mask = fid.read(bands * 4)
    fid.close()

    npix = (2 * ny + 1) * (2 * nx + 1)
    masker = BitMask(ny, nx)

    bitmask = np.zeros(npix, dtype=bool)
    count = 0
    ind = 0
    for ii in range(-ny, ny + 1):
        for jj in range(-nx, nx + 1):
            flag = masker.getbit(mask, ii, jj)
            bitmask[ind] = flag == 1
            if flag:
                count += 1
            ind += 1

    return count


def write_slc_stack(neighbor_stack, output_slc_dir, nx, ny, dt=12):
    os.makedirs(output_slc_dir, exist_ok=False)

    nslc = neighbor_stack.shape[0]
    t0 = datetime.datetime(2022, 1, 1)
    for ii in range(nslc):
        t = t0 + datetime.timedelta(dt * ii)
        slc_dir = os.path.join(output_slc_dir, t.strftime("%Y%m%d"))
        os.makedirs(slc_dir, exist_ok=False)
        filename = os.path.join(slc_dir, t.strftime("%Y%m%d") + ".slc.full")
        drv = gdal.GetDriverByName("ENVI")
        ds = drv.Create(filename, nx, ny, 1, gdal.GDT_CFloat32)

        data = neighbor_stack[ii, :].reshape(ny, nx)
        ds.GetRasterBand(1).WriteArray(data)
        ds = None

    return None


def write_dummy_geometry(output_dir, nx, ny):
    os.makedirs(output_dir, exist_ok=False)
    lat_name = os.path.join(output_dir, "lat.rdr.full")
    lon_name = os.path.join(output_dir, "lon.rdr.full")
    data = np.ones((ny, nx), dtype=np.float64)
    for filename in [lat_name, lon_name]:
        drv = gdal.GetDriverByName("ENVI")
        ds = drv.Create(filename, nx, ny, 1, gdal.GDT_Float64)
        ds.GetRasterBand(1).WriteArray(data)
        ds = None

    return None


def get_dates(slc_dir):
    date_list = []
    slcs = glob.glob(os.path.join(slc_dir, "*.slc"))
    for slc in slcs:
        dd = os.path.basename(slc)
        dd = dd.replace(".slc", "")
        date_list.append(dd)

    date_list.sort()
    return date_list


def read_wrapped_phase(output_dir, x, y):
    date_list = get_dates(output_dir)
    nt = len(date_list)
    estimated_phase = np.zeros((nt), dtype=np.complex64)
    for ii in range(nt):
        filename = f"{output_dir}/{date_list[ii]}.slc"
        print(f"reading {filename}")
        ds = gdal.Open(filename, gdal.GA_ReadOnly)
        estimated_phase[ii] = ds.ReadAsArray(x, y, 1, 1)[0]
        ds = None
    return estimated_phase


def main():

    # simulate ideal wrapped phase series without noise and without any short lived signals
    signal_phase, t = simulate_phase_timeSeries(
        time_series_length=700,
        acquisition_interval=12,
        signal_rate=1,
        std_random=0.3,
        k=2,
    )

    # paraneters of a coherence model
    gamma0 = 0.999
    gamma_inf = 0.99
    Tau0 = 72
    # full neighborhood window size in y direction
    ny = 11
    # full neighborhood window size in x direction
    nx = 11
    # number of samples in the neighborhood
    neighbor_samples = ny * nx

    # simulate a complex covraince matrix based on the
    # simulated phase and coherence model
    simulated_covariance_matrix = simulate_coherence_matrix(
        t, gamma0, gamma_inf, Tau0, signal_phase
    )

    # simulate a neighborhood of SLCs with size of
    # neighbor_samples for Nt acquisitions
    neighbor_stack = simulate_neighborhood_stack(
        simulated_covariance_matrix, neighbor_samples=neighbor_samples
    )

    # estimate complex covariance matrix from the neighbor stack
    estimated_covariance_matrix = compute_covariance_matrix(neighbor_stack)

    # estimate wrapped phase with a prototype estimator of EVD
    evd_estimate = estimate_evd(estimated_covariance_matrix)

    # compute residual which is the phase difference between
    # simulated and estimated wrapped phase series
    residual_evd = np.angle(np.exp(1j * signal_phase) * np.conjugate(evd_estimate))
    # RMSE of the residual phase
    rmse_evd = np.sqrt(np.sum(residual_evd**2, 0) / len(t))

    #######################
    # write nmap which all the neighbors are self similar with the center pixel
    weight_dataset_name = "neighborhood_map"
    simulate_bit_mask(ny, nx, filename=weight_dataset_name)
    test_bit_mask(weight_dataset_name)

    # output directory to store the simulated data for this unit test
    output_simulation_dir = "simulations"
    # output subdirectory to store SLCs
    output_slc_dir = os.path.join(output_simulation_dir, "SLC")
    # write flat binary SLCs that Fringe can read
    write_slc_stack(neighbor_stack, output_slc_dir, nx, ny)
    # a dummy geometry directory similar to isce2 results
    output_geometry_dir = os.path.join(output_simulation_dir, "geom_reference")
    write_dummy_geometry(output_geometry_dir, nx, ny)

    # different subdirectories for fringe outputs
    output_timeseries_dir = os.path.join(output_simulation_dir, "timeseries")
    coreg_stack_dir = os.path.join(output_timeseries_dir, "coreg_stack")
    geometry_stack_dir = os.path.join(output_timeseries_dir, "geometry")
    slc_stack_dir = os.path.join(output_timeseries_dir, "slcs")

    #create a VRT pointing to the stack
    cmd = f"tops2vrt.py -i {output_simulation_dir} -s {coreg_stack_dir} -g {geometry_stack_dir} -c {slc_stack_dir}"
    os.system(cmd)

    # estimate neighborhood map with fringe
    nmap_output = os.path.join(output_simulation_dir, "KS2/nmap")
    count_output = os.path.join(output_simulation_dir, "KS2/count")
    cmd = (
        f"nmap.py -i {coreg_stack_dir}/slcs_base.vrt -o {nmap_output} -c {count_output}"
    )
    os.system(cmd)

    # run fringe evd module with EVD estimator
    evd_output = os.path.join(output_timeseries_dir, "evd")
    cmd = f"evd.py -i {coreg_stack_dir}/slcs_base.vrt -w {nmap_output} -o {evd_output} -m EVD"
    os.system(cmd)

    # run fringe evd module with MLE estimator
    mle_output = os.path.join(output_timeseries_dir, "mle")
    cmd = f"evd.py -i {coreg_stack_dir}/slcs_base.vrt -w {nmap_output} -o {mle_output} -m MLE"
    os.system(cmd)

    # run fringe phase_linking module
    phase_link_output = os.path.join(output_timeseries_dir, "phase_link")
    cmd = f"phase_link.py -i {coreg_stack_dir}/slcs_base.vrt -w {nmap_output} -o {phase_link_output} -m MLE"
    os.system(cmd)

    # read the estimated wrapped phase
    # Pixel of interest is at the center of the neighborhood box
    x0 = int(nx / 2)
    y0 = int(ny / 2)
    est_wrapped_phase_fringe_evd = read_wrapped_phase(evd_output, x0, y0)

    est_wrapped_phase_fringe_mle = read_wrapped_phase(mle_output, x0, y0)

    est_wrapped_phase_fringe_phase_link = read_wrapped_phase(phase_link_output, x0, y0)

    # compare with simulated phase and calculate RMSE
    print(signal_phase.shape)
    residual_evd = np.angle(
        np.exp(1j * signal_phase) * np.conjugate(est_wrapped_phase_fringe_evd)
    )
    rmse_fringe_evd = np.degrees(np.sqrt(np.sum(residual_evd**2, 0) / len(t)))

    residual_mle = np.angle(
        np.exp(1j * signal_phase) * np.conjugate(est_wrapped_phase_fringe_mle)
    )
    rmse_fringe_mle = np.degrees(np.sqrt(np.sum(residual_mle**2, 0) / len(t)))

    residual_phase_link = np.angle(
        np.exp(1j * signal_phase) * np.conjugate(est_wrapped_phase_fringe_phase_link)
    )
    rmse_fringe_phase_link = np.degrees(np.sqrt(np.sum(residual_phase_link**2, 0) / len(t)))

    print("rmse for evd [degrees]:", np.degrees(rmse_evd))
    print("rmse for evd fringe [degrees]:", rmse_fringe_evd)
    print("rmse for mle fringe [degrees]:", rmse_fringe_mle)
    print("rmse for phase-link fringe [degrees]:", rmse_fringe_phase_link)


    #######################
    # for debugging purpose
    plot_flag = False

    if plot_flag:
        import matplotlib.pyplot as plt

        plt.figure(1)

        plt.subplot(2, 2, 1)
        plt.imshow(np.abs(simulated_covariance_matrix), vmin=0, vmax=1)

        plt.subplot(2, 2, 2)
        plt.imshow(np.abs(estimated_covariance_matrix), vmin=0, vmax=1)

        plt.subplot(2, 2, 3)
        plt.imshow(np.angle(simulated_covariance_matrix), vmin=-np.pi, vmax=np.pi)

        plt.subplot(2, 2, 4)
        plt.imshow(np.angle(estimated_covariance_matrix), vmin=-np.pi, vmax=np.pi)

        plt.figure(2)
        plt.plot(signal_phase, "-", linewidth=4)
        plt.plot(np.angle(evd_estimate), "--*")
        plt.plot(np.angle(est_wrapped_phase_fringe_evd), "-^", ms=10)
        plt.plot(np.angle(est_wrapped_phase_fringe_mle), "--s", ms=4)
        plt.legend(
            [
                "simulated",
                "estimated evd (python)",
                "estimated EVD fringe",
                "estimated MLE fringe",
            ]
        )
        plt.show()

    # check the RMSE of the FRINGE results
    assert rmse_fringe_evd <= 10
    assert rmse_fringe_mle <= 10
    assert rmse_fringe_phase_link <= 10

    # check the neighborhood map
    count = test_bit_mask(weight_dataset_name)
    print(f"count: {count}")
    # we have simulated an ideah homogeneous neighborhood of nx*ny.
    # Therefore the number of self-similar pixels in the neighborhood
    # should be nx*ny
    assert count == nx * ny

if __name__ == "__main__":

    main()
