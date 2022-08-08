#!/usr/bin/env python3
import datetime
import glob
import itertools
import os
import re
from array import array
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from numba import njit
from osgeo import gdal


@njit(cache=True)
def simulate_noise(corr_matrix: np.array) -> np.array:
    N = corr_matrix.shape[0]

    w, v = np.linalg.eigh(corr_matrix)
    w[w < 1e-3] = 0.0
    w = w.astype(v.dtype)

    vstar = np.conj(v.T)  # Hermetian
    C = v @ np.diag(np.sqrt(w)) @ vstar
    z = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)
    z = z.astype(C.dtype)
    return C @ z
    # slc = np.dot(C, z)


@njit(cache=True)
def simulate_neighborhood_stack(
    corr_matrix: np.array, neighbor_samples: int = 200
) -> np.array:

    nslc = corr_matrix.shape[0]
    # A 2D matrix for a neighborhood over time.
    # Each column is the neighborhood complex data for each acquisition date
    neighbor_stack = np.zeros((nslc, neighbor_samples), dtype=np.complex64)
    for ii in range(neighbor_samples):
        slcs = simulate_noise(corr_matrix)
        # To ensure that the neighborhood is homogeneous,
        # we set the amplitude of all SLCs to one
        neighbor_stack[:, ii] = np.exp(1j * np.angle(slcs))

    return neighbor_stack


@njit(cache=True)
def simulate_coherence_matrix(t, gamma0, gamma_inf, Tau0, ph):
    length = t.shape[0]
    C = np.ones((length, length), dtype=np.complex64)
    for ii in range(length):
        for jj in range(ii + 1, length):
            gamma = (gamma0 - gamma_inf) * np.exp((t[ii] - t[jj]) / Tau0) + gamma_inf
            C[ii, jj] = gamma * np.exp(1j * (ph[ii] - ph[jj]))
            C[jj, ii] = np.conj(C[ii, jj])

    return C


@njit(cache=True)
def simulate_phase_timeseries(
    time_series_length: int = 365,
    acquisition_interval: int = 12,
    signal_rate: float = 1.0,
    std_random: float = 0,
    k: int = 1,
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


@njit(cache=True)
def covariance(c1, c2):

    a1 = np.sum(np.abs(c1) ** 2)
    a2 = np.sum(np.abs(c2) ** 2)

    cov = np.sum(c1 * np.conjugate(c2)) / (np.sqrt(a1) * np.sqrt(a2))
    return cov


@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
def estimate_temp_coh(est, cov_matrix):
    gamma = 0
    N = len(est)
    count = 0
    for i in range(N):
        for j in range(i + 1, N):
            theta = np.angle(cov_matrix[i, j])
            phi = np.angle(est[i] * np.conj(est[j]))

            gamma += np.exp(1j * theta) * np.exp(-1j * phi)
            count += 1
    # assert count == (N * (N - 1)) / 2
    return np.abs(gamma) / count


# # TODO
# @njit(cache=True)
# def estimate_temp_coh_unbiased(est, cov_matrix):
#     # TODO
#     gamma = 0
#     N = len(est)
#     count = 0
#     for i in range(N):
#         for j in range(i + 1, N):
#             theta = np.angle(cov_matrix[i, j])
#             phi = np.angle(est[i] * np.conj(est[j]))

#             gamma += np.exp(1j * theta) * np.exp(-1j * phi)
#             count += 1
#     # assert count == (N * (N - 1)) / 2
#     return np.abs(gamma) / count


@njit(cache=True)
def getC(
    gamma_inf=0.1,
    gamma0=0.999,
    Tau0=72,
    num_acq=31,
    acq_interval=12,
    add_signal=False,
):
    time_series_length = num_acq * acq_interval + 1
    if add_signal:
        k, std_random, signal_rate = 1, 0.3, 2
    else:
        k, std_random, signal_rate = 0, 0, 0
    signal_phase, t = simulate_phase_timeseries(
        time_series_length=time_series_length,
        acquisition_interval=acq_interval,
        signal_rate=signal_rate,
        std_random=std_random,
        k=k,
    )
    # simulated_covariance_matrix = simulate_coherence_matrix(
    C = simulate_coherence_matrix(t, gamma0, gamma_inf, Tau0, signal_phase)
    return C


@njit(cache=True)
def simulate_temp_coh(C, neighbor_samples=11 * 11) -> np.ndarray:

    # simulate a complex covraince matrix based on the
    # simulated phase and coherence model
    # if C is None:
    # simulate ideal wrapped phase series without noise and without any short lived signals

    # simulate a neighborhood of SLCs with size of
    # neighbor_samples for Nt acquisitions
    neighbor_stack = simulate_neighborhood_stack(C, neighbor_samples=neighbor_samples)

    # estimate complex covariance matrix from the neighbor stack
    C_hat = compute_covariance_matrix(neighbor_stack)

    # estimate wrapped phase with a prototype estimator of EVD
    evd_estimate = estimate_evd(C_hat)

    # Return both the temp coherence using (estimated C, true C):
    return estimate_temp_coh(evd_estimate, C_hat), estimate_temp_coh(evd_estimate, C)


# def _run_cur(ns, curC, combo):
def _run_cur(trip):
    ns, curC, combo = trip
    # num_acq, Tau0, gamma_inf, gamma0, ns, nrepeat = combo
    # num_acq, Tau0, gamma_inf, gamma0, ns, idx = combo
    # num_acq, Tau0, gamma_inf, gamma0, ns, curC = combo
    num_acq, Tau0, gamma_inf, gamma0 = combo

    # return [
    return (
        num_acq,
        Tau0,
        gamma_inf,
        gamma0,
        ns,
        *simulate_temp_coh(
            curC,
            neighbor_samples=ns,
        ),
    )
    # for _ in range(nrepeat)
    # ]

def _run_cur_n(trip_n):
    return [_run_cur(trip) for trip in trip_n]


def _get_all_cov_matrices(cov_combos):
    return [
        getC(
            gamma_inf=gamma_inf,
            gamma0=gamma0,
            Tau0=Tau0,
            num_acq=num_acq,
            add_signal=False,
        )
        for (num_acq, Tau0, gamma_inf, gamma0) in cov_combos
    ]


def run_simulation(
    outname="sim.csv",
    num_acq_arr=[20],
    Tau0_arr=[1.0],
    gamma_inf_arr=[0.01],
    gamma0_arr=[0.01],
    n_samples_arr=[50],
    nrepeat=1000,
    max_workers=10,
):
    os.environ["OPENBLAS_NUM_THREADS"] = "6"

    outputs = []
    columns = "num_acq, Tau0, gamma_inf, gamma0, ns".split(", ")
    print(columns)
    cov_combos = list(itertools.product(num_acq_arr, Tau0_arr, gamma_inf_arr, gamma0_arr))
    C_arr = _get_all_cov_matrices(cov_combos)

    # num_acq_arr, Tau0_arr, gamma_inf_arr, gamma0_arr, n_samples_arr, [nrepeat]
    ns_C_pairs = list(itertools.product(
        # num_acq_arr,
        # Tau0_arr,
        # gamma_inf_arr,
        # gamma0_arr,
        # # The previous 4 were iterated in the cov_combos
        n_samples_arr,
        range(len(C_arr)),
        # C_arr,
        # np.arange(nrepeat),
    ))
    print(len(ns_C_pairs), len(cov_combos))
    ns_c_combos = [(a, C_arr[c_idx], cov_combos[c_idx]) for idx, (a, c_idx) in enumerate(ns_C_pairs)]
    # nrepeat = 50
    print(f"{len(ns_C_pairs)} ns_C_pairs, {len(cov_combos) = } {nrepeat} repeats")
    if max_workers == 1:
        # for (num_acq, Tau0, gamma_inf, gamma0, ns, _) in ns_C_pairs:
        for ridx in range(nrepeat):
            outs = []
            print(f"{ridx}/{nrepeat}")
            # outs = [_run_cur(ns, curC, combo) for (ns, curC, combo) in zip(*ns_C_pairs, cov_combos)]
            outs = [_run_cur((ns, curC, combo)) for (ns, curC, combo) in ns_c_combos]
            np.save(f"r{ridx}.npy", np.array(outs))
    else:
        repeating_trips = itertools.repeat(ns_c_combos, nrepeat)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # executor.map(_run_cur, repeating_trips)
            future_list = [executor.submit(_run_cur_n, trip) for trip in repeating_trips]
            for future in as_completed(future_list):
            # future_to_combo = {
            #     executor.submit(_run_cur, combo): combo for combo in combos
            # }
            # for future in as_completed(future_to_combo):
                # combo = future_to_combo[future]
                # print(combo)
                results = future.result()
                # outputs.extend(results)
                outputs.extend(results)

    print(np.array(outputs).shape)
    df_low_coh = pd.DataFrame(
        data=outputs, columns=columns + ["tcoh_C_hat", "tcoh_C_true"]
    )
    df_low_coh.to_csv(outname, index=False)
    return df_low_coh


import sys


def test_low_coh():
    mw = int(sys.argv[1])
    num_acq_arr = [5, 15, 30, 60, 90]
    samples_arr = [20, 50, 100, 200, 500]
    # num_acq_arr = [5, 15, 30]
    # samples_arr = [20, 50, 100]
    gamma_inf_arr = [0.00]
    gamma0_arr = [0.00]
    # Tau0_arr = [1.0]
    # return run_simulation(max_workers=mw)
    return run_simulation(
        outname="sim_low_coh.csv",
        num_acq_arr=num_acq_arr,
        gamma_inf_arr=gamma_inf_arr,
        gamma0_arr=gamma0_arr,
        n_samples_arr=samples_arr,
        max_workers=mw,
    )


def test_exp_coh():
    mw = int(sys.argv[1])
    num_acq_arr = [5, 15, 30, 60, 90]
    samples_arr = [20, 50, 100, 200]  # , 500]
    # num_acq_arr = [5, 15, 30]
    # samples_arr = [20, 50, 100]
    gamma_inf_arr = [0.0]
    gamma0_arr = [0.99, 0.5]
    Tau0_arr = [12.0, 72.0]
    # return run_simulation(max_workers=mw)
    return run_simulation(
        outname="sim_exp_coh.csv",
        num_acq_arr=num_acq_arr,
        Tau0_arr=Tau0_arr,
        gamma_inf_arr=gamma_inf_arr,
        gamma0_arr=gamma0_arr,
        n_samples_arr=samples_arr,
        max_workers=mw,
    )


def simulate_bit_mask(ny, nx, filename="neighborhood_map"):
    # flags = np.ones((ny, nx), dtype=np.bool_)
    # flag_bits = np.zeros((ny, nx), dtype=np.uint8)

    # for ii in range(ny):
    #     for jj in range(nx):
    #         flag_bits[ii, jj] = flags[ii, jj].astype(np.uint8)

    # create the weight dataset for 1 neighborhood
    # number of uint32 bytes needed to store weights
    number_of_bytes = np.ceil((ny * nx) / 32)
    n_bands = int(number_of_bytes)

    drv = gdal.GetDriverByName("ENVI")
    options = ["INTERLEAVE=BIP"]
    ds = drv.Create(filename, nx, ny, n_bands, gdal.GDT_UInt32, options)

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
    signal_phase, t = simulate_phase_timeseries(
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

    # create a VRT pointing to the stack
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

    # run fringe evd module with MLE estimato
    mle_output = os.path.join(output_timeseries_dir, "mle")
    cmd = f"evd.py -i {coreg_stack_dir}/slcs_base.vrt -w {nmap_output} -o {mle_output} -m MLE"
    os.system(cmd)

    # read the estimated wrapped phase
    # Pixel of interest is at the center of the neighborhood box
    x0 = int(nx / 2)
    y0 = int(ny / 2)
    est_wrapped_phase_fringe_evd = read_wrapped_phase(evd_output, x0, y0)

    est_wrapped_phase_fringe_mle = read_wrapped_phase(mle_output, x0, y0)

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
    print("rmse for evd [degrees]:", np.degrees(rmse_evd))
    print("rmse for evd fringe [degrees]:", rmse_fringe_evd)
    print("rmse for mle fringe [degrees]:", rmse_fringe_mle)

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

    # check the neighborhood map
    count = test_bit_mask(weight_dataset_name)
    print(f"count: {count}")
    # we have simulated an ideah homogeneous neighborhood of nx*ny.
    # Therefore the number of self-similar pixels in the neighborhood
    # should be nx*ny
    assert count == nx * ny


if __name__ == "__main__":

    # main()
    test_low_coh()
    # test_exp_coh()
