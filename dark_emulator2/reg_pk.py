import pathlib
import pickle
import torch
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from .reg_lin_pk import LinPkEmulator
from . import utils as ut

_dir = str(pathlib.Path(__file__).parent) + "/"
pk_torch_path = _dir + "data/torch/nlin_pk/"
noise_path = _dir + "data/nlin_pk/noise/"


### debug model ###
default_model = {"release": "release",
                 "type": "rand",
                 "opt": "c_ones"}
# release type choices ["release","develop"]
# model type choices ["fixed","rand"]
# fixed opt choices [None, "p9"]
# rnad opt choices ["mean", "ones", "c_ones"]


class PkEmulator():
    def __init__(self, npart=3000, boxsize=1000,
                 debug=False):

        self.npart = npart  # npart is N^(1/3)
        self.boxsize = boxsize  # unit of [Mpc/h]

        self.verbose = False

        if not debug:
            model = default_model
        else:
            assert type(debug) == dict
            if "release" not in debug.keys(): debug["release"] = "develop"
            if "type" not in debug.keys(): debug["type"] = "rand"
            if "opt" not in debug.keys(): debug["opt"] = "c_ones"
            print("### Running in debug mode ###")
            print(debug)
            model = debug

        self.model = model

        self.set_input_files(model)
        torch_file = self.input_file
        self.net_pk, self.netargs = self.load_torch(torch_file)

        self.kmin = self.netargs.krange[0]
        self.kmax = self.netargs.krange[1]
        self.nklist = self.netargs.nklist
        self.klist = np.logspace(np.log10(self.kmin), np.log10(self.kmax), self.nklist)

        self.support_nklist = 150
        self.support_klist = np.logspace(-2, 1, self.support_nklist)

        # input linear based on "cb" in non-linear pk emulator
        # This self.lin_pk_emu and DarkEmulator2.lin_pk_emu are a different space.
        self.lpe = LinPkEmulator()
        self.fid_lin_pk_cb_spl = self.lpe.fid_lin_pk_cb_spl

        if (model["type"] == "rand") and (model["opt"] == "mean"):
            self.nrand = 100
            self.init_white_noise()
            self.grf_filename = self.grf_file[str(npart)]
            self.get_pk = self.get_pk_rand
            self.gen_noise = self.gen_white_noise
        else:
            self.get_pk = self.get_pk_non_rand
            self.gen_noise = self.gen_ones_noise
            if model["opt"] == "c_ones":
                # interp_method = "cubic"
                interp_method = "linear"
                self.set_calibration_table(interp_method)

    def set_input_files(self, model):
        input_pth = {}

        release = model["release"]
        if model["type"] == "fixed":
            pth_key = model["type"] + (model["opt"] or "")
        else:
            pth_key = model["type"]

        input_pth['fixed'] = str(pathlib.Path(pk_torch_path, release, "fix_mix_pk.pth"))
        input_pth['fixedp9'] = str(pathlib.Path(pk_torch_path, "develop", "fix_mix_pk_p9.pth"))
        input_pth['fixedmid'] = str(pathlib.Path(pk_torch_path, "develop", "fix_midonly_pk.pth"))
        input_pth['rand'] = str(pathlib.Path(pk_torch_path, release, "rand_mix_pk.pth"))
        self.input_file = input_pth[pth_key]

        grf_file = {}
        grf_file["1024"] = str(pathlib.Path(noise_path, "white_power_n1024_l1024_fixed.dat"))
        grf_file["2048"] = str(pathlib.Path(noise_path, "white_power_n2048_l1024_fixed.dat"))
        grf_file["3000"] = str(pathlib.Path(noise_path, "white_power_n3000_l1000_fixed.dat"))
        self.grf_file = grf_file

        self.white_noise_range_file = str(pathlib.Path(noise_path, "white_noise_range.dat"))
        self.calibration_file = [str(pathlib.Path(noise_path, release, "calibration_mean.dat")),
                                 str(pathlib.Path(noise_path, release, "calibration_std.dat"))]

        if self.npart == 1024 and release == "develop":
            self.calibration_file = [str(pathlib.Path(noise_path, release, "calibration_mean_LR.dat")),
                                     str(pathlib.Path(noise_path, release, "calibration_std_LR.dat"))]

    def load_torch(self, model_path):
        """load_torch: Load torch script file.

        Args:
            model_path (str): Path of model.

        Returns:
            net : Network model.
            netargs : Arguments for network construction.
        """

        self.device = 'cpu'

        ef = {"attr": ''}
        net = torch.jit.load(model_path, _extra_files=ef)
        net.to(self.device)
        netargs = pickle.loads(ef["attr"])
        return net, netargs

    def set_pk_param(self, param, method="emulator"):
        if method == "emulator":
            sig8 = self.lpe.get_sigma8(param)
        else:
            sig8 = self.lpe.get_class_sigma8(param)
        param = ([param["Omega_m"], sig8, param["omega_b"],
                  param["ns"], param["Mnu"], param["Omega_k"],
                  param["w0"], param["wa"], param["h0"]])
        return param

    def set_support_lin_pk(self, param, zred, klist, method="emulator"):
        zred = np.asarray(zred)

        if method == "emulator":
            lin_pk = self.lpe.get_lin_pk(param, zred, pk_type="cb")
            k = self.lpe.klist
        elif method == "class":
            lin_pk = self.lpe.get_class_lin_pk(param, zred, pk_type="cb")
            k = self.lpe.class_klist

        lin_pk_spl = ius(k, lin_pk)
        lin_pk = lin_pk_spl(klist)
        return lin_pk

    def get_pk_non_rand(self, param, zred, method="emulator"):
        dataset = None
        for z in zred:
            self.set_dataset(param, z, method)
            _dataset = [self.dataset[ikey] for ikey in self.datakey]
            if dataset == None: dataset = _dataset
            else: dataset = [torch.cat([dataset[i], _dataset[i]]) for i in range(len(dataset))]

        dataset = [dataset[i].to(self.device) for i in range(len(dataset))]

        self.net_pk.eval()
        output = self.net_pk(dataset)
        output = output.to('cpu').detach().numpy().copy()
        pk = output
        pk = self.detach_normalization_pk(self.klist, pk, zred)

        if self.model["type"] == "rand" and self.model["opt"] == "c_ones":
            pk *= ut.call_interp2d(self.klist, zred[np.newaxis].T, self.calibration_mean)
            errs = ut.call_interp2d(self.klist, zred[np.newaxis].T, self.calibration_std)

        return pk

    def get_pk_rand(self, param, zred, method="emulator", noise_list=[]):
        # Remove the noiselist argument later.
        dataset = None
        if len(noise_list) == 0:
            noise_list = self.gen_white_noise(self.nrand)
        else:
            noise_list = torch.tensor(noise_list, dtype=torch.float32)

        for z in zred:
            self.set_dataset(param, z, method)
            # ir ==0
            fixed_model_noise = torch.tensor(self.input_grf_noise_file(self.grf_filename)[1], dtype=torch.float32)
            self.dataset["grf_del_pk"][0] = fixed_model_noise
            _dataset = [self.dataset[ikey] for ikey in self.datakey]
            if dataset == None: dataset = _dataset
            else: dataset = [torch.cat([dataset[i], _dataset[i]]) for i in range(len(dataset))]

            for ir in range(1, self.nrand):
                self.dataset["grf_del_pk"][0] = noise_list[ir]
                _dataset = [self.dataset[ikey] for ikey in self.datakey]
                if dataset == None: dataset = _dataset
                else: dataset = [torch.cat([dataset[i], _dataset[i]]) for i in range(len(dataset))]

        dataset = [dataset[i].to(self.device) for i in range(len(dataset))]

        self.net_pk.eval()
        output = self.net_pk(dataset)
        output = output.to('cpu').detach().numpy().copy()
        pk_rand = output

        pk_mean = [[] for iz in range(len(zred))]

        for iz in range(len(zred)):
            sid = iz * self.nrand
            eid = (iz + 1) * self.nrand
            pk_mean[iz] = np.mean(pk_rand[sid:eid], axis=0)

        pk_mean = np.array(pk_mean, dtype=np.float32)
        pk_mean = self.detach_normalization_pk(self.klist, pk_mean, zred)
        return pk_mean

    def set_dataset(self, param, zred, method="emulator"):
        input_type = self.netargs.input_type

        npart = self.npart / 1024.
        boxsize = self.boxsize / 1000.

        dataset = {}
        datakey = []

        datakey.append("npart")
        datakey.append("box")
        datakey.append("z")

        dataset["npart"] = [npart]
        dataset["box"] = [boxsize]
        dataset["z"] = [zred]

        dataset["param9d"] = []
        dataset["lin_pk"] = []
        dataset["grf_del_pk"] = []

        if "param9d" in input_type:
            datakey.append("param9d")
            add_param = self.set_pk_param(param, method)
            dataset["param9d"].append(add_param)

        if "linear_power" in input_type:
            datakey.append("lin_pk")
            # -> z (+ params) + lin_power
            kk = self.support_klist
            pk = self.set_support_lin_pk(param, zred, kk, method)
            pk = self.attach_normalization_pk(kk, pk, zred)
            dataset["lin_pk"].append(pk)

        if "GRF_delta" in input_type:
            datakey.append("grf_del_pk")
            grf_delta = self.gen_noise()
            dataset["grf_del_pk"].append(grf_delta)

        for ikey in datakey:
            if ikey == "grf_del_pk":
                dataset[ikey] = torch.stack(dataset[ikey])
            else:
                dataset[ikey] = torch.tensor(np.array(dataset[ikey]), dtype=torch.float32)

        self.dataset = dataset
        self.datakey = datakey

    def input_grf_noise_file(self, filename, kcol=0, pcol=1):
        power = np.loadtxt(filename, unpack=True, dtype=np.float32)
        kk = power[kcol][:self.support_nklist]
        pk = power[pcol][:self.support_nklist]
        return kk, pk

    def attach_normalization_pk(self, kk, pk, zred):
        # zred is scalar
        if self.netargs.use_normalized_fiducial_Pk:
            pk /= ut.call_interp2d(kk, zred, self.fid_lin_pk_cb_spl)
            pk = np.where(kk <= 0.1, pk, pk / ((10 * kk)**0.75))
        if self.netargs.use_log_Pk:
            pk = np.log10(pk)
        return pk

    def detach_normalization_pk(self, kk, pk, zred):
        # zred is scalar or ndarray
        if self.netargs.use_log_Pk:
            pk = 10.0**pk
        if self.netargs.use_normalized_fiducial_Pk:
            pk = np.where(kk <= 0.1, pk, pk * ((10 * kk)**0.75))
            pk *= ut.call_interp2d(kk, zred[np.newaxis].T, self.fid_lin_pk_cb_spl)
        return pk

    def init_white_noise(self):
        self.noises = np.loadtxt(self.white_noise_range_file, usecols=(1, 2))[:self.support_nklist]
        self.noises_kvec_cnt = np.loadtxt(self.white_noise_range_file, usecols=(4), dtype=np.int32)[:self.support_nklist]
        self.noise_rng = np.random.default_rng()

    def uniform_noise(self):
        noise = np.array([self.noise_rng.uniform(*self.noises[ix]) for ix in range(len(self.noises))])
        return np.array(noise, dtype=np.float32)

    def chisquare_noise(self):
        df = 2 * self.noises_kvec_cnt
        noise = self.noise_rng.chisquare(df) / df
        # noise = np.fmax(noise, self.noises[:,0])
        noise = np.fmin(noise, 1.1 * self.noises[:, 1])
        return noise

    def chisquare_noise_nrand(self, nrand):
        df = 2 * self.noises_kvec_cnt
        noise = np.array([self.noise_rng.chisquare(df) / df for ir in range(nrand)])
        # noise = np.fmax(noise, self.noises[:,0])
        noise = np.fmin(noise, 1.1 * self.noises[:, 1])
        return noise

    def gen_white_noise(self, nrand=1):
        if nrand == 1:
            # noise = self.uniform_noise()
            noise = self.chisquare_noise()
        else:
            noise = self.chisquare_noise_nrand(nrand)
        return torch.tensor(noise, dtype=torch.float32)

    def gen_ones_noise(self):
        ones = np.ones(self.support_nklist, dtype=np.float32)
        return torch.tensor(ones, dtype=torch.float32)

    def set_calibration_table(self, interp_method):
        zlist = np.logspace(np.log10(4), np.log10(1), 51) - 1
        klist = np.loadtxt(self.calibration_file[0], dtype=np.float32, unpack=True)[0]
        assert np.isclose(klist, self.klist).all()

        cmean = np.loadtxt(self.calibration_file[0], dtype=np.float32, unpack=True)[1:]
        cstd = np.loadtxt(self.calibration_file[1], dtype=np.float32, unpack=True)[1:]
        self.calibration_mean = ut.create_interp2d(klist, zlist, cmean, method=interp_method)
        self.calibration_std = ut.create_interp2d(klist, zlist, cstd, method=interp_method)

    def reduce_shotnoise_grad(self, pk, k, kny, cutoff_grad=-1.1):
        scalar_input = False
        if pk.ndim == 1:
            pk = pk[np.newaxis]
            scalar_input = True

        support_kmax = np.where(k < 100)[0][-1]  # First point to hit from the right
        support_kmin = np.where(k > 1e-3)[0][0]  # First point to hit from the left

        k_order = 1
        min_cutoff_k = 0.8 * kny
        logk = np.log10(k)
        logpk = np.log10(np.abs(pk))
        ign_k_len = (k <= min_cutoff_k).sum()
        dpk = np.gradient(logpk, logk, axis=1)  # gradient of log-log scale
        pk_extrap = []
        for idpk in range(len(dpk)):
            cutoff_logk_idx = np.where(dpk[idpk][ign_k_len:] >= cutoff_grad)[0]
            if len(cutoff_logk_idx) > 0: cutoff_idx = ign_k_len + cutoff_logk_idx[0]
            else: cutoff_idx = support_kmax

            if k[cutoff_idx] > 100: cutoff_idx = support_kmax

            #  print(cutoff_idx, k[cutoff_idx])

            pk_tmp = ius(logk[support_kmin:cutoff_idx], logpk[idpk][support_kmin:cutoff_idx], k=k_order)(logk)
            pk_extrap.append(10.0**pk_tmp)

        pk_extrap = np.array(pk_extrap)
        if scalar_input:
            pk_extrap = np.squeeze(pk_extrap)
        return pk_extrap

    def reduce_shotnoise_pshot(self, pk, k, pshot, cutoff_factor=10.0):
        scalar_input = False
        if pk.ndim == 1:
            pk = pk[np.newaxis]
            scalar_input = True

        support_kmax = np.where(k < 100)[0][-1]  # First point to hit from the right
        support_kmin = np.where(k > 1e-3)[0][0]  # First point to hit from the left

        k_order = 1
        logk = np.log10(k)
        logpk = np.log10(np.abs(pk))
        cutoff_pk = np.log10(cutoff_factor * pshot)
        pk_extrap = []
        for ilogpk in logpk:
            cutoff_idx = np.where(ilogpk < cutoff_pk)[0]  # First point to hit from the left
            if len(cutoff_idx) > 0: cutoff_idx = cutoff_idx[0]
            else: cutoff_idx = support_kmax

            if k[cutoff_idx] > 100: cutoff_idx = support_kmax

            #  print(cutoff_idx, k[cutoff_idx])

            pk_tmp = ius(logk[support_kmin:cutoff_idx], ilogpk[support_kmin:cutoff_idx], k=k_order)(logk)
            pk_extrap.append(10.0**pk_tmp)

        pk_extrap = np.array(pk_extrap)
        if scalar_input:
            pk_extrap = np.squeeze(pk_extrap)
        return pk_extrap

    def reduce_shotnoise(self, pk, k, method="pshot"):
        if method["type"] == "grad":
            return self.reduce_shotnoise_grad(pk, k, method["base"], method["factor"])
        elif method["type"] == "pshot":
            return self.reduce_shotnoise_pshot(pk, k, method["base"], method["factor"])

    def extrap_pk(self, pk, orig_k, extrap_k, cutoff_pk=0.3):
        scalar_input = False
        if pk.ndim == 1:
            pk = pk[np.newaxis]
            scalar_input = True

        pk = np.array([ius(np.log(orig_k), _pk)(np.log(extrap_k)) for _pk in pk])

        kmin, kmax = min(orig_k), max(orig_k)
        support_kmax = np.where(extrap_k < kmax)[0][-1]  # First point to hit from the right
        support_kmin = np.where(extrap_k > kmin)[0][0]  # First point to hit from the left

        k_order = 1
        logk = np.log10(extrap_k)
        logpk = np.log10(np.abs(pk))
        cutoff_pk = np.log10(cutoff_pk)
        pk_extrap = []
        for ilogpk in logpk:
            cutoff_idx = np.where(ilogpk < cutoff_pk)[0]  # First point to hit from the left
            if len(cutoff_idx) > 0: cutoff_idx = cutoff_idx[0]
            else: cutoff_idx = support_kmax

            if extrap_k[cutoff_idx] > kmax: cutoff_idx = support_kmax

            pk_tmp = ius(logk[support_kmin:cutoff_idx], ilogpk[support_kmin:cutoff_idx], k=k_order)(logk)
            pk_extrap.append(10.0**pk_tmp)

        pk_extrap = np.array(pk_extrap)
        if scalar_input:
            pk_extrap = np.squeeze(pk_extrap)
        return pk_extrap
