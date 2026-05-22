import sys
import torch
import pickle
import pathlib
import numpy as np

from . import class_lin_power as clp
from . import utils as ut

_dir = str(pathlib.Path(__file__).parent) + "/"

def_amp_torch = _dir + "data/torch/amp/amp.pth"
def_lin_pk_cb_torch = _dir + "data/torch/lin_pk/lin_pk_cb.pth"
def_lin_ratio_torch = _dir + "data/torch/lin_pk/lin_ratio.pth"

fid_lin_pk_cb_file = _dir + "data/lin_pk/fiducial/power_cb.dat"
fid_lin_pk_tot_file = _dir + "data/lin_pk/fiducial/power_tot.dat"


class LinPkEmulator():
    def __init__(self):
        """__init__: initializer

        Args:
            lin_pk_torch (str, optional): Path of lin_pk torch. Defaults to lin_pk_torch.
            amp_torch (str, optional): Path of amp torch. Defaults to amp_torch.
        """

        self.verbose = False

        # Change to "cubic" when this bug is fixed.
        # <https://github.com/scipy/scipy/issues/18010>
        # interp_method = "cubic"
        interp_method = "linear"

        # for amplitude
        self.net_amp, self.param_amp = self.load_torch(def_amp_torch)
        self.norm_sig8 = 0.83100
        self.norm_lnAs = 3.094
        self.norm_dist = 10.0

        # for cb linear power
        self.net_lin_pk_cb, self.param_lin_pk_cb = self.load_torch(def_lin_pk_cb_torch)
        klist = np.loadtxt(fid_lin_pk_cb_file, unpack=True, dtype=np.float32)[0]
        self.klist = klist[::self.param_lin_pk_cb.skip_klist]

        # for power ratio
        self.net_lin_ratio, self.param_lin_ratio = self.load_torch(def_lin_ratio_torch)
        klist = np.loadtxt(fid_lin_pk_tot_file, unpack=True, dtype=np.float32)[0]
        klist = klist[::self.param_lin_ratio.skip_klist]

        assert np.all(np.isclose(self.klist, klist))

        with open(fid_lin_pk_cb_file) as f:
            zline = f.readline().rstrip().split()
            zline = [float(zline[iz]) for iz in range(len(zline)) if ut.is_num(zline[iz])]
        zlist = np.array(zline, dtype=np.float32)
        fid_lin_pk = np.loadtxt(fid_lin_pk_cb_file, unpack=True, dtype=np.float32)
        self.fid_lin_pk_cb_spl = ut.create_interp2d(fid_lin_pk[0], zlist, fid_lin_pk[1:], method=interp_method)

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

    def get_sigma8(self, param):
        if "sigma8" in param.keys() and param["sigma8"] is not None:
            return param["sigma8"]
        elif "ln(10^10As)" in param.keys() and param["ln(10^10As)"] is not None:
            amp_flag = 0
            input_param = ([amp_flag,
                            param["Omega_m"], param["omega_b"], param["ln(10^10As)"],
                            param["ns"], param["Mnu"], param["Omega_k"],
                            param["w0"], param["wa"], param["h0"]])
        elif "As" in param.keys() and param["As"] is not None:
            amp_flag = 0
            lnAs = np.log(1.0e+10 * param["As"])
            input_param = ([amp_flag,
                            param["Omega_m"], param["omega_b"], lnAs,
                            param["ns"], param["Mnu"], param["Omega_k"],
                            param["w0"], param["wa"], param["h0"]])
        else:
            print("Undefined amplitude parameter key ; sigma8 or As or ln(10^10As)")
            sys.exit(1)

        input_param = torch.tensor(input_param, dtype=torch.float32)
        input_param = input_param.to(self.device)

        self.net_amp.eval()
        output = self.net_amp(input_param)
        output = output.to('cpu').detach().numpy().copy()
        output = output.astype(np.float32)
        sig8 = output[0] * self.norm_sig8  # [sig8, lnAs, dist]
        return sig8

    def get_lnAs(self, param):
        """get_lnAs: ln(10^10 As)
        """
        amp_flag = 1
        input_param = ([amp_flag,
                        param["Omega_m"], param["omega_b"], param["sigma8"],
                        param["ns"], param["Mnu"], param["Omega_k"],
                        param["w0"], param["wa"], param["h0"]])

        input_param = torch.tensor(input_param, dtype=torch.float32)
        input_param = input_param.to(self.device)

        self.net_amp.eval()
        output = self.net_amp(input_param)
        output = output.to('cpu').detach().numpy().copy()
        output = output.astype(np.float32)
        lnAs = output[1] * self.norm_lnAs  # [sig8, lnAs, dist]
        return lnAs

    def get_As(self, param):
        return np.exp(self.get_lnAs(param)) * 1e-10

    def set_lin_pk_param_from_file(self, param_file, use_amp="sigma8"):
        # ID Omega_m sigma8 omega_b omega_c Omega_de ln(10^10As) ns Mnu Omega_k w0 wa h0 S8 dist
        param = np.loadtxt(param_file, dtype=np.float32)
        sig8 = param[2]
        As = param[6]
        omb = param[3]

        if use_amp == "sigma8":
            idx = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0])
            param = param[idx == 1]
            param[1] = omb
            param[2] = sig8
            amp_flag = 1
        else:
            idx = np.array([0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0])
            param = param[idx == 1]
            param[1] = omb
            param[2] = As
            amp_flag = 0

        param = np.concatenate([[amp_flag], param])
        param = torch.tensor(param, dtype=torch.float32)

        if use_amp == "sigma8":
            return param

        param = param.to(self.device)

        self.net_amp.eval()
        output = self.net_amp(param)
        output = output.to('cpu').detach().numpy().copy()
        output = output.astype(np.float32)
        sig8 = output[0] * self.norm_sig8

        param[0] = 1  # amp flag
        param[3] = sig8  # As to sig8
        return param

    def set_lin_pk_param(self, param):
        sig8 = self.get_sigma8(param)
        amp_flag = 1
        lin_pk_param = ([amp_flag,
                        param["Omega_m"], param["omega_b"], sig8,
                        param["ns"], param["Mnu"], param["Omega_k"],
                        param["w0"], param["wa"], param["h0"]])
        return torch.tensor(lin_pk_param, dtype=torch.float32)

    def get_lin_tot_cb_ratio(self, param, zred=0.0):
        return self.get_lin_ratio(param, zred)

    def get_lin_pk(self, param, zred=0.0, pk_type="cb"):
        lin_pk = self.get_lin_pk_cb(param, zred)
        if pk_type == "total":
            # plin_tot = plin_cb * (plin_tot/plin_cb)
            lin_pk *= self.get_lin_tot_cb_ratio(param, zred)
        return lin_pk

    def get_lin_pk_cb(self, param, zred=0.0):
        param = self.set_lin_pk_param(param)
        # param = torch.cat((torch.tensor([zred], dtype=torch.float32), param))
        param = torch.cat([torch.tensor(zred, dtype=torch.float32).view(-1, 1),
                           param.tile(zred.size, 1)], dim=1)
        param = param.to(self.device)

        self.net_lin_pk_cb.eval()
        output = self.net_lin_pk_cb(param)
        output = output.to('cpu').detach().numpy().copy()
        lin_pk = output
        lin_pk = self.detach_normalization_pk_cb(lin_pk, zred)
        return lin_pk

    def get_lin_ratio(self, param, zred=0.0):
        param = self.set_lin_pk_param(param)
        # param = torch.cat((torch.tensor([zred], dtype=torch.float32), param))
        param = torch.cat([torch.tensor(zred, dtype=torch.float32).view(-1, 1),
                           param.tile(zred.size, 1)], dim=1)
        param = param.to(self.device)

        self.net_lin_ratio.eval()
        output = self.net_lin_ratio(param)
        output = output.to('cpu').detach().numpy().copy()
        lin_ratio = output
        return lin_ratio

    def detach_normalization_pk_cb(self, pk, zred):
        pk = 10.0**pk
        pk *= ut.call_interp2d(self.klist, zred[np.newaxis].T, self.fid_lin_pk_cb_spl)
        return pk

    def class_lin_pk_param(self, param):
        """class_class_param: Set parameter for class linear power

        Args:
            param (dict): 9D cosmological parameters

        Returns:
            param (array): cosmological parameters for class
        """

        # amplitude parameter
        if "sigma8" in param.keys():
            # temporary value
            param['ln(10^10As)'] = 3.094
            param['As'] = np.exp(param['ln(10^10As)']) * 1e-10
        else:
            param['sigma8'] = None
            if "ln(10^10As)" in param.keys():
                param['As'] = np.exp(param['ln(10^10As)']) * 1e-10
            else:
                param['ln(10^10As)'] = np.log(param['As'] * 1e10)
        # fixed values
        param['k_pivot'] = 0.05
        param['cs2_fld'] = 1.0
        return param

    ##########################
    ### for Boltzmann code ###
    ##########################

    def get_class_lin_pk(self, param, zred=0.0, pk_type="cb"):
        # No spline interpolation for k.
        if pk_type == "cb":
            lin_pk = self.get_class_lin_pk_cb(param, zred)
        elif pk_type == "total":
            lin_pk = self.get_class_lin_pk_total(param, zred)
        return lin_pk

    def get_class_lin_pk_cb(self, param, zred=0.0):
        # No spline interpolation for k.
        param = self.class_lin_pk_param(param)
        lin_pk_cb, _, param = clp.calc_class_lin_pk(param, zred)
        self.class_klist = clp.klist
        self.class_sigma8 = param["sigma8"]
        lin_pk = lin_pk_cb
        return lin_pk

    def get_class_lin_pk_total(self, param, zred=0.0):
        # No spline interpolation for k.
        param = self.class_lin_pk_param(param)
        _, lin_pk_total, param = clp.calc_class_lin_pk(param, zred)
        self.class_klist = clp.klist
        self.class_sigma8 = param["sigma8"]
        lin_pk = lin_pk_total
        return lin_pk

    def get_class_sigma8(self, param):
        param = self.class_lin_pk_param(param)
        sig8 = param["sigma8"]
        if sig8 == None:
            sig8 = clp.get_sigma8(param)
        self.class_sigma8 = sig8
        return sig8

    def ratio_of_cb_to_total_lin_pk(self, param, zred=0.0):
        # total/cb
        return self.get_lin_tot_cb_ratio(param, zred)

    def ratio_of_cb_to_total_class_lin_pk(self, param, zred=0.0):
        # total/cb
        pk_cb = self.get_class_lin_pk(param, zred, pk_type="cb")
        pk_total = self.get_class_lin_pk(param, zred, pk_type="total")
        return pk_total / pk_cb

    def get_ratio_of_cb_to_total(self, param, zred=None, method="emulator"):
        # lin_pk_total / lin_pk_cb
        if method == "emulator":
            lin_pk_ratio = self.ratio_of_cb_to_total_lin_pk(param, zred)
            k = self.klist
        elif method == "class":
            lin_pk_ratio = self.ratio_of_cb_to_total_class_lin_pk(param, zred)
            k = self.class_klist
        return k, lin_pk_ratio

    def calc_power_distance(self, k, pk):
        z = 0.0
        fid_pk = ut.call_interp2d(k, z, self.fid_lin_pk_cb_spl)
        denominator = np.fmin(fid_pk, pk)
        rela_err_dist = np.abs(fid_pk - pk) / np.abs(denominator)
        power_dist = sum(rela_err_dist) / len(k)
        return power_dist
