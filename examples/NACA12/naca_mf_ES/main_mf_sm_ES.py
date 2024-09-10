import argparse
import sys
import dill as pickle
import numpy as np
import os
import subprocess
import time
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.core.loop.user_function import MultiSourceFunctionWrapper
from scipy.stats.qmc import LatinHypercube
import emukit.test_functions
import emukit.multi_fidelity
import GPy
from functools import partial
from aero_optim.geom import get_area
from aero_optim.ffd.ffd import FFD_2D

from smt.applications.mfk import MFK
from aero_optim.utils import check_config
import json


def modif_dir(it, config, fid):
    """ 
    For the entropy search method, one point-one generation optimization will be generated for each point of the
    training set. To not overrite the files, a directory will contain all the simulations diretory. 
    The output directory must thus be changed at each new simulation.
    This function take a configuration file and modifie its output directory depending on the iteration number.
    """
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"modification de dossier sortie dans le fichier : {config}")
    
    # Load the config file
    with open(config, 'r', encoding='utf-8') as fichier:
        data = json.load(fichier)
    
    # Afficher le contenu avant modification
    # print("Contenu avant modification :")
    # print(json.dumps(data, indent=4, ensure_ascii=False))
    
    if 'study' in data:
        if 'outdir' in data['study']:
            if fid ==1:
                data['study']['outdir'] = f'output_hf_doe/output_hf_doe_{it}'
            else:
                data['study']['outdir'] = f'output_lf_doe/output_lf_doe_{it}'
        else:
            print("NO OUTDIR IN data['study']")
    else:
        print('NO STUDY IN DATA')
    
    # Afficher le contenu après modification
    # print("\nContenu après modification :")
    # print(json.dumps(data, indent=4, ensure_ascii=False))
    
    # Ouvrir le fichier en mode écriture et enregistrer les modifications
    with open(config, 'w', encoding='utf-8') as fichier:
        json.dump(data, fichier, ensure_ascii=False, indent=4)
    
    print(f"\nLes données ont été modifiées et enregistrées dans le fichier {config}.")
    
def add_custom(config):
    """ 
    Once the initial doe is computed using averaged latin hypercube, custom_doe is added into config file
    for the one by one in-fill
    """
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print(f"écriture du custom_doe dans le fichier : {config}")
    
    # Load the config file
    with open(config, 'r', encoding='utf-8') as fichier:
        data = json.load(fichier)
    
    hf_outdir = data['study']['outdir']
    
    if 'optim' in data:
        data['optim']["custom_doe"] = os.path.join(hf_outdir, "custom_doe.txt")
    else:
        print('NO STUDY IN DATA')
        
    
    # Ouvrir le fichier en mode écriture et enregistrer les modifications
    with open(config, 'w', encoding='utf-8') as fichier:
        json.dump(data, fichier, ensure_ascii=False, indent=4)
    
    print(f"\n custom_doe added to {config}.")
    
def model_eval_hf(it, args, X_f, n_design, area_margin, penalty, pen_type, pen_A, pen_Cl):
    """
    modify the output simulation directory with modif_dir function to not overrite the previous simulation. 
    Load the config dictionnary. Create the execution command. Call model_eval.
    """
    print("evalutation de la fonction haute fidelité avec Xf = ", X_f)
    
    if not isinstance(X_f, np.ndarray):
        X_f = np.array(X_f)
    
    if X_f.size != 0:
        # 1. modify the output directory in the configuration file to prevent overrwiting
        modif_dir(it, args.hf_config, 1)
        add_custom(args.hf_config)
        # setting optim to ensures the output dir is created if necessary
        hf_config, _, _ = check_config(args.hf_config, optim=True)
        hf_outdir = hf_config["study"]["outdir"]
        
        # 2. create the execution command. ( args.hf_config = path to dict file, hf_config = dict )
        exec_cmd = ["optim", "-c", f"{args.hf_config}", "-v", f"{args.verbose}", "--pymoo"]
    
        return model_eval(it, X_f, hf_outdir, exec_cmd, n_design, area_margin, penalty, pen_type, pen_A, pen_Cl)
    else :
        print("X_f = void, return 0")
        return np.empty((0,1))

def model_eval_lf(it, args, X_f, n_design, area_margin, penalty, pen_type, pen_A, pen_Cl):
    """
    modify the output simulation directory with modif_dir function to not overrite the previous simulation. 
    Load the config dictionnary. Create the execution command. Call model_eval.
    """
    
    print("evalutation de la fonction basse fidelité avec Xf = ", X_f)
    
    if not isinstance(X_f, np.ndarray):
        X_f = np.array(X_f)
        
    if X_f.size != 0:
        # 1. modify the output directory in the configuration file to prevent overrwiting
        modif_dir(it, args.lf_config, 0)
        add_custom(args.lf_config)
        # setting optim to ensures the output dir is created if necessary
        lf_config, _, _ = check_config(args.lf_config, optim=True)
        lf_outdir = lf_config["study"]["outdir"]
        
        # 2. create the execution command. ( args.hf_config = path to dict file, hf_config = dict )
        exec_cmd = ["optim", "-c", f"{args.lf_config}", "-v", f"{args.verbose}", "--pymoo"]
    
        return model_eval(it, X_f, lf_outdir, exec_cmd, n_design, area_margin, penalty, pen_type, pen_A, pen_Cl)
    else :
        print("X_f = void, return 0")
        return np.empty((0,1))
    
    
    
def model_eval(it, X_f, outdir, exec_cmd, n_design, area_margin, penalty, pen_type, pen_A, pen_Cl):
    print("creation du fichier custom avec X_f =", X_f)
    # 1. save the custom doe 
    np.savetxt(
        os.path.join(outdir, "custom_doe.txt"), X_f
    )
    print("SM: HF DOE computation..")
    
    # 2. run the simulation
    
    subprocess.run(exec_cmd, env=os.environ, stdin=subprocess.DEVNULL, check=True)

    # 3. loads results
    print(f"SM: data loading from {outdir}..")
    X_f = np.loadtxt(os.path.join(outdir, "candidates.txt"))
    Y_f = []
    with open(os.path.join(outdir, "df_dict.pkl"), "rb") as handle:
        df_dict = pickle.load(handle)
    for gid in range(len(df_dict)):
        for cid in range(len(df_dict[gid])):
            Y_f.append([df_dict[gid][cid]["CD"].iloc[-1], df_dict[gid][cid]["CL"].iloc[-1]])
    Y_f = np.array(Y_f)
    del df_dict
    
    A = collect_Area(1, outdir, n_design) # Area_list_high
    print(f"Area for iteration {it} = {A}")
    Y_fc = apply_const(Y_f, A, area_margin, penalty[1], pen_type, pen_A, pen_Cl)
    print(f"it {it}, Constrained Fonc({X_f}) = {Y_fc}")
    return Y_fc.reshape((len(Y_fc), 1))


def collect_Area(doe_size, outdir, n_design):
    A_list = np.array([])
    for i in range(doe_size):
        dat_file = os.path.join(outdir, f"FFD/naca12_g0_c{i}.dat")
        ffd = FFD_2D(dat_file, n_design // 2)
        pts = ffd.pts
        baseline = get_area(pts)
        A_list = np.append(A_list, baseline) #Y_baseline_high
    
    return A_list

def apply_const(Y, A_list, area_margin, Cl_lim, type, pen_A, pen_Cl):
    A_bl = 0.081682
    if (len(A_list) != len(Y)):
        raise Exception("Area list size does not match with objective function list size")
    else :
        if (type == 'linear'):
            Y_c = Y[:, 0]*(1 + pen_Cl*(Y[:, 1] < Cl_lim)*abs(Y[:, 1] - Cl_lim) + pen_A*(abs(A_list - A_bl)>area_margin*A_bl)*abs(abs(A_list - A_bl)-area_margin*A_bl))
        if (type == 'quadratic'):
            Y_c = Y[:, 0]*(1 + pen_Cl*(Y[:, 1] < Cl_lim)*abs(Y[:, 1] - Cl_lim)**2 + pen_A*(abs(A_list - A_bl)>area_margin*A_bl)*abs(abs(A_list - A_bl)-area_margin*A_bl)**2)
        if (type == 'semi quadratic'):
            Y_c = Y[:, 0]*(1 + pen_Cl*(Y[:, 1] < Cl_lim)*abs(Y[:, 1] - Cl_lim)**(3/2) + pen_A*(abs(A_list - A_bl)>area_margin*A_bl)*abs(abs(A_list - A_bl)-area_margin*A_bl)**(3/2))  
    return Y_c

def apply_const_Area(Y, A_list, area_margin):
    A_bl = 0.081682
    if (len(A_list) != len(Y)):
        raise Exception("Area list size does not match with objective function list size")
    else :
        if(type == 'linear'):
            Y_c = Y[:]*(1 + 300*(abs(A_list - A_bl)>area_margin*A_bl)*abs(abs(A_list - A_bl)-area_margin*A_bl)**2)
        return Y_c
    

class CustomSM:
    def __init__(self, list_of_surrogates: list[MFK]):
        self.los: list[MFK] = list_of_surrogates

    def predict(self, x: np.ndarray) -> list[float]:
        return [sm.predict_values(x) for sm in self.los]  # [Cd, Cl]
    
class mfsm_Gpy:
    def __init__(self, X_lf, X_hf, Y_lf, Y_hf, n_design):
        self.X_lf = X_lf
        self.X_hf = X_hf
        self.Y_lf = Y_lf.reshape((len(Y_lf), 1))
        self.Y_hf = Y_hf.reshape((len(Y_hf), 1))
        self.n_design = n_design
        self.model_def()
        #y_train_l = y_train_l.reshape((len(y_train_l), 1))
        
    def model_def(self):
        print("shape(self.X_lf) =", np.shape(self.X_lf) )
        print("shape(self.X_hf) =", np.shape(self.X_hf) )
        print("shape(self.Y_lf) =", np.shape(self.Y_lf) )
        print("shape(self.Y_hf) =", np.shape(self.Y_hf) )
        # print("self.Y_lf =", self.Y_lf)
        # print("self.Y_hf =", self.Y_hf)
        X_train, Y_train = convert_xy_lists_to_arrays(
            [self.X_lf, self.X_hf], [self.Y_lf, self.Y_hf])
        

        kernels = [GPy.kern.RBF(input_dim=self.n_design),
                   GPy.kern.RBF(input_dim=self.n_design)]
        
        lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(
            kernels)
        
        gpy_lin_mf_model = GPyLinearMultiFidelityModel(
            X_train, Y_train, lin_mf_kernel, n_fidelities=2)
        
        gpy_lin_mf_model.mixed_noise.Gaussian_noise.fix(0.)
        gpy_lin_mf_model.mixed_noise.Gaussian_noise_1.fix(0.)
        
        lin_mf_model = GPyMultiOutputWrapper(
            gpy_lin_mf_model, 2, n_optimization_restarts=10) #5
        
        self.model = lin_mf_model

    def model_train(self):
        self.model.optimize()
        L_param =self.model.gpy_model._param_array_
        print("L_param modele train =", L_param)
    
    def predict_values(self, x):
        self.mean, self.var = self.model.predict(x)
        return self.mean


def main():
    """
    Core script.
    
    1.a builds LF DOE i.e. single generation optimization,  [optional]
    1.b loads lf results,                                   [optional]
    1.c builds HF DOE from the best candidates              [optional]
    1.d loads hf results                                    [optional]
    1.e trains and saves surrogate,                         [optional]
    2. performs MF SM based optimization.
    """
    #%% Definition des options et parametres à l'execution du fichier
    print("1 - Definition des options et parametres à l'execution du fichier")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-clf", "--lf-config", type=str, help="/path/to/lf_config.json")
    parser.add_argument("-chf", "--hf-config", type=str, help="/path/to/hf_config.json")
    parser.add_argument("-cmfsm", "--config-mfsm", type=str, help="/path/to/config_mfsm.json") #unconstrained surrogate
    parser.add_argument("-cmfsmCS", "--config-mfsmCS", type=str, help="/path/to/config_mfsmCS.json") #constrained surrogate
    parser.add_argument("-cmfsmCSCO", "--config-mfsmCSCO", type=str, help="/path/to/config_mfsmCSCO.json") #constrained surrogate
    parser.add_argument("-l", "--load", action="store_true", help="load trained model")
    parser.add_argument("-v", "--verbose", type=int, help="logger verbosity level", default=3)
    args = parser.parse_args()

    t0 = time.time()
    
    #%% Simu lf sans custom doe => hypercube latin par defaut

    print("2 - Simu lf sans custom doe")
    if not args.load:
        # 1a. low-fidelity doe generation
        print("SM: LF DOE computation..")
        exec_cmd = ["optim", "-c", f"{args.lf_config}", "-v", f"{args.verbose}", "--pymoo"]
        subprocess.run(exec_cmd, env=os.environ, stdin=subprocess.DEVNULL, check=True)
        print(f"SM: initial DOE computation finished after {time.time() - t0} seconds")
        
        #%% Chargement des resultats de la simu lf

        # 1.b loads LF results
        lf_config, _, _ = check_config(args.lf_config)
        lf_outdir = lf_config["study"]["outdir"]
        print(f"SM: LF data loading from {lf_outdir}..")
        X_lf = np.loadtxt(os.path.join(lf_outdir, "candidates.txt"))
        Y_lf = []
        with open(os.path.join(lf_outdir, "df_dict.pkl"), "rb") as handle:
            df_dict = pickle.load(handle)
        for gid in range(len(df_dict)):
            for cid in range(len(df_dict[gid])):
                Y_lf.append([df_dict[gid][cid]["CD"].iloc[-1], df_dict[gid][cid]["CL"].iloc[-1]])
        Y_lf = np.array(Y_lf)
        del df_dict
        
        #%% recuperation des parametres pour la simulation hf
        print("3 - recuperation des parametres pour la simulation hf")

        # 1.c high-fidelity doe generation
        # setting optim to ensures the output dir is created if necessary
        hf_config, _, _ = check_config(args.hf_config, optim=True)
        hf_outdir = hf_config["study"]["outdir"]
        hf_doe_size = hf_config["optim"]["doe_size"]
        lf_doe_size = lf_config["optim"]["doe_size"]
        Opt_config, _, _ = check_config(args.config_mfsm, optim = True)
        hf_doe_size_fin = 25
        ##### ALH method
        
        n_design = hf_config["optim"]["n_design"]
        bound = hf_config["optim"]["bound"]

        # Compute bound interval size
        bound_size = float(np.diff(bound)[0])
        lower_bound = float(bound[0])
        
        print("bound_sizer =", bound_size)
        print("lower_bound =", lower_bound)
        
        #%% calcul du ALH pour la hf
        print("4 - calcul du ALH pour la hf")
        x_min = np.zeros(n_design)
        X_hf = np.empty((0, n_design))

        engine = LatinHypercube(d=n_design)
        Xlhs_hf = engine.random(n=hf_doe_size)*bound_size + lower_bound

        for k in range(hf_doe_size):
            distmin = np.inf
            for j in range(lf_doe_size):
                distk = np.linalg.norm(Xlhs_hf[k] - X_lf[j])
                if distk < distmin:
                    distmin = distk
                    x_min = X_lf[j]
            X_hf = np.append(X_hf, [x_min], axis=0)
            
        #%% sauvegarde du ALH hf dans custom_doe.txt
        print("5 - sauvegarde du ALH hf dans custom_doe.txt")
        ##### END ALH method
        np.savetxt(
            os.path.join(hf_outdir, "custom_doe.txt"), X_hf
        )
        
        #%% execution de la simulation hf avec custom_doe 
        print("6 - execution de la simulation hf avec custom_doe ")
        print("SM: HF DOE computation..")
        exec_cmd = ["optim", "-c", f"{args.hf_config}", "-v", f"{args.verbose}", "--pymoo"]
        subprocess.run(exec_cmd, env=os.environ, stdin=subprocess.DEVNULL, check=True)
        #%% Chargement des resultats de la simu hf
        print("7 - Chargement des resultats de la simu hf")

        # 1.d loads HF results
        print(f"SM: HF data loading from {hf_outdir}..")
        X_hf = np.loadtxt(os.path.join(hf_outdir, "candidates.txt"))
        Y_hf = []
        with open(os.path.join(hf_outdir, "df_dict.pkl"), "rb") as handle:
            df_dict = pickle.load(handle)
        for gid in range(len(df_dict)):
            for cid in range(len(df_dict[gid])):
                Y_hf.append([df_dict[gid][cid]["CD"].iloc[-1], df_dict[gid][cid]["CL"].iloc[-1]])
        Y_hf = np.array(Y_hf)
        del df_dict
        
        #%% 9 - entrainement du surrogate MF Gpy
        print("9 - entrainement du surrogate MF Gpy pour l Cd avec contrainte sur la func obj")
        ##### Gpy Kriging
        
        # 1.d trains and saves surrogates
        print("SM: training models..")
        # Cd
        
        

        # chargement des paramètres de penalité

        area_margin: float = Opt_config["optim"].get("area_margin", 40.) / 100.
        baseline_CL: float = Opt_config["optim"].get("baseline_CL", 0.36)
        penalty: list = Opt_config["optim"].get("penalty", ["CL", baseline_CL])
        pen_A: float = Opt_config["optim"].get("pen_A", 200.)
        pen_Cl: float = Opt_config["optim"].get("pen_Cl", 200.)
        pen_type: str = Opt_config["optim"].get("pen_type", "quadratic")
        it_max: int = Opt_config["optim"].get("nb_infill", 50)
        print("AREA_MARGIN = ", area_margin)
        print("penalty[1] =", penalty[1])
        
        # Collect the respective area of each evaluated profile of the low fidelity data-set
        A_ll = collect_Area(lf_doe_size, lf_outdir, n_design) # Area_list_low
        print("A_list for training set low fidelity =", A_ll)
        
        # Apply penalty on the objective function
        """
        il faut appliquer la penalité au surrogate dès l'initialisation par ALH
        """
        Y_lfc = apply_const(Y_lf, A_ll, area_margin, penalty[1], pen_type, pen_A, pen_Cl)
        
        # Collect the respective area of each evaluated profile of the high fidelity data-set
        A_lh = collect_Area(hf_doe_size, hf_outdir, n_design) # Area_list_high
        
        Y_hfc = apply_const(Y_hf, A_lh, area_margin, penalty[1], pen_type, pen_A, pen_Cl)
        
        
        
        print("fonction objectif non pénalisée évaluée au set d'entrainement haute fidelité =", Y_hf)
        print("fonction objectif pénalisée évaluée au set d'entrainement haute fidelité =", Y_hfc)
        
        print("fonction objectif non pénalisée évaluée au set d'entrainement basse fidelité =", Y_lf)
        print("fonction objectif pénalisée évaluée au set d'entrainement basse fidelité =", Y_lfc)
        
        # Mise en forme du set d'entrainement pour initialisation de l'entropy search
        
        X_train, Y_train = convert_xy_lists_to_arrays(
            [X_lf, X_hf], [Y_lfc.reshape((len(Y_lfc), 1)), Y_hfc.reshape((len(Y_hfc), 1))])
        
        mfsmGpy_cd_const = mfsm_Gpy(X_lf, X_hf, Y_lfc, Y_hfc, n_design)
        mfsmGpy_cd_const.model_def()
        mfsmGpy_cd_const.model_train()
        
        #%% 10 - Entropy Search : Definition de l'espace des parametres
        print("10 - Entropy Search : Definition de l'espace des parametres")
        ########################################### ENTROPY SEARCH
        
        np.random.seed(20)


        # Bayesian optimisation

        # Define parameter space

        from emukit.core import ParameterSpace, ContinuousParameter, InformationSourceParameter
        n_fidelities = 2
        # Définition des paramètres continus pour 30 dimensions
        continuous_parameters = [ContinuousParameter(f'x{i}', lower_bound, lower_bound + bound_size) for i in range(1, n_design + 1)]

        # Définition du paramètre de source d'information
        information_source_parameter = InformationSourceParameter(n_fidelities)

        # Création de l'espace des paramètres en combinant les paramètres continus et le paramètre de source d'information
        parameter_space = ParameterSpace(continuous_parameters + [information_source_parameter])


        #%% 11 - Definition de la fonction multifidelité contient : redefinition de l'espace des parametres + appel des fonctions haute et basse fid
        print("11 - Definition de la fonction multifidelité")
        def MF_Cd_func(it, args,n_design, bound, lower_bound, area_margin, penalty, pen_type, pen_A, pen_Cl, high_fidelity_noise_std_deviation=0, low_fidelity_noise_std_deviation=0):
            """
            Two-level multi-fidelity polynomial function
            """
            
            n_fidelities = 2

            # Définition des paramètres continus pour 30 dimensions
            continuous_parameters = [ContinuousParameter(f'x{i}', lower_bound, lower_bound + bound_size) for i in range(1, n_design + 1)]

            # Définition du paramètre de source d'information
            information_source_parameter = InformationSourceParameter(n_fidelities)

            # Création de l'espace des paramètres en combinant les paramètres continus et le paramètre de source d'information
            parameter_space = ParameterSpace(continuous_parameters + [information_source_parameter])
            
            user_function = MultiSourceFunctionWrapper(
                [
                    lambda x: model_eval_lf(it, args, x, n_design, area_margin, penalty, pen_type, pen_A, pen_Cl),
                    lambda x: model_eval_hf(it, args, x, n_design, area_margin, penalty, pen_type, pen_A, pen_Cl),
                ]
            )
            return user_function, parameter_space

        # MF_Cd, _ = MF_Cd_func()
       

        #%% 12 - Definition de la fonction d'acquisition
        
        print("12 - Definition de la fonction d'acquisition")
        
        # Model definition
        
        model = mfsmGpy_cd_const.model

        # Define acquisition function
        
        # Assign costs
        low_fidelity_cost = 1
        high_fidelity_cost = 3

        from emukit.bayesian_optimization.acquisitions.entropy_search import MultiInformationSourceEntropySearch
        from emukit.core.acquisition import Acquisition

        # Define cost of different fidelities as acquisition function
        class Cost(Acquisition):
            def __init__(self, costs):
                self.costs = costs

            def evaluate(self, x):
                fidelity_index = x[:, -1].astype(int)
                x_cost = np.array([self.costs[i] for i in fidelity_index])
                return x_cost[:, None]
            
            @property
            def has_gradients(self):
                return True
            
            def evaluate_with_gradients(self, x):
                return self.evalute(x), np.zeros(x.shape)

        cost_acquisition = Cost([low_fidelity_cost, high_fidelity_cost])
        acquisition = MultiInformationSourceEntropySearch(model, parameter_space) / cost_acquisition
        
        #%% 13 - definition de la boucle d'optimisation 
        print("13 - definition de la boucle d'optimisation")
        # Create outer loop

        from emukit.core.loop import FixedIntervalUpdater, OuterLoop, SequentialPointCalculator
        from emukit.core.loop.loop_state import create_loop_state
        from emukit.core.optimization.multi_source_acquisition_optimizer import MultiSourceAcquisitionOptimizer
        from emukit.core.optimization import GradientAcquisitionOptimizer

        initial_loop_state = create_loop_state(X_train, Y_train)
        acquisition_optimizer = MultiSourceAcquisitionOptimizer(GradientAcquisitionOptimizer(parameter_space), parameter_space)
        candidate_point_calculator = SequentialPointCalculator(acquisition, acquisition_optimizer)
        model_updater = FixedIntervalUpdater(model)
        loop = OuterLoop(candidate_point_calculator, model_updater, initial_loop_state)

        # add plotting

        def plot_acquisition(loop, loop_state):
            print("iteration ", loop_state.iteration)
            
        loop.iteration_end_event.append(plot_acquisition)
        print("test 1")

            
        def evolution_x(loop, loop_state):
            print("loop.loop_state.X =", loop.loop_state.X)
            is_high_fidelity = loop.loop_state.X[:, -1] == 1
            x_low=loop.loop_state.X[~is_high_fidelity, 0]
            x_high=loop.loop_state.X[is_high_fidelity, 0]
            print("shape low fid =",np.shape(x_low))
            print("shape high fid =", np.shape(x_high))
            
            
        print("test 2")

        # subscribe to event
        loop.iteration_end_event.append(evolution_x)
        
        #%% 14 - lancement de la boucle d'optimisation
        print("14 - lancement de la boucle d'optimisation")
        # Run optimization
        List_l_fid = []
        List_h_fid = []
        MF_Cd, _ = MF_Cd_func(0, args, n_design, bound, lower_bound, area_margin, penalty, pen_type, pen_A, pen_Cl)
        it = 0
        loop.run_loop(MF_Cd, 1)
        is_high_fidelity = loop.loop_state.X[:, -1] == 1
        
        # indent the list of fidelity function call
        if len(loop.loop_state.X[is_high_fidelity, 0]) - hf_doe_size > len(List_h_fid): 
            List_h_fid.append(it)
        else :
            List_l_fid.append(it)
            
        while(len(loop.loop_state.X[is_high_fidelity, 0])<hf_doe_size_fin and it < it_max):
            print("size high fid training-set = ", len(loop.loop_state.X[is_high_fidelity, 0]) )
            it = loop.loop_state.iteration
            print(f"loop state iteration = {it}")
            MF_Cd, _ = MF_Cd_func(it, args, n_design, bound, lower_bound, area_margin, penalty, pen_type, pen_A, pen_Cl)
            #MF_Cd_it = partial(MF_Cd, it, args)
            loop.run_loop(MF_Cd, 1)
            is_high_fidelity = loop.loop_state.X[:, -1] == 1
            
            if len(loop.loop_state.X[is_high_fidelity, 0]) - hf_doe_size > len(List_h_fid): 
                List_h_fid.append(it)
            else :
                List_l_fid.append(it)

        print("size high fid training-set = ", len(loop.loop_state.X[is_high_fidelity, 0]) )
        print("Optimisation finished")
        
        print("loop.loop_state.X[:, -1] == 1 =", loop.loop_state.X[:, -1] == 1)
        print("loop.loop_state.X[:, -1] == 0 =", loop.loop_state.X[:, -1] == 0)
        
        ########################################### END ENTROPY SEARCH
        
        #%% 15 - Chargement des Cd et Cl associés au training set
        print("15 - Chargement des Cd et Cl associés au training set haute fidelité")
        
        hf_doe_size_fin = len(loop.loop_state.X[is_high_fidelity, 0]) # Réajustement de la taille du set d'entrainement haute fidelité 
                                                                        # dans le cas ou l'optim n'est pas allée jusqu'à l'objectif initial
        
        for i in List_h_fid:
            hf_outdir_i = f'output_hf_doe/output_hf_doe_{i}'
            # loads HF results
            X_i = np.loadtxt(os.path.join(hf_outdir_i, "candidates.txt"))
            
            print(f"X_{i} =", X_i)
            with open(os.path.join(hf_outdir_i, "df_dict.pkl"), "rb") as handle:
                df_dict = pickle.load(handle)
            for gid in range(len(df_dict)):
                for cid in range(len(df_dict[gid])):
                    Y_hf= np.append(Y_hf, [[df_dict[gid][cid]["CD"].iloc[-1], df_dict[gid][cid]["CL"].iloc[-1]]], axis = 0)
                    X_hf = np.append(X_hf, [X_i], axis = 0)
            del df_dict
            
        print("15 - Chargement des Cd et Cl associés au training set basse fidelité")
        
        is_low_fidelity = loop.loop_state.X[:, -1] == 0
        lf_doe_size_fin = len(loop.loop_state.X[is_low_fidelity, 0])
        
        List_l_fid2 = list(int(i) for i in is_low_fidelity[hf_doe_size + lf_doe_size:])
        List_h_fid2 = list(int(i) for i in is_high_fidelity[hf_doe_size + lf_doe_size:])
        
        print(f"List_l_fid = {List_l_fid}, List_l_fid2 = {List_l_fid2}")
        print(f"List_h_fid = {List_h_fid}, List_h_fid2 = {List_h_fid2}")
        
        
        for i in List_l_fid:
            lf_outdir_i = f'output_lf_doe/output_lf_doe_{i}'
            # loads HF results
            X_i = np.loadtxt(os.path.join(lf_outdir_i, "candidates.txt"))
            
            print(f"X_{i} =", X_i)
            with open(os.path.join(lf_outdir_i, "df_dict.pkl"), "rb") as handle:
                df_dict = pickle.load(handle)
            for gid in range(len(df_dict)):
                for cid in range(len(df_dict[gid])):
                    Y_lf= np.append(Y_lf, [[df_dict[gid][cid]["CD"].iloc[-1], df_dict[gid][cid]["CL"].iloc[-1]]], axis = 0)
                    X_lf = np.append(X_lf, [X_i], axis = 0)
            del df_dict
            
        print("fonction objectif non pénalisée évaluée au set d'entrainement haute fidelité =", Y_hf)
        print("set d'entrainement haute fidelité = ", X_hf)
        print("fonction objectif non pénalisée évaluée au set d'entrainement basse fidelité =", Y_lf)
        print("set d'entrainement basse fidelité = ", X_lf)
        
        #%% 16 Entrainement des surrogates pour l'optimisation$
        print("16 - Entrainement des surrogates pour l'optimisation")
        
        print("SM: training models..")
        # Cd
        
        mfsmGpy_cl = mfsm_Gpy(X_lf, X_hf, Y_lf[:, 1], Y_hf[:, 1], n_design)
        mfsmGpy_cl.model_def()
        mfsmGpy_cl.model_train()
        
        # Cl
        
        mfsmGpy_cd = mfsm_Gpy(X_lf, X_hf, Y_lf[:, 0], Y_hf[:, 0], n_design)
        mfsmGpy_cd.model_def()
        mfsmGpy_cd.model_train()


        #%% 17 - Sauvegarde des surrogates
        print("17 - Sauvegarde des surrogates")
        
        # saves combined sm
        customGpy_mf_sm = CustomSM([mfsmGpy_cd, mfsmGpy_cl])
        customGpy_mf_sm_const = CustomSM([mfsmGpy_cd_const, mfsmGpy_cl])
        
        #save unconstrained surrogate model for penalty surrogate model optimisation
        with open(os.path.join(hf_outdir, "modelGpy_CO.pkl"), "wb") as handle:
            pickle.dump(customGpy_mf_sm, handle)
        print(f"SM: unconstrained model saved to {hf_outdir}")
        
        #save constrained surrogate model for free surrogate model optimisation
        with open(os.path.join(hf_outdir, "modelGpy_CS.pkl"), "wb") as handle:
            pickle.dump(customGpy_mf_sm_const, handle)
        print(f"SM: constrained model saved to {hf_outdir}")
        
        ##### End Gpy Kriging
        
    #%% Optimisation basée sur le surrogate

    # 2. SM based constrained optimization on unconstrained surrogate
    print("SM: surrogate model based optimization (constraint on optimisation)..")
    exec_cmd = ["optim", "-c", f"{args.config_mfsm}", "-v", f"{args.verbose}", "--pymoo"]
    subprocess.run(exec_cmd, env=os.environ, stdin=subprocess.DEVNULL, check=True)
    print(f"SM: surrogate model based optimization (constraint on optimisation) finished after {time.time() - t0} seconds")
    
    # 2.b SM based unconstrained optimization on constrained surrogate
    print("SM: surrogate model based optimization (constraint on surrogate)..")
    exec_cmd = ["optim", "-c", f"{args.config_mfsmCS}", "-v", f"{args.verbose}", "--pymoo"]
    subprocess.run(exec_cmd, env=os.environ, stdin=subprocess.DEVNULL, check=True)
    print(f"SM: surrogate model based optimization (constraint on surrogate) finished after {time.time() - t0} seconds")

    # 2.b SM based unconstrained optimization on constrained surrogate
    print("SM: surrogate model based optimization (constraint on surrogate and optimisation)..")
    exec_cmd = ["optim", "-c", f"{args.config_mfsmCSCO}", "-v", f"{args.verbose}", "--pymoo"]
    subprocess.run(exec_cmd, env=os.environ, stdin=subprocess.DEVNULL, check=True)
    print(f"SM: surrogate model based optimization (constraint on surrogate and optimisation) finished after {time.time() - t0} seconds")

#%% execution main

if __name__ == '__main__':
    #sys.argv = ["mon_script.py", "--name", "Alice", "--age", "30"]
    main()
