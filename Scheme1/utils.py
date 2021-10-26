from envs.envs import GaussianLinearEnvironment, LoanEnvironment
from servers.LDP_servers import NPUCBserver, LDPUCBServer, GOLSServer, GreedySGDServer, LDPGLMServer, RSGDServer, ROLSServer, RLUCBServer, RGLMServer
from mechanisms.LDP import NMechanism, OLSMechanism, SGDLDPMechanism, GLMMechanism
from mechanisms.RLDP import RSGDLDPMechanism, ROLSMechanism, RGLMMechanism
from profilers.profilers import Profiler
from workers.workers import WorkerA, WorkerB

settings_dict = dict()

settings_dict['env'] = dict()
settings_dict['env']['GaussianLinear'] = GaussianLinearEnvironment
settings_dict['env']['OnlineLoan'] = LoanEnvironment

settings_dict['mechanism'] = dict()
settings_dict['mechanism']['N'] = NMechanism
settings_dict['mechanism']['OLS'] = OLSMechanism
settings_dict['mechanism']['ROLS'] = ROLSMechanism
settings_dict['mechanism']['SGD'] = SGDLDPMechanism
settings_dict['mechanism']['RSGD'] = RSGDLDPMechanism
settings_dict['mechanism']['GLM'] = GLMMechanism
settings_dict['mechanism']['RGLM'] = RGLMMechanism

settings_dict['server'] = dict()
settings_dict['server']['NP'] = NPUCBserver
settings_dict['server']['LUCB'] = LDPUCBServer
settings_dict['server']['GOLS'] = GOLSServer
settings_dict['server']['GSGD'] = GreedySGDServer
settings_dict['server']['RLUCB'] = RLUCBServer
settings_dict['server']['RSGD'] = RSGDServer
settings_dict['server']['ROLS'] = ROLSServer
settings_dict['server']['RGLM'] = RGLMServer
settings_dict['server']['LUCBG'] = LDPGLMServer

settings_dict['profiler'] = dict()
settings_dict['profiler']['Profiler'] = Profiler

settings_dict['worker'] = dict()
settings_dict['worker']['A'] = WorkerA
settings_dict['worker']['B'] = WorkerB


def wrap_params(name, 
                worker = 'A', 
                random_seed = 0, 
                data_id = 'eye_movements', 
                env = 'GaussianLinear', 
                reward = 'normal', 
                mechanism = 'NP', 
                server = 'LDPServer', 
                agent = 'LDPUCB', 
                profiler = 'Profiler', 
                num_batch = int(1e5), 
                dimension = 2, 
                n_action = 5000, 
                T = int(1e5), 
                optimal_gap = 0.1, 
                eps = 1e4, 
                context_type = 'gapless', 
                echo = False, 
                echo_freq = 100, 
                batch = False):

    params = dict()

    # global setting
    params['name'] = name 
    params['worker'] = worker
    params['random_seed'] = random_seed
    params['echo'] = echo
    params['echo_freq'] = echo_freq
    params['env_'] = env
    params['mechanism_'] = mechanism
    params['server_'] = server
    params['agent_'] = agent
    params['profiler_'] = profiler
    params['num_batch'] = num_batch
    params['fail_prob'] = 0.1

    # environment instances
    params['env'] = dict()
    params['env']['data_id'] = data_id
    params['env']['instance_variance'] = 1
    params['env']['dimension'] = dimension
    params['env']['n_action'] = n_action
    params['env']['time_horizon'] = int(T)

    params['env']['reward_variance'] = 1
    params['env']['optimal_value'] = 0.75
    params['env']['optimal_gap'] = optimal_gap
    params['env']['context_type'] = context_type
    params['env']['yx'] = reward
    
    # server
    params['server'] = dict()
    params['server']['batch'] = batch

    # Mechanism params
    params['mechanism'] = dict()
    params['mechanism']['epsilon'] = eps
    params['mechanism']['delta'] = 0.01

    return params

def get_settings(params):
    Env = settings_dict['env'][params['env_']]
    Mechanism = settings_dict['mechanism'][params['mechanism_']]
    Server = settings_dict['server'][params['server_']]
    Profiler = settings_dict['profiler'][params['profiler_']] 
    env = Env(params)
    mechanism = Mechanism(params)
    server = Server(params)
    profiler = Profiler(params)
    return env, server, mechanism, profiler

