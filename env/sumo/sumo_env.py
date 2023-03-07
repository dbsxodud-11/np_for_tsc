import random
from copy import deepcopy
from os.path import join as pjoin

import torch
import torch.multiprocessing as mp
from torch.quasirandom import SobolEngine
from tqdm import tqdm

import traci
from sumolib import checkBinary
from sumolib.miscutils import getFreeSocketPort

class SumoEnv():
    def __init__(self, task, scenario, run_type='train', visualize=False):
        # SUMO Configuration Setting
        task, subtask = task.split("_")

        if visualize:
            sumoBinary = checkBinary('sumo-gui')
        else:
            sumoBinary = checkBinary('sumo')
        sumocfg_path = pjoin("env", task, subtask, f"{run_type}_{scenario}.sumocfg")
        init_tllogic_path = pjoin("env", task, subtask, f"{subtask}.tll.xml")

        self.sumoCmd = [sumoBinary, "-c", sumocfg_path]
        self.sumoCmd += ["-a", init_tllogic_path]
        self.sumoCmd += ['--time-to-teleport', '600'] # long teleport for safety
        self.sumoCmd += ['--no-warnings', 'True']
        self.sumoCmd += ['--duration-log.disable', 'True']
        self.sumoCmd += ['--no-step-log', 'True']

        self.port = getFreeSocketPort()

        self.time = None
        self.decision_time = 1800
        self.eval_time = 60
        self.yellow_time = 3

        self.min_time = 30
        self.cycle_time = 180

        self.input_dim = 4
        if subtask == "3by3":
            self.n_intersections = 9
        elif subtask == "4by4":
            self.n_intersections = 16
        elif subtask == "5by5":
            self.n_intersections = 25
        elif subtask == "hangzhou":
            self.n_intersections = 16
        elif subtask == "manhattan":
            self.n_intersections = 48

    def apply_action(self, x):
        traci.start(self.sumoCmd, port=self.port)

        for i, intersection_ID in enumerate(traci.trafficlight.getIDList()):
            current_logic = traci.trafficlight.getAllProgramLogics(intersection_ID)[1]
            next_logic = deepcopy(current_logic)
            
            current_phases = current_logic.getPhases()
            next_phases = []
            for j, phase in enumerate(current_phases):
                if j % 2 == 0:
                    phase.duration = int(x[i][j//2].item() * (self.cycle_time - (self.min_time + self.yellow_time) * self.input_dim) + self.min_time)
                    phase.minDur = phase.duration
                    phase.maxDur = phase.duration
                next_phases.append(phase)
            next_phases = tuple(next_phases)

            next_logic.phases = next_phases
            traci.trafficlight.setProgramLogic(intersection_ID, next_logic)

        y = torch.zeros(self.n_intersections, self.decision_time // self.eval_time).to(x.device)
        for t in range(self.decision_time):
            traci.simulationStep()
            if (t+1) % self.eval_time == 0:
                for i, intersection_ID in enumerate(traci.trafficlight.getIDList()) :
                    for lane in traci.trafficlight.getControlledLanes(intersection_ID):
                        y[i][t // self.eval_time] += traci.lane.getLastStepHaltingNumber(lane)

        traci.close()
        return y.mean(dim=-1)

    def get_constraints(self, dtype, device):
        equality_constraints = []
        total_input_dim = self.input_dim * self.n_intersections

        prev_input_dim = 0
        for _ in range(self.n_intersections):
            indices = torch.arange(prev_input_dim, prev_input_dim+self.input_dim).to(device)
            coefficients = torch.ones(self.input_dim).to(device)
            equality_constraints.append((indices, coefficients, 1.0))
            prev_input_dim += self.input_dim

        bounds = torch.stack([torch.zeros(total_input_dim), torch.ones(total_input_dim)], dim=0).to(dtype=dtype, device=device)
        return total_input_dim, bounds, equality_constraints

    def get_init_points(self, init_num_points, dim, seed=42):
        sobol = SobolEngine(dim, scramble=True, seed=seed)
        x = sobol.draw(init_num_points)
        return x

    def input_transform(self, x):
        return x.view(-1, self.n_intersections, self.input_dim).softmax(dim=-1)

    def evaluate(self, xs, aggregate=True):
        xs = self.input_transform(xs)
        if aggregate:
            ys = [self.apply_action(x).sum(dim=-1, keepdim=True) for x in xs]
        else:
            ys = []
            for i in tqdm(range(len(xs))):
                ys.append(self.apply_action(xs[i]))

        if len(xs) == 1:
            xs = xs.squeeze(0).tolist()
        ys = torch.stack(ys)
        return xs, ys


def env_evaluation(config):
    env, xs = config
    ys = []
    for i in range(len(xs)):
        ys.append(env.apply_action(xs[i]))
    return xs, torch.stack(ys)


def parallelize_sumo(args, scenario, settings, number_of_worker, run_type):
    envs = [SumoEnv(args.task, scenario, run_type=run_type) for _ in range(number_of_worker)]
    if run_type == "extreme":
        torch.manual_seed(42)
        xs_orig = torch.rand(settings["num_samples_per_scenario"], settings["raw_input_dim"])
    else:
        seed = random.randint(0, 1e6)
        xs_orig = envs[0].get_init_points(settings["num_samples_per_scenario"], settings["raw_input_dim"], seed=seed)
    xs = envs[0].input_transform(xs_orig)
    xs = xs.reshape((number_of_worker, -1, *xs.shape[-2:]))
    xs = [xs[i] for i in range(number_of_worker)]

    multi_worker = mp.Pool(number_of_worker)
    multi_worker_result = multi_worker.map(env_evaluation, zip(envs, xs))
    ys = []
    for result in multi_worker_result:
        _, y = result
        ys.append(y)
    ys = torch.cat(ys, dim=0)

    return xs_orig, ys
