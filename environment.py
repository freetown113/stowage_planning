import gym
from gym.utils import seeding
from gym.envs.registration import EnvSpec
import numpy as np
import random
import time
import torch
import copy
from typing import Iterable, List, Tuple
from stowing import Terminal, Container, get_letter_codes 



class ContTermEnv:
    metadata = {'render.modes': ['human']}
    spec = EnvSpec("ContTermEnv-v0", entry_point='environment.ContTermEnv')

    def __init__(self,
                 max_steps: int,
                 batch_size: int,
                 length: int,
                 breadth: int,
                 height: int,
                 num_destinations: int,
                 num_containers: int,
                 display: bool = False
                 ) -> None:
        self.max_steps = max_steps
        self.breadth = breadth
        self.height = height
        self.length = length
        self.batch_size = batch_size
        self.show = display
        # self.num_containers_to_discharge = 0
        # self.num_containers_to_load = num_containers
        # self.total_num_containers = num_containers
        self.num_containers = num_containers
        self.num_destinations = num_destinations
        self.loading_order = get_letter_codes(num_containers, num_destinations)
        self.discharging_order = copy.deepcopy(self.loading_order) 
        self.terminal = Terminal(length, breadth, height, num_destinations, 
                                 destin_classes=np.unique(self.loading_order))
        # self.current_ids = None
        # self.ids_lists = None
        

        self.ob_space_shape = [length, breadth, height]
        self.info = dict()
        self.action_space = gym.spaces.Discrete(n=length*breadth)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=self.ob_space_shape,
                                                dtype=np.float32)

    def reset(self):
        self.terminal.init_terminal()
        self.cumul_reward = 0.
        self.current_slot = None
        self.current_container = None
        self.num_containers_to_discharge = 0
        self.step_count = 0
        self.current_ids = None
        self.ids_lists = None
        self.num_containers_to_load = self.num_containers
        self.total_num_containers = self.num_containers
        random.shuffle(self.loading_order)
        random.shuffle(self.discharging_order)
        state = self.terminal.get_containers_destinations_map()
        return state

    def step(self,
             action: np.array
             ) -> Tuple[np.array, float, bool, bool, str]:
        reward = 0.
        position = action.item()
        if self.num_containers_to_load:
            reward = self.load_container(position)
            self.num_containers_to_load -= 1
            # print(f'Container was loaded in position {position}, {self.num_containers_to_load} left to load')

            if self.num_containers_to_load == 0:
                self.num_containers_to_discharge = self.total_num_containers
                self.prepare_discharge_plan()
        if self.num_containers_to_discharge:
            self.discharge_container(position)

        self.step_count += 1
        done = self.check_for_eoe()
        next_state = self.terminal.get_containers_destinations_map()
        trunc = False
        if self.step_count > self.num_containers * 5:
            trunc = True
        info = self.info
        if done:
            reward = self.cumul_reward

        return next_state, reward, done, trunc, info
    
    def load_container(self, position):
        container = Container(self.loading_order[self.step_count])
        reward = self.terminal.place_container(position, container)
        return reward

    def discharge_container(self, position):
        stop = False
        if self.current_slot is not None and self.current_container is not None:
            stop = self.shift_or_discharge(self.current_container, position)
        while not stop:
            if self.check_for_eoe():
                return
            if not len(self.current_ids):
                self.current_ids = self.ids_lists.pop()
            idx = self.current_ids.pop()

            stop = self.shift_or_discharge(idx, position)

    def shift_or_discharge(self, idx, position):
        number, dict_key, _ = self.terminal.num_containers_ontop(idx)
        if number:
            self.current_slot = dict_key
            self.current_container = idx
            self.terminal.replace_container(position=dict_key, new_position=position)
            self.cumul_reward -= 1
            return True
        else:
            container = self.terminal.send_container(idx)
            assert idx == container.id, True
            # print(idx)
            self.num_containers_to_discharge -= 1
            # if self.current_slot is not None and self.current_container is not None and len(self.current_ids):
            #     assert self.current_ids.pop() == idx, True
            self.current_slot = None
            self.current_container = None
            # print(f'Container was discharged from position {position}, {self.num_containers_to_discharge} left to discharge')
            return False
        

    
    # def shift_or_discharge(self, idx, position):
    #     ret_val = False
    #     number, dict_key, _ = self.terminal.num_containers_ontop(idx)
    #     if number:
    #         top_id = self.terminal.get_top_container_dest(dict_key)
    #         self.current_slot = dict_key
    #         self.current_container = idx
    #         if top_id not in self.current_ids:
    #             self.terminal.replace_container(position=dict_key, new_position=position)
    #             self.cumul_reward -= 1
    #             return True
    #         else:
    #             idx = top_id
    #             ret_val = True

    #     container = self.terminal.send_container(idx)
    #     assert idx == container.id, True
    #     # print(idx)
    #     self.num_containers_to_discharge -= 1
    #     # if self.current_slot is not None and self.current_container is not None and len(self.current_ids):
    #     #     assert self.current_ids.pop() == idx, True
    #     self.current_slot = None
    #     self.current_container = None
    #     # print(f'Container was discharged from position {position}, {self.num_containers_to_discharge} left to discharge')
    #     return ret_val



    def check_for_eoe(self):
        if self.current_ids is not None and self.ids_lists is not None:
            return not len(self.current_ids) and not len(self.ids_lists)
        else:
            return False

    def prepare_discharge_plan(self):
        destinations = np.unique(self.discharging_order)
        self.ids_lists = [self.terminal.combine_by_destination(destinations[i]) 
                        for i in range(self.num_destinations)]
        self.current_ids = self.ids_lists.pop()

    def render(self):
        pass

    def close(self):
        pass

    def seed(self,
             seed: int = None
             ) -> List:
        self.np_random, seed1 = seeding.np_random(seed)
        self.np_random, seed2 = seeding.np_random(hash(seed1 + 1) % 2 ** 31)
        return [seed1, seed2]

