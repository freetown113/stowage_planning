import gym
from gym.utils import seeding
from gym.envs.registration import EnvSpec
import numpy as np
import random
import copy
from typing import Iterable, List, Tuple
from stowing import Terminal, Container, get_letter_codes 


class ContTermEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    spec = EnvSpec("ContTermEnv-v2", entry_point='environment.ContTermEnv')
    '''
    Environment is an imitation of a container terminal that is used to stow containers.
    The containers come in one portion to be loaded one by one according to the loading plan.
    After all containers were loaded, they started to be discharged according to the discharging plan.
    The discharge plan is just a shuffled version of the loading plan. Containes are combined by the 
    destination and will be dicharged one by one group by group.  
    Each time when a contained is going to be loaded agent is asked to make an ACTION in the ENV.
    If it try to load container in place where is prohibited to load it will be penalized with 
    some negative reward.
    Each time when a container is going to be discharged it is checked if precise contained is covered ontop.
    If it is the case the AGENT will get negative reward. Container from top will be removed and the AGENT 
    will be asked to make an ACTION to place the container in some other position.   
    '''
    def __init__(self,
                 max_steps: int,
                 batch_size: int,
                 length: int,
                 breadth: int,
                 height: int,
                 num_destinations: int,
                 num_containers: int,
                 act_type: str,
                 obs_type: str,
                 out_type: str,
                 version: str
                 ) -> None:
        self.max_steps = max_steps
        self.breadth = breadth
        self.height = height
        self.length = length
        self.batch_size = batch_size
        self.act_type = act_type
        self.version = version
        self.obs_type = obs_type
        self.out_type = out_type
        self.num_containers = num_containers
        self.num_destinations = num_destinations
        self.loading_order = get_letter_codes(num_containers, num_destinations)
        self.discharging_order = copy.deepcopy(self.loading_order) 
        self.terminal = Terminal(length, breadth, height, num_destinations, 
                                 destin_classes=np.unique(self.loading_order))
        
        self.info = dict()
        if self.act_type == 'discret':
            self.action_space = gym.spaces.Discrete(n=length*breadth)
        elif self.act_type == 'continuous':
            self.action_space = gym.spaces.Box(low=-1., high=1,
                                                shape=[2],
                                                dtype=np.float32)
        else:
            raise TypeError(f'The provided action type doesnt exist {self.act_type}')
        
        if self.obs_type == 'mlp':
            self.ob_space_shape = [breadth * length * height]
        elif self.obs_type == 'cnn':
            if self.out_type == 'torch':
                self.ob_space_shape = [height, length, breadth]
            elif self.out_type == 'numpy':
                self.ob_space_shape = [length, breadth, height]
        elif self.obs_type == '1d':
            self.ob_space_shape = [height, breadth * length]
        else:
            raise TypeError
        self.observation_space = gym.spaces.Box(low=-1., high=1.,
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
        return self.get_state()
    
    def step(self,
             action: np.array
             ) -> Tuple[np.array, float, bool, bool, str]:
        return self.step_v2(action)
    
    def step_v2(self,
             action: np.array
             ) -> Tuple[np.array, float, bool, bool, str]:
        reward = 0.
        if self.act_type == 'discret':
            position = action.item()
        elif self.act_type == 'continuous':
            position = self.get_from_continious(action)
        if self.num_containers_to_load:
            if self.terminal.slot_is_full(position):
                reward = -1.0 * self.num_containers_to_load / self.num_containers - 1.
                done = True
                self.step_count += 1
                info = {'r': reward, 'lgth': self.step_count}
                return self.get_state(), reward, done, info
            else:
                reward = self.load_container(position)
                self.num_containers_to_load -= 1
            
            if self.num_containers_to_load == 0:
                self.prepare_discharge_plan()


                if self.version == 'complex':
                    self.num_containers_to_discharge = self.total_num_containers
                    self.discharge_while_possible()
                elif self.version == 'simple':
                    self.num_containers_to_discharge = 0
                    self.calculate_final_reward()


                # self.num_containers_to_discharge = self.total_num_containers
                # self.discharge_while_possible()

        elif self.num_containers_to_discharge:
            if self.terminal.slot_is_full(position):
                reward = -1.0 * self.num_containers_to_discharge / self.num_containers + 1. + self.cumul_reward
                done = True
                self.step_count += 1
                info = {'r': reward, 'lgth': self.step_count}
                return self.get_state(), reward, done, info
            self.discharge_container(position)

        self.step_count += 1
        done = self.check_for_eoe() #or self.step_count > self.num_containers * 3

        if done:
            assert self.num_containers_to_discharge == 0, True
            # if self.num_containers_to_discharge != 0:
            #     print()
            if self.version == 'simple':
                reward = 1. + self.cumul_reward
            elif self.version == 'complex':
                reward = 2. + self.cumul_reward
        info = {'r': reward, 'lgth': self.step_count}

        if self.step_count > self.num_containers * 3:
            return (self.get_state(), 
                    -1.0 * self.num_containers_to_discharge / self.num_containers + 1. + self.cumul_reward, 
                    True, 
                    {'r': reward, 'lgth': self.step_count, 'bad_transition': True})

        return self.get_state(), reward, done, info
    
    def load_container(self, position):
        '''
        Create container with unique ID and destination code according to loading plan
        Set container to the position provided by the AGENT
        '''
        container = Container(self.loading_order[self.step_count])
        reward = self.terminal.place_container(position, container)
        return reward
    
    def discharge_while_possible(self):
        self.discharge_container(position=None)

    def discharge_container(self, position):
        '''
        Try to discharge 
        '''
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
        '''
        Discharge container by provided ID:
            - if container is on top of it's row it'll be discharged and removed from the discharge plan
            - if container is covered from top it's ID will be saved. Until all containers from top will
                be moved in other positions. Only after it is free it will be discharged and removed from
                the discharge plan.  
        '''
        ret_val = False
        number, dict_key, _ = self.terminal.num_containers_ontop(idx)
        if number:       
            self.current_slot = dict_key
            self.current_container = idx
            if position is None:
                return True
            top_id = self.terminal.get_top_container_dest(dict_key)
            if top_id not in self.current_ids:
                self.terminal.replace_container(position=dict_key, new_position=position)
                self.cumul_reward -= 1 / self.num_containers
                return True
            else:
                idx = top_id
                self.current_ids.remove(top_id)
                container = self.terminal.send_container(idx)
                assert idx == container.id, True
                self.num_containers_to_discharge -= 1
                return True

        container = self.terminal.send_container(idx)
        assert idx == container.id, True
        # print(idx)
        self.num_containers_to_discharge -= 1
        if (self.terminal.get_containers_destinations_map()!= -1).sum() != self.num_containers_to_discharge:
            print()
        if self.num_containers_to_discharge - np.asarray(self.ids_lists).size - len(self.current_ids) > 1:
            print()
        # if self.current_slot is not None and self.current_container is not None and len(self.current_ids):
        #     assert self.current_ids.pop() == idx, True
        self.current_slot = None
        self.current_container = None
        # print(f'Container was discharged from position {position}, {self.num_containers_to_discharge} left to discharge')
        return ret_val
    
    def calculate_final_reward(self):
        '''
        Calculate reward according to:
            - after all containers were loaded, start to discharge them one by one following the discharge plan
            - if the selected container is free, discharge it without negative reward
            - if the selected container is enclosed, remove containers from top to the nearest empty bays
            - for each removed container set reward -1.0
            - continue untill discharge plan is empty
        '''
        while True:
            if self.check_for_eoe():
                return
            if not len(self.current_ids):
                self.current_ids = self.ids_lists.pop()
            idx = self.current_ids.pop()

            while True:
                number, dict_key, _ = self.terminal.num_containers_ontop(idx)
                if number:       
                    top_id = self.terminal.get_top_container_dest(dict_key)
                    if top_id not in self.current_ids:
                        position = self.terminal.find_nearest_free_slot(dict_key)
                        self.terminal.replace_container(position=dict_key, new_position=position)
                        self.cumul_reward -= 1 / self.num_containers
                        continue
                    else:
                        self.current_ids.remove(top_id)
                        self.current_ids.append(idx)
                        idx = top_id

                container = self.terminal.send_container(idx)
                assert idx == container.id, True
                break


    def check_for_eoe(self):
        '''
        Check is episode is completed:
            - episode ends only when discharge plan is empty
        '''
        if self.current_ids is not None and self.ids_lists is not None:
            # if self.current_slot is None and self.current_container is None:
            return not len(self.current_ids) and not len(self.ids_lists) and self.current_slot is None and self.current_container is None
        else:
            return False

    def prepare_discharge_plan(self):
        '''
        Preparation of the discharge plan: 
            - get unique destination codes from all the containers. 
            - combine container in groups according to the destinations codes
            - prepare list of lists with ID of containers
        '''
        destinations = np.unique(self.discharging_order)
        self.ids_lists = [self.terminal.combine_by_destination(destinations[i]) 
                        for i in range(self.num_destinations)]
        self.current_ids = self.ids_lists.pop()

    def get_from_continious(self, action):
        length = int(round((action[0] / 2 + 0.5) * (self.length - 1), 0))
        breadth = int(round((action[1] / 2 + 0.5) * (self.breadth - 1), 0))
        position = length + breadth * self.length
        return position
    
    def get_state(self):
        state = self.terminal.get_containers_destinations_map()
        if self.obs_type == 'mlp':
            state = np.reshape(state, (self.length * self.breadth * self.height))
        elif self.obs_type == 'cnn':
            if self.out_type == 'torch':
                state = state.transpose(2,0,1)
            elif self.out_type == 'numpy':
                state = state
        elif self.obs_type == '1d':
            state = np.reshape(state, (self.length * self.breadth, self.height)).transpose(1, 0)
        else:
            raise TypeError(f'Observation type {self.obs_type} is unknown')
        return state

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



from gym.envs.registration import register

register(
    id='ContTermEnv-v2',
    entry_point='environment:ContTermEnv',
    # max_episode_steps=10000,
    kwargs = {
    'max_steps': 10000,
    'batch_size': 32,
    'length': 8,
    'breadth': 8,
    'height': 5,
    'num_destinations': 5,
    'num_containers': 200,
    'act_type': 'continuous',
    'obs_type': '1d',
    'out_type': 'torch',
    'version': 'simple'
    }    
)













    # def shift_or_discharge(self, idx, position):
    #     number, dict_key, _ = self.terminal.num_containers_ontop(idx)
    #     if number:
    #         self.current_slot = dict_key
    #         self.current_container = idx
    #         self.terminal.replace_container(position=dict_key, new_position=position)
    #         self.cumul_reward -= 1
    #         return True
    #     else:
    #         container = self.terminal.send_container(idx)
    #         assert idx == container.id, True
    #         # print(idx)
    #         self.num_containers_to_discharge -= 1
    #         # if self.current_slot is not None and self.current_container is not None and len(self.current_ids):
    #         #     assert self.current_ids.pop() == idx, True
    #         self.current_slot = None
    #         self.current_container = None
    #         # print(f'Container was discharged from position {position}, {self.num_containers_to_discharge} left to discharge')
    #         return False