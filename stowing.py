import numpy as np
import uuid
import itertools
import random


def get_letter_codes(num_containers, destinations_num):
    acronims = 'ABCDEFGHIJKLMNOPQRSTUVWZYX'
    destinations = itertools.permutations(acronims, r=2)
    codes = [''.join(el) for el in destinations]        
    times = int(num_containers/destinations_num)
    output = codes[:destinations_num] * times
    assert len(np.unique(output)) == destinations_num, True
    random.shuffle(output)
    return output


class Container:
    def __init__(self, destination):
        self.destination = destination
        self.id = uuid.uuid4().hex


class Terminal:
    def __init__(self,
                 length: int,
                 breadth: int,
                 height: int,
                 num_destinations: int,
                 destin_classes: np.array
                 ):
        self.length = length
        self.breadth = breadth
        self.height = height
        self.num_destinations = num_destinations
        self.destin_classes = self.codes_to_numbers(destin_classes)   
        self.init_terminal()

    def build_field(self):
        size = self.length * self.breadth
        for i in range(size):
            self.slots[i] = list()

    def codes_to_numbers(self, codes):
        return dict({code: digit for digit, code in enumerate(codes)})

    def init_terminal(self):
        self.slots = dict({i: list() for i in range(self.length * self.breadth)})
        self.containers_position = dict()
        self.containers_list = dict()
        self.create_containers_destinations_map()

    def get_containers_locations_map(self):
        map = np.zeros((self.length, self.breadth, self.height), dtype=np.int32)
        for key in self.containers_position.keys():
            l, h = self.containers_position[key]
            map[l,:,h] = 1
        return map.sum(axis=-1)
    
    def get_containers_destinations_map(self):
        return self.map
    
    def create_containers_destinations_map(self):
        self.map = np.ones((self.length, self.breadth, self.height), dtype=np.float32) * -1
        for key in self.containers_list.keys():
            l, h = self.containers_position[key]
            destination = self.containers_list[key].destination
            b = int(l / self.length)
            l = l % self.length
            self.map[l,b,h] = self.destin_classes[destination] / self.num_destinations
    
    def update_containers_destinations_map(self, position, height, dest):
        b = int(position / self.length)
        l = position % self.length

        if dest is not None:
            assert self.map[l,b,height] == -1, True
            self.map[l,b,height] = self.destin_classes[dest] / self.num_destinations
        else:
            assert self.map[l,b,height] != -1, True
            if height < self.height - 1:
                assert self.map[l,b,height+1] == -1, True
            self.map[l,b,height] = -1

    def get_containers_location(self, id):
        return self.containers_position[id]
    
    def get_top_container_dest(self, position):
        return self.slots[position][-1].id

    def track_containers(self, id, dict_key, height):
        self.containers_position[id] = (dict_key, height)

    def combine_by_destination(self, destination):
        container_ids = list()
        for id in self.containers_list.keys():
            if self.containers_list[id].destination == destination:
                container_ids.append(id)
        return container_ids

    def place_container(self,
                        position: int,
                        container: Container):

        reward = 0.
        if self.slot_is_full(position):
            reward =- 1.
            position = self.find_nearest_free_slot(position)
        
        self.slots[position].append(container)
        self.track_containers(container.id, position, len(self.slots[position]) - 1)
        self.containers_list[container.id] = container
        self.update_containers_destinations_map(position, len(self.slots[position]) - 1, container.destination)
        return reward

    def find_containers(self, ids):
        positions = dict()
        for id in ids:
            positions[id] = self.get_containers_location(id)
        return positions

    def find_nearest_free_slot(self, slot):
        addon = 1
        for i in itertools.count():
            if slot + addon < 0 or slot + addon > len(self.slots) - 1:
                addon += -1*(i+2) if not i%2 else i + 2
                continue
            full = self.slot_is_full(slot + addon)
            if not full:
                return slot + addon
            if slot + addon + self.length < len(self.slots):
                full = self.slot_is_full(slot + addon + self.length)
                if not full:
                    return slot + addon + self.length
            if slot + addon - self.length > 0:
                full = self.slot_is_full(slot + addon - self.length)
                if not full:
                    return slot + addon - self.length
            addon += -1*(i+2) if not i%2 else i + 2           

    def replace_container(self, position, new_position=None):
        slot = self.slots[position]
        id = slot[-1].id
        container = self.send_container(id)
        if new_position is None:
            position = self.find_nearest_free_slot(position)
        else:
            position = new_position
        self.place_container(position, container)

    def slot_is_full(self, slot):
        return len(self.slots[slot]) == self.height

    def num_containers_ontop(self, id):
        dict_key, height = self.get_containers_location(id)
        if height == self.height - 1:
            return 0, dict_key, height
        slot = self.slots[dict_key]
        return len(slot) - height - 1, dict_key, height

    def take_container(self, id):
        ontop, dict_key, height = self.num_containers_ontop(id)
        if ontop:
            for i in range(ontop): 
                self.replace_container(dict_key)

        slot = self.slots[dict_key]
        container = slot.pop()
        assert container.id == id, True

        return container, dict_key, len(slot)
    
    def send_container(self, id):
        container, position, height = self.take_container(id)
        del self.containers_position[id]
        del self.containers_list[id]
        self.update_containers_destinations_map(position, height, None)
        return container


def make_action(terminal):
    map = terminal.get_containers_locations_map()
    mask = map == terminal.height
    while True:
        act = random.sample(range(terminal.length*terminal.breadth), k=1)[0]
        if not mask[act,...]:
            break
    return act

def load_containers(terminal, num_containers, destinations_num):
    dest = get_letter_codes(num_containers, destinations_num)
    for i in range(num_containers):
        container = Container(dest[i])
        position = make_action(terminal)
        terminal.place_container(position, container)

    for i in range(terminal.length*terminal.breadth):
        size = len(terminal.slots[i])
        print(f'Slot number {i} has {size} containers')
        for j in range(size):
            print(f'       container {j} with number: {terminal.slots[i][j].id} | destination: {terminal.slots[i][j].destination}')

    map_l = terminal.get_containers_locations_map()
    print(f'Terminal map after loading of {num_containers} containers: \n{map_l.reshape((10,10))}')


    # ids_lists = [terminal.combine_by_destination(dest[random.randint(0, num_containers)]) for _ in range(5)]

    quantity = 25
    destinations = np.random.choice(np.unique(dest), size=quantity, replace=False)
    ids_lists = [terminal.combine_by_destination(destinations[i]) for i in range(quantity)]
    ids = [id for id_lst in ids_lists for id in id_lst]
    containers = [terminal.send_container(id) for id in ids]

    map_d = terminal.get_containers_locations_map()
    print(f'Terminal map after {len(containers)} were discharged: \n{map_d.reshape((10,10))}')

    print(f'Terminal map after discharging: \n{(map_l - map_d).reshape((10,10))}')


if __name__ == '__main__':
    destinations_num = 50
    terminal = Terminal(length=100, breadth=1, height=5, num_destinations=destinations_num)
    load_containers(terminal, 300, destinations_num)
