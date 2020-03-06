class Node:
    state = None
    depth = None
    cost = None
    node = None

    def __init__(self,state=None,depth=0,cost=0.0, node=None):
        self.state = state
        self.depth = depth
        self.cost = cost
        self.node = node

    def get_cost(self):
        return self.cost

    def set_cost(self,cost):
        self.cost = cost

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def get_depth(self):
        return self.depth

    def set_depth(self, depth):
        self.depth = depth

    def get_node(self):
        return self.node

    def set_node(self,node):
        self.node = node

    def __str__(self):
        return "state:" + self.state + "depth:" + self.depth + "cost:"+self.cost+"node:"+ self.node
