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

    def return_cost(self):
        return self.cost

    def put_cost(self,cost):
        self.cost = cost

    def return_state(self):
        return self.state

    def put_state(self, state):
        self.state = state

    def return_depth(self):
        return self.depth

    def put_depth(self, depth):
        self.depth = depth

    def return_node(self):
        return self.node

    def put_node(self,node):
        self.node = node

    def __str__(self):
        return "state:" + self.state + "depth:" + self.depth + "cost:"+self.cost+"node:"+ self.node
