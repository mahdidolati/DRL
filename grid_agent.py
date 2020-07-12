import numpy as np
import networkx as nx

try:
    import Queue as Q  # ver. < 3.0
except ImportError:
    import queue as Q

HORIZONTAL_LINK = 0
VERTICAL_LINK = 1
P_UP = 0
P_DOWN = 1
P_LEFT = 2
P_RIGHT = 3
V_UP = 4
V_DOWN = 5
V_LEFT = 6
V_RIGHT = 7
EMBED = 8
LAYER_1 = 1
LAYER_2 = 2
LAYER_3 = 3
LAYER_4 = 4

class netProp:
    def __init__(self):
        self.net_size = 3
        self.cpu_min = 50
        self.cpu_max = 100
        self.bw_min = 50
        self.bw_max = 100

    def get_p_prop(self):
        n = netProp()
        n.net_size = 5
        n.cpu_min = 50
        n.cpu_max = 100
        n.bw_min = 50
        n.bw_max = 100
        return n

    def get_v_prop(self):
        n = netProp()
        n.net_size = 3
        n.cpu_min = 1
        n.cpu_max = 10
        n.bw_min = 1
        n.bw_max = 10
        return n
        

class gridVneEnv:
    def __init__(self, p_net_prop, v_net_prop):
        self.p_net_prop = p_net_prop
        self.v_net_prop = v_net_prop
        self.p_pos = (0,0)
        self.v_pos = (0,0)
        self.embeds = {}
        self.link_embeds = {}
        self.path_exist = True
        self.actions = 9
        self.cur_edges = {}
        self.p_graph_copy = None
        self.cost = 0.0
        self.rev = 0.0
        self.node_bw_embed = [0.0]*p_net_prop.net_size
        self.cost_rev_ratio = 0.0
        self.avg_cpu_util = 0.0
        self.avg_bw_util = 0.0
        self.max_revenue_so_far = 0.0

    def read_p_net(self, r_path):
        self.p_graph = nx.read_gpickle(r_path)
        self.p_graph_copy = self.p_graph.copy()

    def write_p_net(self, w_path):
        nx.write_gpickle(self.p_graph, w_path)

    def create_p_net(self):
        p_net_prop = self.p_net_prop
        v_net_prop = self.v_net_prop
        n = p_net_prop.net_size
        env_graph = nx.Graph()
        for i in range(n):
            for j in range(n):
                env_graph.add_node((i,j))
                env_graph.node[(i,j)]['cpu'] = np.random.uniform(p_net_prop.cpu_min, p_net_prop.cpu_max, 1)[0]
                env_graph.node[(i,j)]['here'] = 0
                env_graph.node[(i,j)]['embeds'] = [0]*(v_net_prop.net_size*v_net_prop.net_size)
        for i in range(n):
            for j in range(n):
                if i+1 <= n-1:
                    env_graph.add_edge((i,j),(i+1,j))
                if j+1 <= n-1:
                    env_graph.add_edge((i, j), (i, j+1))
                if i > 0:
                    env_graph.add_edge((i, j), (i-1, j))
                if j > 0:
                    env_graph.add_edge((i, j), (i, j-1))
                for (s,t) in env_graph.edges((i,j)):
                    env_graph[s][t]['bw'] = np.random.uniform(p_net_prop.bw_min, p_net_prop.bw_max, 1)[0]
                    if s[1] == t[1]: # if it is a vertical link
                        env_graph[s][t]['here'] = [0]
                        env_graph[s][t]['embeds'] = [0] * (v_net_prop.net_size*v_net_prop.net_size)
                        if s[1] != self.p_net_prop.net_size - 1: #last b
                            env_graph[s][t]['u_bw'] = [0]
        self.p_graph = env_graph
        self.p_graph_copy = self.p_graph.copy()
        #print self.p_graph.nodes
        #print self.p_graph.edges


    def create_v_net(self):
        v_net_prop = self.v_net_prop
        n = v_net_prop.net_size
        v_graph = nx.Graph()
        node_id = 0
        this_revenue = 0.0
        for i in range(n):
            for j in range(n):
                v_graph.add_node((i, j))
                v_graph.node[(i,j)]['cpu'] = np.random.uniform(v_net_prop.cpu_min, v_net_prop.cpu_max, 1)[0]
                v_graph.node[(i,j)]['id'] = node_id
                v_graph.node[(i, j)]['oh_id'] = [0] * (v_net_prop.net_size*v_net_prop.net_size)
                v_graph.node[(i, j)]['oh_id'][node_id] = 1
                v_graph.node[(i,j)]['placed'] = 0
                v_graph.node[(i,j)]['here'] = 0
                node_id += 1
                this_revenue += v_graph.node[(i,j)]['cpu']
        for i in range(n):
            for j in range(n):
                if i+1 <= n-1:
                    v_graph.add_edge((i,j),(i+1,j))
                if j+1 <= n-1:
                    v_graph.add_edge((i, j), (i, j+1))
                if i > 0:
                    v_graph.add_edge((i, j), (i-1, j))
                if j > 0:
                    v_graph.add_edge((i, j), (i, j-1))
                for (s,t) in v_graph.edges((i,j)):
                    v_graph[s][t]['bw'] = np.random.uniform(v_net_prop.bw_min, v_net_prop.bw_max, 1)[0]
                    if s[1] == t[1]: # if it is a vertical link
                        v_graph[s][t]['here'] = [0]
                        v_graph[s][t]['id'] = [0] * (v_net_prop.net_size*v_net_prop.net_size)
                        v_graph[s][t]['placed'] = [0]
                        if s[1] != self.v_net_prop.net_size - 1: #last vertical link does not have upper vertical link
                            v_graph[s][t]['u_bw'] = [0]
                    this_revenue += v_graph[s][t]['bw']
        if this_revenue > self.max_revenue_so_far:
            self.max_revenue_so_far = this_revenue
        self.v_graph = v_graph

    def move_in_p(self, a):
        self.p_graph.node[self.p_pos]['here'] = 0
        if a == P_LEFT and self.p_pos[1] > 0:
            self.p_pos = (self.p_pos[0], self.p_pos[1] - 1)
        if a == P_RIGHT and self.p_pos[1] < self.p_net_prop.net_size-1:
            self.p_pos = (self.p_pos[0], self.p_pos[1] + 1)
        if a == P_UP and self.p_pos[0] > 0:
            self.p_pos = (self.p_pos[0] - 1, self.p_pos[1])
        if a == P_DOWN and self.p_pos[0] < self.p_net_prop.net_size-1:
            self.p_pos = (self.p_pos[0] + 1, self.p_pos[1])
        #print 'current p position:', self.p_pos
        self.p_graph.node[self.p_pos]['here'] = 1

    def move_in_v(self, a):
        self.v_graph.node[self.v_pos]['here'] = 0
        if a == V_LEFT and self.v_pos[1] > 0:
            self.v_pos = (self.v_pos[0], self.v_pos[1] - 1)
        if a == V_RIGHT and self.v_pos[1] < self.v_net_prop.net_size - 1:
            self.v_pos = (self.v_pos[0], self.v_pos[1] + 1)
        if a == V_UP and self.v_pos[0] > 0:
            self.v_pos = (self.v_pos[0] - 1, self.v_pos[1])
        if a == V_DOWN and self.v_pos[0] < self.v_net_prop.net_size - 1:
            self.v_pos = (self.v_pos[0] + 1, self.v_pos[1])
        #print 'current v position:', self.v_pos
        self.v_graph.node[self.v_pos]['here'] = 1

    def reset(self):
        self.p_graph = self.p_graph_copy.copy()
        self.create_v_net()
        self.embeds = {}
        self.link_embeds = {}
        self.path_exist = True
        self.p_pos = (0, 0)
        self.v_pos = (0, 0)
        self.cost = 0.0
        self.rev = 0.0
        self.cost_rev_ratio = 0.0
        self.avg_bw_util = 0.0
        self.avg_cpu_util = 0.0
        return self.get_rpr()

    def release_reserves(self):
        for v in self.embeds:
            self.p_graph.node[self.embeds[v]]['cpu'] += self.v_graph.node[v]['cpu']
        for e in self.link_embeds:
            self.p_graph[e[0]][e[1]]['bw'] += self.link_embeds[e]
        self.embeds = {}
        self.link_embeds = {}
        self.path_exist = True
        self.p_pos = (0, 0)
        self.v_pos = (0, 0)

    def calc_utils(self):
        self.avg_bw_util = 0.0
        self.avg_cpu_util = 0.0
        for v in self.p_graph.nodes:
            self.avg_cpu_util += (1- (self.p_graph.node[v]['cpu']/self.p_graph_copy.node[v]['cpu']))
        self.avg_cpu_util /= self.p_graph.number_of_nodes()
        for e in self.p_graph.edges:
            self.avg_bw_util += (1 - (self.p_graph[e[0]][e[1]]['bw']/self.p_graph_copy[e[0]][e[1]]['bw']))
        self.avg_bw_util /= self.p_graph.number_of_edges()

    def calc_cost(self):
        for v in self.embeds:
            self.cost += self.v_graph.node[v]['cpu']
        for e in self.link_embeds:
            self.cost += self.link_embeds[e]

    def calc_revenue(self):
        for v in self.v_graph.nodes:
            self.rev += self.v_graph.node[v]['cpu']
        for e in self.v_graph.edges:
            self.rev += self.v_graph[e[0]][e[1]]['bw']

    # amount of bandwidth inside
    def bw_in_node(self, p_node):
        bw = 0.0
        in_p_node = []
        for v in self.v_graph.nodes:
            if v in self.embeds and self.embeds[v] == p_node:
                for w in in_p_node:
                    if self.v_graph.has_edge(v,w):
                        bw += self.v_graph[v][w]['bw']
                in_p_node.append(v)
        return bw/40.0

    def bw_in_cur_node(self):
        bw = 0.0
        for v in self.get_grid_neighbor(self.v_pos, self.v_net_prop.net_size-1):
            if v in self.embeds:
                bw += self.v_graph[v][self.v_pos]['bw']
        return bw/40.0

    def embed(self):
        done = False
        if self.path_exist and self.p_graph.node[self.p_pos]['cpu'] >= self.v_graph.node[self.v_pos]['cpu']:
            v_id = self.v_graph.node[self.v_pos]['id']
            self.v_graph.node[self.v_pos]['placed'] = 1
            self.p_graph.node[self.p_pos]['embeds'][v_id] = 1
            self.p_graph.node[self.p_pos]['cpu'] -= self.v_graph.node[self.v_pos]['cpu']
            self.embeds[self.v_pos] = self.p_pos
            for e in self.cur_edges:
                self.p_graph[e[0]][e[1]]['bw'] -= self.cur_edges[e]
                if e not in self.link_embeds:
                    self.link_embeds[e] = self.cur_edges[e]
                else:
                    self.link_embeds[e] += self.cur_edges[e]
            rew = 0.01 #rew = self.bw_in_cur_node() # -len(self.cur_edges) / 100.0  #
            self.cur_edges = {}
        else:
            rew = -0.1
            self.release_reserves()
            #self.calc_utils()
            done = True
        if len(self.embeds) == self.v_net_prop.net_size*self.v_net_prop.net_size:
            #print 'embedded!'
            rew = 1
            self.calc_cost()
            self.calc_revenue()
            self.cost_rev_ratio = self.rev / self.cost
            rew = self.rev / self.max_revenue_so_far
            #rew = self.cost_rev_ratio
            #done = True
            self.create_v_net()
            self.embeds = {}
            self.link_embeds = {}
            self.path_exist = True
            self.p_pos = (0, 0)
            self.v_pos = (0, 0)
        return rew, done

    def get_rpr_layer(self, layer):
        p_rpr = self.get_p_rpr(layer)
        v_rpr = self.get_v_rpr()
        p_width = len(p_rpr) / (2 * self.p_net_prop.net_size - 1)
        v_width = len(v_rpr) / (2 * self.v_net_prop.net_size - 1)
        h_diff = (2 * self.p_net_prop.net_size - 1) - (2 * self.v_net_prop.net_size - 1)
        h_diff /= 2
        padding_size = h_diff * v_width
        v_rpr = np.concatenate((padding_size * [0], v_rpr, padding_size * [0]))
        p_rpr = np.reshape(p_rpr, ((2 * self.p_net_prop.net_size - 1), p_width))
        v_rpr = np.reshape(v_rpr, ((2 * self.p_net_prop.net_size - 1), v_width))
        rpr = np.hstack((p_rpr, v_rpr))
        pad = np.zeros((2 * self.p_net_prop.net_size - 1, 3))
        rpr = np.hstack((rpr, pad))
        rpr = np.concatenate(rpr)
        return rpr

    def get_rpr(self):
        l1_rpr = self.get_rpr_layer(LAYER_1)
        l2_rpr = self.get_rpr_layer(LAYER_2)
        l3_rpr = self.get_rpr_layer(LAYER_3)
        return np.concatenate((l1_rpr, l2_rpr, l3_rpr))

    def get_v_node_rpr(self, n_id, layer=LAYER_1):
        if layer == LAYER_1 or layer == LAYER_2:
            rpr = np.concatenate(([self.v_graph.node[n_id]['cpu']/self.p_net_prop.cpu_max],
                                  [self.v_graph.node[n_id]['here']],
                                  [self.v_graph.node[n_id]['placed']],
                                  self.v_graph.node[n_id]['oh_id']))
        elif layer == LAYER_3:
            if self.v_graph.node[n_id]['placed'] == 1:
                rpr = np.array([1] * (1+1+1+(self.v_net_prop.net_size)*(self.v_net_prop.net_size)))
            else:
                rpr = np.array([0] * (1 + 1 + 1 + (self.v_net_prop.net_size) * (self.v_net_prop.net_size)))
        return rpr

    def get_p_node_rpr(self, n_id, layer=LAYER_1):
        cpu = self.p_graph.node[n_id]['cpu']
        if n_id == self.p_pos:
            if self.v_graph.node[self.v_pos]['placed'] == 0:
                cpu -= self.v_graph.node[self.v_pos]['cpu']
        cpu /= self.p_net_prop.cpu_max
        if layer == LAYER_1 or layer == LAYER_3:
            rpr = np.concatenate((np.array([cpu]),
                                  [self.p_graph.node[n_id]['here']],
                                  self.p_graph.node[n_id]['embeds']))
        elif layer == LAYER_2:
            if cpu < 0:
                rpr = np.array([1]*(1+1+self.v_net_prop.net_size*self.v_net_prop.net_size))
            else:
                rpr = np.array([0] * (1 + 1 + self.v_net_prop.net_size * self.v_net_prop.net_size))
        elif layer == LAYER_4:
            bw_in_n = self.bw_in_node(n_id)
            rpr  = np.array([bw_in_n]*(1+1+self.v_net_prop.net_size*self.v_net_prop.net_size))
        return rpr

    def get_p_link_bw(self, s, t):
        bw = self.p_graph[s][t]['bw']
        for e in self.cur_edges:
            if (e[0] == s and e[1] == t) or (e[0] == t and e[1] == s):
                bw -= self.cur_edges[e]
        return bw

    def get_p_link_rpr(self, l_id, layer=LAYER_1):
        s = l_id[0]
        t = l_id[1]
        l_bw = self.get_p_link_bw(s, t)
        l_bw /= self.p_net_prop.bw_max
        if layer == LAYER_1 or layer == LAYER_4:
            if s[0] == t[0]: #horizontal link
                rpr = np.array([l_bw])
            elif s[1] == t[1] and s[1] == self.p_net_prop.net_size-1: #vertical link at last column
                rpr = np.concatenate((np.array([l_bw]),
                                      self.p_graph[s][t]['here'],
                                      self.p_graph[s][t]['embeds']))
            else: #intermediate vertical links
                rpr = np.concatenate((np.array([l_bw]),
                                      self.p_graph[s][t]['here'],
                                      self.p_graph[s][t]['embeds'],
                                      self.p_graph[s][t]['u_bw']))
        elif layer == LAYER_2:
            if l_bw < 0:
                if s[0] == t[0]: #horizontal link
                    rpr = np.array([1])
                elif s[1] == t[1] and s[1] == self.p_net_prop.net_size-1: #vertical link at last column
                    rpr = np.array([1]*(1+1+self.v_net_prop.net_size*self.v_net_prop.net_size))
                else: #intermediate vertical links
                    rpr = np.array([1]*(1+1+self.v_net_prop.net_size*self.v_net_prop.net_size+1))
            else:
                if s[0] == t[0]: #horizontal link
                    rpr = np.array([0])
                elif s[1] == t[1] and s[1] == self.p_net_prop.net_size-1: #vertical link at last column
                    rpr = np.array([0]*(1+1+self.v_net_prop.net_size*self.v_net_prop.net_size))
                else: #intermediate vertical links
                    rpr = np.array([0]*(1+1+self.v_net_prop.net_size*self.v_net_prop.net_size+1))
        elif layer == LAYER_3:
            if s[0] == t[0]:  # horizontal link
                rpr = np.array([0])
            elif s[1] == t[1] and s[1] == self.p_net_prop.net_size - 1:  # vertical link at last column
                rpr = np.array([0] * (1 + 1 + self.v_net_prop.net_size * self.v_net_prop.net_size))
            else:  # intermediate vertical links
                rpr = np.array([0] * (1 + 1 + self.v_net_prop.net_size * self.v_net_prop.net_size + 1))
            for e in self.cur_edges:
                if (e[0] == s and e[1] == t) or (e[1] == s and e[0] == t):
                    if s[0] == t[0]:  # horizontal link
                        rpr = np.array([1])
                    elif s[1] == t[1] and s[1] == self.p_net_prop.net_size - 1:  # vertical link at last column
                        rpr = np.array([1] * (1 + 1 + self.v_net_prop.net_size * self.v_net_prop.net_size))
                    else:  # intermediate vertical links
                        rpr = np.array([1] * (1 + 1 + self.v_net_prop.net_size * self.v_net_prop.net_size + 1))
        return rpr

    def get_v_link_rpr(self, l_id, layer=LAYER_1):
        s = l_id[0]
        t = l_id[1]
        bw = self.v_graph[s][t]['bw'] / self.p_net_prop.bw_max
        if s[0] == t[0]: #horizontal link
            rpr = np.array([bw])
        elif s[1] == t[1] and s[1] == self.v_net_prop.net_size-1: #vertical link at last column
            rpr = np.concatenate(([bw],
                                  self.v_graph[s][t]['here'],
                                  self.v_graph[s][t]['id'],
                                  self.v_graph[s][t]['placed']))
        else: #intermediate vertical links
            rpr = np.concatenate(([bw],
                                  self.v_graph[s][t]['here'],
                                  self.v_graph[s][t]['id'],
                                  self.v_graph[s][t]['placed'],
                                  self.v_graph[s][t]['u_bw']))
        return rpr

    def get_p_rpr(self, layer=LAYER_1):
        n = self.p_net_prop.net_size
        all_rpr = [None]*(3*n*n - 2*n)
        rpr_index = 0
        for i in range(n):
            for j in range(n):
                rpr = self.get_p_node_rpr((i,j), layer=layer)
                all_rpr[rpr_index] = rpr
                rpr_index += 1
                if j != n-1:
                    rpr = self.get_p_link_rpr(((i,j),(i,j+1)), layer=layer)
                    all_rpr[rpr_index] = rpr
                    rpr_index += 1
            for j in range(n):
                if i != n-1:
                    rpr = self.get_p_link_rpr(((i,j),(i+1,j)), layer=layer)
                    all_rpr[rpr_index] = rpr
                    rpr_index += 1
        return np.concatenate(all_rpr, axis=0)

    def get_v_rpr(self, layer=LAYER_1):
        n = self.v_net_prop.net_size
        all_rpr = [None]*(3*n*n - 2*n)
        rpr_index = 0
        for i in range(n):
            for j in range(n):
                rpr = self.get_v_node_rpr((i,j), layer)
                all_rpr[rpr_index] = rpr
                rpr_index += 1
                if j != n-1:
                    rpr = self.get_v_link_rpr(((i,j),(i,j+1)))
                    all_rpr[rpr_index] = rpr
                    rpr_index += 1
            for j in range(n):
                if i != n-1:
                    rpr = self.get_v_link_rpr(((i,j),(i+1,j)))
                    all_rpr[rpr_index] = rpr
                    rpr_index += 1
        return np.concatenate(all_rpr, axis=0)

    def step(self, a):
        rew = 0
        done = False
        if a >= 0 and a <= 3:
            self.move_in_p(a)
        if a >= 4 and a <= 7:
            self.move_in_v(a)
        if a == 8:
            rew, done = self.embed()
        self.update_env()
        return self.get_rpr(), rew, done

    def update_env(self):
        self.cur_bw_usage()

    def get_grid_neighbor(self, pos, idx_max):
        neighbors = []
        if pos[0] > 0:
            n = (pos[0]-1, pos[1])
            neighbors.append(n)
        if pos[1] > 0:
            n = (pos[0], pos[1] - 1)
            neighbors.append(n)
        if pos[0] < idx_max:
            n = (pos[0] + 1, pos[1])
            neighbors.append(n)
        if pos[1] < idx_max:
            n = (pos[0], pos[1] + 1)
            neighbors.append(n)
        return neighbors

    def cur_bw_usage(self):
        self.path_exist = True
        self.cur_edges = {}
        neighbors = self.get_grid_neighbor(self.v_pos, self.v_net_prop.net_size-1)
        for n in neighbors:
            if n in self.embeds:
                links = self.links_bw_usage(self.p_pos, self.embeds[n], self.v_graph[n][self.v_pos]['bw'])
                if links is None:
                    self.path_exist = False
                    links = self.links_bw_usage(self.p_pos, self.embeds[n], 0)
                if links is None:
                    self.path_exist = False
                    continue
                for l in links:
                    if l not in self.cur_edges:
                        self.cur_edges[l] = self.v_graph[n][self.v_pos]['bw']
                    else:
                        self.cur_edges[l] += self.v_graph[n][self.v_pos]['bw']

    def links_bw_usage(self, s, t, bw):
        if s == t:
            return []
        q = Q.Queue()
        q.put(s)
        visited = {}
        visited[s] = None
        while q.empty() == False:
            cur = q.get()
            for n in self.p_graph.neighbors(cur):
                if self.p_graph[cur][n]['bw'] < bw:
                    continue
                if n == t:
                    visited[t] = cur
                    break
                if n not in visited:
                    visited[n] = cur
                    q.put(n)
            if t in visited:
                break
        if t not in visited:
            return None
        links = []
        cur = t
        while visited[cur] is not None:
            links.append((visited[cur],cur))
            cur = visited[cur]
        return links


if __name__ == "__main__":
    net_prop = netProp()
    p_net_prop = net_prop.get_p_prop()
    v_net_prop = net_prop.get_v_prop()
    grid_env = gridVneEnv(p_net_prop, v_net_prop)
    grid_env.create_p_net()
    grid_env.create_v_net()
    grid_env.update_env()
    all_rpr = grid_env.get_rpr()
    print len(all_rpr)
    grid_env.step(EMBED)
    grid_env.step(V_RIGHT)
    grid_env.step(P_RIGHT)
    grid_env.step(EMBED)
