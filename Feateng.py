class MOFNet(nn.Module):
    def __init__(self, params, ni, dim, aggr, act=None):
        super(MOFNet, self).__init__()
        self.params = params

        if act == "tanh":
            self.act = torch.tanh
        elif act == "sigmoid":
            self.act = torch.sigmoid
        elif act == "relu":
            self.act = torch.relu
        elif act == "softplus":
            self.act = F.softplus
        elif act == "elu":
            self.act = F.elu

        if not self.params["use_n2v_emb"]:
            self.iden_emb = nn.Embedding(50, 8)
            self.linear0 = nn.Linear(8, dim)
        else:
            self.iden_emb = nn.Embedding(params["node_num"], params["n2v_dim"])
            self.linear0 = nn.Linear(params["n2v_dim"], dim)

        def get_conv(ni):
            conv0 = pyg_nn.GINConv(nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)))
            conv1 = pyg_nn.GCNConv(dim, dim, aggr=aggr, add_self_loops=True)
            conv2 = pyg_nn.AGNNConv(add_self_loops=True)
            conv3 = pyg_nn.ClusterGCNConv(dim, dim, aggr=aggr, add_self_loops=True)
            conv4 = pyg_nn.GATConv(dim, dim, aggr=aggr, add_self_loops=True)
            conv5 = pyg_nn.GraphConv(dim, dim, aggr=aggr)
            conv6 = pyg_nn.LEConv(dim, dim, aggr=aggr)
            conv7 = pyg_nn.MFConv(dim, dim, aggr=aggr)
            conv8 = pyg_nn.SAGEConv(dim, dim, aggr=aggr)
            convs = [conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8] 
            conv = convs[ni] 
            return conv

        self.conv = get_conv(ni)
        self.gru = GRU(dim, dim)
        self.set2set = Set2Set(dim, processing_steps=2)

        self.node_emb = nn.Embedding(300, params["embed_dim"]) # params["node_num"], params["embed_dim"])

        self.topo_emb = nn.Embedding(500, params["embed_dim"], padding_idx=params["topo_pad"]) # params["topo_num"], params["embed_dim"], padding_idx=params["topo_pad"])

        dim_ = len(params["struc"]) if params["use_struc"] else 0
        dim__ = len(params["fact"]) if params["use_fact"] else 0  
        fc_dim = 2 * dim + dim_ + dim__ + 2 * params["embed_dim"]

        self.fc1 = torch.nn.Linear(fc_dim, 2 * dim)
        self.fc11 = torch.nn.Linear(2 * dim, dim)
        self.fc12 = torch.nn.Linear(dim, dim)
        self.fc13 = torch.nn.Linear(dim, 1)

    def forward(self, data):
        if not self.params["use_n2v_emb"]:
            x = self.iden_emb(data.x)
        else:
            x = data.x

        out = self.act(self.linear0(x))
        h = out.unsqueeze(0)
        for i in range(3):
            m = self.act(self.conv(out, data.edge_index)) + out
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
        out = self.set2set(out, data.batch)

        out_node = self.node_emb(data.node)
        out_topo = self.topo_emb(data.topo)

        if self.params["use_struc"]:
            if self.params["use_fact"]: 

                out = torch.cat([out, out_node, out_topo, torch.tensor(data.struc, dtype=torch.float32),
                                torch.tensor(data.fact, dtype=torch.float32)], dim=1) 

            else:
                out = torch.cat([out, out_node, out_topo, torch.tensor(data.struc, dtype=torch.float32)], dim=1)
        else:
            out = torch.cat([out, out_node, out_topo], dim=1)

        x1 = self.act(self.fc1(out))
        x1 = self.act(self.fc11(x1))
        x1 = self.act(self.fc12(x1))
        x1 = self.fc13(x1)

        return x1

class MOFNet_FNN(nn.Module):
    def __init__(self, params, ni, dim, aggr):
        super(MOFNet_FNN, self).__init__()
        
        self.topo_emb = nn.EmbeddingBag(params["topo_num"], params["embed_dim"], padding_idx=params["topo_pad"])
        
        layers = [] 
        layers.append(nn.Linear(ni + params["embed_dim"], 2 * dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(2 * dim, dim)) 
        layers.append(nn.ReLU())
        layers.append(nn.Linear(dim, 1))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, data):
        x = self.topo_emb(data.topo)
        x = torch.flatten(x, start_dim=1)
        
        if self.params["use_struc"]:
            x = torch.cat([x, data.struc], dim=1) 
        
        x = self.layers(x)
        return x