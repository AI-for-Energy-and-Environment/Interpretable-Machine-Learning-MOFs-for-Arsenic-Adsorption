class EarlyStopping():
    def __init__(self, patience=5, min_delta=0): 
        self.patience = patience 
        self.min_delta = min_delta 
        self.counter = 0 
        self.best_loss = None 
        self.early_stop = False 
        self.update = False 

    def __call__(self, val_loss): 
        if self.best_loss == None:
            self.best_loss = val_loss
            self.update = True 
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.update = True 
        elif self.best_loss - val_loss < self.min_delta:
            self.update = False
            self.counter += 1
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True 

def y_scaling(dataset): 
    items = dataset.items
    y = torch.Tensor([item.y for item in items]) 
    scaler = StandardScaler()
    scaler.fit(y) 
    print(scaler.mean_, scaler.var_) 
    for item in items:
        item.y = scaler.transform(item.y.reshape(1, -1))[0] 
    return dataset, scaler 

def struc_scaling(dataset):
    items = dataset.items
    y = torch.Tensor([item.struc for item in items]) 
    scaler = StandardScaler()
    scaler.fit(y) 
    for item in items:
        item.struc = scaler.transform(item.struc.reshape(1, -1))[0] 
    return dataset, scaler 

def convert_to_tensor(data):
    return torch.tensor(data, dtype=torch.float32)

def items2data(items):
    names, mx, me, mn, mt = zip(*[(item.name, item.x, item.edge_index, item.node, item.topo) for item in items])
    y = [item.y for item in items]
    return names, mx, me, mn, mt, convert_to_tensor(y)