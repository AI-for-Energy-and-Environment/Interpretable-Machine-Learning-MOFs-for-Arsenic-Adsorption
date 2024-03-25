def pre_processing(molecule_string, has_bracket=True):

  if has_bracket:
    atom_list = [atom for atom in molecule_string.replace(" ","").split("'")
                 if atom not in ["[","]",","]]
  else:
    atom_list = molecule_string.split(",")

  return atom_list

class MOF_encode():
    def __init__(self, name, node, linker, topo, prop,  struc=None, fact=None): 
        self.name = name
        self.node = pre_processing(node, False)
        self.linker = [".".join(pre_processing(linker))]
        self.topo = pre_processing(topo, False)
        self.y = prop
        self.struc = struc
        self.fact = fact

def dict_create(item, item_dict):
    item_list = item
    for n in item_list:
        item_dict[n]
    return item_dict

def get_iden(atom) -> Text:

    formal_charge = atom.GetFormalCharge()
    h_num = atom.GetTotalNumHs()

    if atom.GetIsAromatic():
        atom_symbol = atom.GetSymbol().lower()
    else:
        atom_symbol = atom.GetSymbol()

    return f"[{atom_symbol}H{h_num}|{'+' * formal_charge}-{'-' * -formal_charge}]"

def get_atom_symbols(mol: Chem.Mol) -> List[str]:

  symbols = []
  for atom in mol.GetAtoms():
    symbols.append(get_iden(atom))

  return symbols

def get_adjacency(smiles: str, path: str, index: int):

  mol = Chem.MolFromSmiles(smiles)
  adj_matrix = np.array(Chem.GetAdjacencyMatrix(mol))

  matrix_df = pd.DataFrame(adj_matrix)

  atom_symbols = get_atom_symbols(mol)

  matrix_df.columns = atom_symbols
  matrix_df.index = atom_symbols

  matrix_df.to_csv(f"{path}{index}.csv")

  return index + 1

def read_embeddings(path, embed_dict):
    with open(path) as f:
        for row in csv.reader(f):
            embed_dict[row[0]] = row[1:]
    return embed_dict

class MOFDataset():
    def __init__(self, params):
        self.params = params
        self.items = self._csv_reader()
        self._extract_feature()
        self._generate_feature()

    def _csv_reader(self):
        items = []

        data = pd.read_csv(self.params["input_data"])
        lit = list(range(len(data)))
        if self.params["rand_test"]:
            lit_rand = lit.copy()
            for i in range(self.params["rand_cycle"]):
                random.shuffle(lit_rand)
                print(lit_rand[:10])
        else:
            lit_rand = lit.copy()
        for (i, j) in tqdm(zip(lit, lit_rand)):
            name, node, linker, topo = data["name"][i], \
                                       data["metal_smiles"][i], \
                                       data["organ_smiles"][i], \
                                       data["topo"][i],
            prop = data[self.params["props"]].values[j] if self.params["rand_test"] else \
                data[self.params["props"]].values[i]
            struc = data[self.params["struc"]].values[i] if self.params["use_struc"] else None
            fact = data[self.params["fact"]].values[i] if self.params["use_fact"] else None 
            item = MOF_encode(name, node, linker, topo, prop,  struc, fact)
            items.append(item)

        return items

    def _extract_feature(self):
        bond_dic = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 1.5}
        for item in self.items:
            symbols, edges, edge_attrs = [], [], []
            atom_num = 0
            for linker in item.linker:
                symbol, edge, edge_attr = [], [], [] 
                mol = Chem.MolFromSmiles(linker)
                atom_num += len(mol.GetAtoms())
                for atom in mol.GetAtoms():
                    symbol.append(get_iden(atom))
                symbols.append(symbol)
                for bond in mol.GetBonds():
                    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    edge += [[i, j], [j, i]]
                    bond_type = bond.GetBondType()
                    edge_attr += [[bond_dic[str(bond_type)]], [bond_dic[str(bond_type)]]]
                edges.append(torch.tensor(np.transpose(edge), dtype=torch.long))
                edge_attrs.append(edge_attr)
            item.linker_sym = symbols
            item.edge_index = edges
            item.edge_attr = edge_attrs
            item.atom_num = atom_num

    def _generate_feature(self):
        items = self.items
        print(f"MOF number: {len(items)}")

        iden_set, node_set, topo_set = [], [], []
        for item in items:
            for i in list(item.linker_sym):
                iden_set += i
            for i in list(item.node):
                node_set.append(i)
            for i in list(item.topo):
                topo_set.append(i)

        iden_set, node_set, topo_set = list(set(iden_set)), list(set(node_set)), list(set(topo_set))
        iden_set.sort()
        node_set.sort()
        try:
            topo_set.remove("None")
        except:
            pass
        topo_set.sort()
        topo_set += ["None"]
        self.iden_i2c = {i: c for i, c in enumerate(iden_set)}
        self.iden_c2i = {c: i for i, c in enumerate(iden_set)}
        self.node_i2c = {i: c for i, c in enumerate(node_set)}
        self.node_c2i = {c: i for i, c in enumerate(node_set)}
        self.topo_i2c = {i: c for i, c in enumerate(topo_set)}
        self.topo_c2i = {c: i for i, c in enumerate(topo_set)}
        self.topo_pad = self.topo_c2i["None"]

    def n2v_embedding(self, adj_gen_state=False, n2v_emb_state=False):
        if adj_gen_state is True:
            smiles = set()
            for item in tqdm(self.items):
                smiles.update(item.linker)
            smiles = list(smiles)

            import shutil, time
            try:
                shutil.rmtree(self.params["store_matrix_path"])
            except:
                pass
            time.sleep(3)
            try:
                os.makedirs(self.params["store_matrix_path"])
            except:
                pass
            print("Generating adjacency for molecules...")
            l_1 = 0
            for smile in tqdm(smiles):
                l_1 = get_adjacency(smile, self.params["store_matrix_path"], l_1)

    def assig_feature(self):
        for item in self.items:
            linker_features = []
            for linker_fea in item.linker_sym:
                linker_feature = []
                for a in linker_fea:
                    if not self.params["use_n2v_emb"]:
                        linker_feature.append(self.iden_c2i[a])
                linker_features.append(torch.tensor(linker_feature, dtype=torch.float))
            item.x = linker_features

            node_features = []
            for node in item.node:
                node_features.append(self.node_c2i[node])
            item.node = node_features

            topo_features = []
            for topo in item.topo:
                topo_features.append(self.topo_c2i[topo])
            item.topo = topo_features

    def data_load(self):
        items = self.items
        data_load_er = []
        for item in items:
            if not self.params["use_n2v_emb"]:
                if not self.params["use_fact"]: 
                    data = Data(x=torch.tensor(item.x[0], dtype=torch.long),
                                edge_index=item.edge_index[0],
                                node=torch.tensor(item.node, dtype=torch.long),
                                topo=torch.tensor(item.topo, dtype=torch.long),
                                edge_attr=torch.tensor(item.edge_attr[0], dtype=torch.float32),
                                y=item.y,
                                struc=item.struc,
                                fact=item.fact,  
                                name=item.name,
                                atom_num=item.atom_num,
                                pos=None)
                else:
                    data = Data(x=torch.tensor(item.x[0], dtype=torch.long),
                                edge_index=item.edge_index[0],
                                node=torch.tensor(item.node, dtype=torch.long),
                                topo=torch.tensor(item.topo, dtype=torch.long),
                                edge_attr=torch.tensor(item.edge_attr[0], dtype=torch.float32),
                                y=item.y,
                                struc=item.struc,
                                fact=item.fact,  
                                name=item.name,
                                atom_num=item.atom_num,
                                pos=None)
            else:
                data = Data(x=item.x[0],
                            edge_index=item.edge_index[0],
                            node=torch.tensor(item.node, dtype=torch.long),
                            topo=torch.tensor(item.topo, dtype=torch.long),
                            edge_attr=torch.tensor(item.edge_attr[0], dtype=torch.float32),
                            y=item.y,
                            struc=item.struc,
                            name=item.name,
                            atom_num=item.atom_num,
                            pos=None)
            data_load_er.append(data)
        return data_load_er