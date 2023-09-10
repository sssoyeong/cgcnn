from torch.utils.data import DataLoader
from cgcnn.data import collate_pool
from cgcnn.data import CIFData

batch_size = 256
n_workers = 0
collate_fn = collate_pool
dataset = CIFData('data/sample-inference')
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=n_workers, collate_fn=collate_fn,
                            pin_memory=args.cuda)
print(DataLoader)


dataset_list = CIFData("data/sample-inference")
print(dir(dataset_list))
print(dataset_list.id_prop_data)
len(dataset_list.id_prop_data)
dataset_list.id_prop_data[1][0]

for i in range(len(dataset_list.id_prop_data)):
    if dataset_list.id_prop_data[i][0].startswith("\ufeff"):
        print(i, " before :  ", dataset_list.id_prop_data[i][0])
        dataset_list.id_prop_data[i][0] = id_prop_data[i][0].replace("\ufeff", "")
        print(i, " after  :  ", dataset_list.id_prop_data[i][0])

cif_id_list = []
for i, ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id) in enumerate(dataset_list):
    cif_id_list.append(cif_id)

    n_i = atom_fea.shape[0]  # number of atoms for this crystal
    batch_atom_fea.append(atom_fea)
    batch_nbr_fea.append(nbr_fea)
    batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
    new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
    crystal_atom_idx.append(new_idx)
    batch_target.append(target)
    batch_cif_ids.append(cif_id)
    base_idx += n_i