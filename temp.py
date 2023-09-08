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