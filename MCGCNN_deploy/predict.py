import argparse
import torch
from mcgcnn.data import Paired_CIFData, collate_pool_for_paired
import tomli

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="./predict_default.toml")
args = parser.parse_args()

with open(args.config, "rb") as f:
    config_dict = tomli.load(f)

collate_fn = collate_pool_for_paired(device=config_dict["device"])
datalist = [
    [
        config_dict["charged_data"],
        config_dict["discharged_data"],
    ]
]
dummytarget_value = 0
datalist_with_dummytarget = [i + [dummytarget_value] for i in datalist]
dataset = Paired_CIFData(
    id_prop_data=datalist_with_dummytarget,
    data_dir="./",
    atom_init_file="./mcgcnn/atom_init.json",
    max_num_nbr=12,
    radius=8,
    dmin=0,
    step=0.2,
    is_cif_preload=False,
)
model = torch.jit.load("./inference_model/model.pt", map_location=config_dict["device"])
model.eval()
with torch.no_grad():
    for i, v in enumerate(dataset):
        input, target, batch_cif_ids = collate_fn([v])

        output = model(*input).data.detach().cpu().numpy().item()
        print(output)
