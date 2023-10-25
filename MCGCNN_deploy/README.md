# This model has been tested on pytorch docker(see Dockerfile)

# How to Run
'python predict.py --config {config_toml_file_path}'

# configs
1. device - the machine to run this model: cpu or cuda

2. cif_filepath for charged(deintercalated) structure

3. cif_filepath for discharged(intercalated) structure


## Data Source
https://next-gen.materialsproject.org/batteries
-> Voltage Pair Properties
