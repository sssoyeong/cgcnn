import pandas as pd
import json

inputs = pd.read_csv('data/sample-regression/id_prop.csv', header=None)
result = pd.read_csv('data/sample-regression/test_results.csv', header=None)
print(f' INPUT\n {inputs.head(5)}')
print(f'RESULT\n {result.head(5)}')

with open('data/sample-regression//atom_init.json') as json_file:
    atom_init = json.load(json_file)
print(atom_init)

