[![Build](https://github.com/FedericoGarza/fforma/workflows/Python%20package/badge.svg?branch=master)](https://github.com/FedericoGarza/fforma/tree/master)
[![PyPI version fury.io](https://badge.fury.io/py/fforma.svg)](https://pypi.python.org/pypi/fforma/)
[![Downloads](https://pepy.tech/badge/fforma)](https://pepy.tech/project/fforma)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360+/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/FedericoGarza/fforma/blob/master/LICENSE)

# Installation
```source
pip install git+https://github.com/FedericoGarza/tsfeatures
```

# Download Data
```source
mkdir data/
mkdir data/experiment/
cd data /experiment/ 
wget https://github.com/FedericoGarza/meta-data/releases/download/v.0.0.1/M4_pickle.p # curl -O
wget https://github.com/FedericoGarza/meta-data/releases/download/v.0.0.1/M3_pickle.p
wget https://github.com/FedericoGarza/meta-data/releases/download/v.0.0.1/TOURISM_pickle.p
```

# Usage
See `comparison-fforma-r.ipynb` for an example using the original data.
PYTHONPATH=. python src/experiment.py --dataset 'M4' --start_id 1 --end_id 2 --generate_grid 0 --gpu_id 3 --upload 1


# Current Results

| DATASET   | OUR OWA | OUR OWA  (W OUR FEATS) | M4 OWA (Hyndman et.al.) |
|-----------|:-------:|:---------------------:|:------------------------:|
| Yearly    | 0.802   | 0.818                 | 0.799  |
| Quarterly | 0.849   | 0.857                 | 0.847  |
| Monthly   | 0.860   | 0.877                 | 0.858  |
| Hourly    | 0.510   | 0.489                 | 0.914  |
| Weekly    | 0.887   | 0.884                 | 0.914  |  
| Daily     | 0.977   | 0.977                 | 0.914  |
