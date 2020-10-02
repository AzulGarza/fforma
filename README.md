[![Build](https://github.com/FedericoGarza/fforma/workflows/Python%20package/badge.svg?branch=master)](https://github.com/FedericoGarza/fforma/tree/master)
[![PyPI version fury.io](https://badge.fury.io/py/fforma.svg)](https://pypi.python.org/pypi/fforma/)
[![Downloads](https://pepy.tech/badge/fforma)](https://pepy.tech/project/fforma)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360+/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/FedericoGarza/fforma/blob/master/LICENSE)

# Experiments

1. Create docker image and experiments directory.
```source
make init
```
1. Download datasets.
```source
make datasets
```

1. Create base data (models to ensemble).
```source
make base dataset=tourism
```

1. Compute ensemble benchmarks.
```source
make benchmarks dataset=tourism
```

1. Run experiments.
```source
make run dataset=tourism model=ffnn splits=3 trials=200
```
