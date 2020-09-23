conda env create --name ensambler --file environment.yml
conda activate ensambler

ipython kernel install --user --name=ensambler
conda deactivate
