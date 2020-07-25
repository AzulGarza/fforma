# ts_transfer packages
conda create --name fforma_fed python=3.6
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fforma_fed

# basic
conda install nomkl
conda install -c anaconda numpy==1.16.1
conda install -c anaconda pandas==0.25.2
conda install -c anaconda scikit-learn==0.23.1
conda install -c anaconda scipy==1.4.1

#Forecast
conda install -c conda-forge statsmodels==0.11.1
conda install -c r rpy2==3.3.3
conda install -c conda-forge lightgbm==2.3.1
conda install -c conda-forge supersmoother
conda install -c bashtage arch
pip install stldecompose

# Pytorch
conda install pytorch=1.3.1 -c pytorch

# ESRNN
pip install ESRNN==0.1.2

#Other
conda install -c anaconda patsy==0.5.1
pip install --user rstl
pip install threadpoolctl # este porque?
pip install seaborn
pip install xgboost==0.90
pip install s3fs
pip install boto3
pip instal -U pandas
pip install git+https://github.com/FedericoGarza/tsfeatures
conda install dask
#conda install -c conda-forge pathos

# visualization
conda install -c conda-forge matplotlib==3.2.1
conda install -c conda-forge tqdm==4.46.0

conda install -c conda-forge jupyterlab
ipython kernel install --user --name=fforma_fed
conda install -c anaconda pylint
conda install -c anaconda pyyaml
conda deactivate

# Para que se usa C?
# conda install -c anaconda kiwisolver==1.2.0
#Cython==0.29.19
#ESRNN==0.1.2
#llvmlite==0.32.1
#numba==0.49.1
#property-cached==1.6.4
