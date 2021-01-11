import logging
import socket
from collections import defaultdict
from functools import partial
from threading import Thread

import dask.dataframe as dd
import dask_xgboost as dxgb
import numpy as np
import xgboost as xgb
from dask import delayed
from dask.distributed import Client, wait
from dask_xgboost.core import (
    _has_dask_collections,
    _package_evals,
    concat,
    parse_host_port,
)
from dask_xgboost.tracker import get_host_ip, RabitTracker
from toolz import assoc, first
from tornado import gen

logger = logging.getLogger(__name__)


def start_tracker(host, n_workers, default_host=None):
    """ Start Rabit tracker """
    if host is None:
        try:
            host = get_host_ip("auto")
        except socket.gaierror:
            if default_host is not None:
                host = default_host
            else:
                raise

    env = {"DMLC_NUM_WORKER": n_workers}
    rabit = RabitTracker(hostIP=host, nslave=n_workers)
    env.update(rabit.slave_envs())

    rabit.start(n_workers)
    logger.info("Starting Rabit Tracker")
    thread = Thread(target=rabit.join)
    thread.daemon = True
    thread.start()
    return env


def train_part(
    env,
    param,
    list_of_parts,
    dmatrix_kwargs=None,
    eval_set=None,
    missing=None,
    n_jobs=None,
    sample_weight_eval_set=None,
    **kwargs
):
    """
    Run part of XGBoost distributed workload

    This starts an xgboost.rabit slave, trains on provided data, and then shuts
    down the xgboost.rabit slave

    Returns
    -------
    model if rank zero, None otherwise
    """
    data, labels, sample_weight, errs = zip(*list_of_parts)  # Prepare data
    data = concat(data)  # Concatenate many parts into one
    labels = concat(labels)
    errs = concat(errs)
    sample_weight = concat(sample_weight) if np.all(sample_weight) else None

    if dmatrix_kwargs is None:
        dmatrix_kwargs = {}

    dmatrix_kwargs["feature_names"] = getattr(data, "columns", None)
    dtrain = xgb.DMatrix(data, labels, weight=sample_weight, **dmatrix_kwargs)
    fforma_obj = partial(fobj, errs=errs)
    fforma_eval = partial(feval, errs=errs)

    evals = _package_evals(
        eval_set,
        sample_weight_eval_set=sample_weight_eval_set,
        missing=missing,
        n_jobs=n_jobs,
    )

    args = [("%s=%s" % item).encode() for item in env.items()]
    xgb.rabit.init(args)
    try:
        local_history = {}
        logger.info("Starting Rabit, Rank %d", xgb.rabit.get_rank())
        bst = xgb.train(
            param, dtrain, evals_result=local_history, **kwargs,
            obj=fforma_obj, feval=fforma_eval,
        )

        if xgb.rabit.get_rank() == 0:  # Only return from one worker
            result = bst
            evals_result = local_history
        else:
            result = None
            evals_result = None
    finally:
        logger.info("Finalizing Rabit, Rank %d", xgb.rabit.get_rank())
        xgb.rabit.finalize()
    return result, evals_result

@gen.coroutine
def _train(
    client,
    params,
    data,
    labels,
    errs,
    dmatrix_kwargs={},
    evals_result=None,
    sample_weight=None,
    **kwargs
):
    """
    Asynchronous version of train

    See Also
    --------
    train
    """
    # Break apart Dask.array/dataframe into chunks/parts
    data_parts = data.to_delayed()
    label_parts = labels.to_delayed()
    errs_parts = errs.to_delayed()
    if isinstance(data_parts, np.ndarray):
        assert data_parts.shape[1] == 1
        data_parts = data_parts.flatten().tolist()
    if isinstance(label_parts, np.ndarray):
        assert label_parts.ndim == 1 or label_parts.shape[1] == 1
        label_parts = label_parts.flatten().tolist()
    if isinstance(errs_parts, np.ndarray):
        errs_parts = errs_parts.flatten().tolist()
    if sample_weight is not None:
        sample_weight_parts = sample_weight.to_delayed()
        if isinstance(sample_weight_parts, np.ndarray):
            assert sample_weight_parts.ndim == 1 or sample_weight_parts.shape[1] == 1
            sample_weight_parts = sample_weight_parts.flatten().tolist()
    else:
        # If sample_weight is None construct a list of Nones to keep
        # the structure of parts consistent.
        sample_weight_parts = [None] * len(data_parts)

    # Check that data, labels, and sample_weights are the same length
    lists = [data_parts, label_parts, sample_weight_parts]
    if len(set([len(l) for l in lists])) > 1:
        raise ValueError(
            "data, label, and sample_weight parts/chunks must have same length."
        )

    # Arrange parts into triads.  This enforces co-locality
    parts = list(map(delayed, zip(data_parts, label_parts, sample_weight_parts, errs_parts)))
    parts = client.compute(parts)  # Start computation in the background
    yield wait(parts)

    for part in parts:
        if part.status == "error":
            yield part  # trigger error locally

    _has_dask_collections(
        kwargs.get("eval_set", []), "Evaluation set must not contain dask collections."
    )
    _has_dask_collections(
        kwargs.get("sample_weight_eval_set", []),
        "Sample weight evaluation set must not contain dask collections.",
    )

    # Because XGBoost-python doesn't yet allow iterative training, we need to
    # find the locations of all chunks and map them to particular Dask workers
    key_to_part_dict = dict([(part.key, part) for part in parts])
    who_has = yield client.scheduler.who_has(keys=[part.key for part in parts])
    worker_map = defaultdict(list)
    for key, workers in who_has.items():
        worker_map[first(workers)].append(key_to_part_dict[key])

    ncores = yield client.scheduler.ncores()  # Number of cores per worker

    default_host, _ = parse_host_port(client.scheduler.address)
    # Start the XGBoost tracker on the Dask scheduler
    env = yield client._run_on_scheduler(
        start_tracker, None, len(worker_map), default_host=default_host
    )

    # Tell each worker to train on the chunks/parts that it has locally
    futures = [
        client.submit(
            train_part,
            env,
            assoc(params, "nthread", ncores[worker]),
            list_of_parts,
            workers=worker,
            dmatrix_kwargs=dmatrix_kwargs,
            **kwargs
        )
        for worker, list_of_parts in worker_map.items()
    ]

    # Get the results, only one will be non-None
    results = yield client._gather(futures)
    result, _evals_result = [v for v in results if v.count(None) != len(v)][0]

    if evals_result is not None:
        evals_result.update(_evals_result)

    num_class = params.get("num_class")
    if num_class:
        result.set_attr(num_class=str(num_class))
    raise gen.Return(result)

def train(
    client,
    params,
    data,
    labels,
    errs,
    dmatrix_kwargs={},
    evals_result=None,
    sample_weight=None,
    **kwargs
):
    """ Train an XGBoost model on a Dask Cluster

    This starts XGBoost on all Dask workers, moves input data to those workers,
    and then calls ``xgboost.train`` on the inputs.

    Parameters
    ----------
    client: dask.distributed.Client
    params: dict
        Parameters to give to XGBoost (see xgb.Booster.train)
    data: dask array or dask.dataframe
    labels: dask.array or dask.dataframe
    dmatrix_kwargs: Keywords to give to Xgboost DMatrix
    evals_result: dict, optional
        Stores the evaluation result history of all the items in the eval_set
        by mutating evals_result in place.
    sample_weight : array_like, optional
        instance weights
    **kwargs: Keywords to give to XGBoost train

    Examples
    --------
    >>> client = Client('scheduler-address:8786')  # doctest: +SKIP
    >>> data = dd.read_csv('s3://...')  # doctest: +SKIP
    >>> labels = data['outcome']  # doctest: +SKIP
    >>> del data['outcome']  # doctest: +SKIP
    >>> train(client, params, data, labels, **normal_kwargs)  # doctest: +SKIP
    <xgboost.core.Booster object at ...>

    See Also
    --------
    predict
    """
    return client.sync(
        _train,
        client,
        params,
        data,
        labels,
        errs,
        dmatrix_kwargs,
        evals_result,
        sample_weight,
        **kwargs
    )


def fobj(predt: np.ndarray, dtrain: xgb.DMatrix, errs):
    """
    """
    y = dtrain.get_label().astype(int)
    n_train = len(y)
    # predt in softmax
    preds = np.reshape(predt,
                       errs[y, :].shape)
    weighted_avg_loss_func = (preds * errs[y, :]).sum(axis=1).reshape((n_train, 1))

    grad = preds * (errs[y, :] - weighted_avg_loss_func)
    hess = errs[y,:] * preds * (1.0 - preds) - grad * preds

    return grad.flatten(), hess.flatten()

def feval(predt: np.ndarray, dtrain: xgb.DMatrix, errs):
    """
    """
    y = dtrain.get_label().astype(int)
    preds = np.reshape(predt,
                       errs[y, :].shape)

    weighted_avg_loss_func = (preds * errs[y, :]).sum(axis=1)
    fforma_loss = weighted_avg_loss_func.mean()

    return 'FFORMA-loss', fforma_loss

class DistributedFFORMA:

    def __init__(self, params={}, random_seed=0, n_estimators=10):
        init_params = {
            'objective': 'multi:softprob',
            'seed': random_seed,
            'disable_default_eval_metric': 1
        }
        self.num_round = n_estimators
        self.params = {**params, **init_params}

    def fit(self, client: Client, features: dd.DataFrame, losses: dd.DataFrame):
        errors = losses.to_dask_array(lengths=True)
        labels = errors.map_blocks(lambda x: np.arange(x.shape[0])).persist()
        self.params['num_class'] = errors.shape[1]
        self.gbm_model_ = train(client, self.params,
                                features,
                                labels,
                                errors,
                                num_boost_round=self.num_round)
        weights = dxgb.predict(client, self.gbm_model_, features)
        weights = dd.from_array(weights, columns=losses.columns)
        weights.index = features.index
        self.weights_ = weights
        return self

    def predict(self, client: Client, base_predictions: dd.DataFrame):
        fforma_predictions = (self.weights_ * base_predictions).sum(1)
        fforma_predictions = fforma_predictions.rename('fforma_prediction')
        all_predictions = dd.concat([base_predictions, fforma_predictions], axis=1)
        return all_predictions
