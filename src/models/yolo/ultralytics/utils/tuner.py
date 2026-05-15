from __future__ import annotations
import numpy as np
from ultralytics.cfg import TASK2DATA, TASK2METRIC, get_cfg, get_save_dir
from ultralytics.utils import (
    DEFAULT_CFG,
    DEFAULT_CFG_DICT,
    LOGGER,
    NUM_THREADS,
    checks,
    colorstr,
)

RAY_SEARCH_ALG_REQUIREMENTS = {
    "random": None,
    "ax": "ax-platform",
    "bayesopt": "bayesian-optimization==1.4.3",
    "bohb": ["hpbandster", "ConfigSpace"],
    "hebo": "HEBO>=0.2.0",
    "hyperopt": "hyperopt",
    "nevergrad": "nevergrad",
    "optuna": "optuna",
    "zoopt": "zoopt",
}


def _sanitize_tune_value(value: dict):
    if isinstance(value, dict):
        return {k: _sanitize_tune_value(v) for (k, v) in value.items()}
    if isinstance(value, list):
        return [_sanitize_tune_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple((_sanitize_tune_value(v) for v in value))
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _get_ray_search_alg_kind(search_alg):
    if search_alg is None:
        return None
    if isinstance(search_alg, str):
        normalized = search_alg.strip().lower()
        return normalized or None
    cls = search_alg.__class__
    module, name = (cls.__module__, cls.__name__)
    if name == "AxSearch" and module.startswith("ray.tune.search.ax"):
        return "ax"
    if name == "TuneBOHB" and module.startswith("ray.tune.search.bohb"):
        return "bohb"
    if name == "ZOOptSearch" and module.startswith("ray.tune.search.zoopt"):
        return "zoopt"
    return None


def _validate_ax_search_space(space):
    checks.check_requirements(RAY_SEARCH_ALG_REQUIREMENTS["ax"])
    from ray.tune.search.ax.ax_search import AxSearch

    return AxSearch.convert_search_space(space)


def _create_ax_search(space, task):
    parameters = _validate_ax_search_space(space)
    from ax.service.ax_client import AxClient
    from ax.service.utils.instantiation import ObjectiveProperties
    from ray.tune.search.ax.ax_search import AxSearch

    ax_client = AxClient()
    ax_client.create_experiment(
        parameters=parameters,
        objectives={TASK2METRIC[task]: ObjectiveProperties(minimize=False)},
    )
    return AxSearch(ax_client=ax_client)


def _convert_bohb_search_space(space):
    checks.check_requirements(RAY_SEARCH_ALG_REQUIREMENTS["bohb"])
    import ConfigSpace
    from ray.tune.search.sample import (
        Categorical,
        Float,
        Integer,
        LogUniform,
        Quantized,
        Uniform,
    )
    from ray.tune.search.variant_generator import parse_spec_vars
    from ray.tune.utils import flatten_dict

    resolved_space = flatten_dict(space, prevent_delimiter=True)
    resolved_vars, domain_vars, grid_vars = parse_spec_vars(resolved_space)
    if grid_vars:
        raise ValueError(
            "Grid search parameters cannot be automatically converted to a TuneBOHB search space."
        )
    cs = ConfigSpace.ConfigurationSpace()
    for path, domain in domain_vars:
        par = "/".join((str(p) for p in path))
        sampler = domain.get_sampler()
        if isinstance(sampler, Quantized):
            raise ValueError(
                "TuneBOHB does not support quantized search spaces with the current ConfigSpace version."
            )
        if isinstance(domain, Float) and isinstance(sampler, (Uniform, LogUniform)):
            cs.add(
                ConfigSpace.UniformFloatHyperparameter(
                    par,
                    lower=domain.lower,
                    upper=domain.upper,
                    log=isinstance(sampler, LogUniform),
                )
            )
        elif isinstance(domain, Integer) and isinstance(sampler, (Uniform, LogUniform)):
            upper = domain.upper - 1
            cs.add(
                ConfigSpace.UniformIntegerHyperparameter(
                    par,
                    lower=domain.lower,
                    upper=upper,
                    log=isinstance(sampler, LogUniform),
                )
            )
        elif isinstance(domain, Categorical) and isinstance(sampler, Uniform):
            cs.add(
                ConfigSpace.CategoricalHyperparameter(par, choices=domain.categories)
            )
        else:
            raise ValueError(
                f"TuneBOHB does not support parameters of type {type(domain).__name__} with sampler type {type(domain.sampler).__name__}."
            )
    fixed_param_space = {
        "/".join((str(p) for p in path)): value for (path, value) in resolved_vars
    }
    return (cs, fixed_param_space)


def _create_bohb_search(space, task):
    cs, fixed_param_space = _convert_bohb_search_space(space)
    from ray.tune.search.bohb.bohb_search import TuneBOHB

    return (TuneBOHB(space=cs, metric=TASK2METRIC[task], mode="max"), fixed_param_space)


def _create_nevergrad_search(task):
    checks.check_requirements(RAY_SEARCH_ALG_REQUIREMENTS["nevergrad"])
    import nevergrad as ng
    from ray.tune.search.nevergrad import NevergradSearch

    return NevergradSearch(
        optimizer=ng.optimizers.OnePlusOne, metric=TASK2METRIC[task], mode="max"
    )


def _convert_zoopt_search_space(space):
    checks.check_requirements(RAY_SEARCH_ALG_REQUIREMENTS["zoopt"])
    from ray.tune.search.variant_generator import parse_spec_vars
    from ray.tune.search.zoopt import ZOOptSearch
    from ray.tune.utils import flatten_dict

    resolved_space = flatten_dict(space, prevent_delimiter=True)
    resolved_vars, _, _ = parse_spec_vars(resolved_space)
    fixed_param_space = {
        "/".join((str(p) for p in path)): value for (path, value) in resolved_vars
    }
    dim_dict = ZOOptSearch.convert_search_space(space)
    return (dim_dict, fixed_param_space)


def _create_zoopt_search(space, task, iterations):
    dim_dict, fixed_param_space = _convert_zoopt_search_space(space)
    from ray.tune.search.zoopt import ZOOptSearch

    return (
        ZOOptSearch(
            algo="asracos",
            budget=iterations,
            dim_dict=dim_dict,
            metric=TASK2METRIC[task],
            mode="max",
        ),
        fixed_param_space,
    )


def _resolve_ray_search_alg(search_alg, task, space, iterations):
    if search_alg is None:
        return (None, space, None)
    normalized = _get_ray_search_alg_kind(search_alg)
    if isinstance(search_alg, str):
        if not normalized:
            return (None, space, None)
        if normalized not in RAY_SEARCH_ALG_REQUIREMENTS:
            supported = ", ".join(sorted(RAY_SEARCH_ALG_REQUIREMENTS))
            raise ValueError(
                f"Unsupported Ray Tune search_alg '{search_alg}'. Supported values: {supported}."
            )
        if normalized == "random":
            return (None, space, normalized)
    try:
        if normalized == "ax":
            if isinstance(search_alg, str):
                return (_create_ax_search(space, task), {}, normalized)
            _validate_ax_search_space(space)
            return (search_alg, {}, normalized)
        if normalized == "bohb":
            if isinstance(search_alg, str):
                resolved_search_alg, tuner_param_space = _create_bohb_search(
                    space, task
                )
            else:
                _, tuner_param_space = _convert_bohb_search_space(space)
                resolved_search_alg = search_alg
            return (resolved_search_alg, tuner_param_space, normalized)
        if normalized == "nevergrad":
            return (_create_nevergrad_search(task), space, normalized)
        if normalized == "zoopt":
            if isinstance(search_alg, str):
                resolved_search_alg, tuner_param_space = _create_zoopt_search(
                    space, task, iterations
                )
            else:
                _, tuner_param_space = _convert_zoopt_search_space(space)
                resolved_search_alg = search_alg
            return (resolved_search_alg, tuner_param_space, normalized)
        if not isinstance(search_alg, str):
            return (search_alg, space, None)
        requirements = RAY_SEARCH_ALG_REQUIREMENTS[normalized]
        if requirements:
            checks.check_requirements(requirements)
        from ray.tune.search import create_searcher

        return (
            create_searcher(normalized, metric=TASK2METRIC[task], mode="max"),
            space,
            normalized,
        )
    except (ImportError, ModuleNotFoundError) as e:
        raise ModuleNotFoundError(
            f"Ray Tune search_alg '{search_alg}' requires additional dependencies. Original error: {e}"
        ) from e


def run_ray_tune(
    model,
    space: dict | None = None,
    grace_period: int = 10,
    gpu_per_trial: int | None = None,
    iterations: int = 10,
    search_alg=None,
    **train_args,
):
    LOGGER.info(
        "💡 Learn about RayTune at https://docs.ultralytics.com/integrations/ray-tune"
    )
    try:
        checks.check_requirements("ray[tune]")
        import ray
        from ray import tune
        from ray.tune import RunConfig
        from ray.tune.schedulers import ASHAScheduler, HyperBandForBOHB
    except ImportError:
        raise ModuleNotFoundError(
            'Ray Tune required but not found. To install run: pip install "ray[tune]"'
        )
    try:
        import wandb

        assert hasattr(wandb, "__version__")
    except (ImportError, AssertionError):
        wandb = False
    checks.check_version(ray.__version__, ">=2.0.0", "ray")
    default_space = {
        "lr0": tune.uniform(1e-05, 0.01),
        "lrf": tune.uniform(0.01, 1.0),
        "momentum": tune.uniform(0.7, 0.98),
        "weight_decay": tune.uniform(0.0, 0.001),
        "warmup_epochs": tune.uniform(0.0, 5.0),
        "warmup_momentum": tune.uniform(0.0, 0.95),
        "box": tune.uniform(1.0, 20.0),
        "cls": tune.uniform(0.1, 4.0),
        "cls_pw": tune.uniform(0.0, 1.0),
        "dfl": tune.uniform(0.4, 12.0),
        "hsv_h": tune.uniform(0.0, 0.1),
        "hsv_s": tune.uniform(0.0, 0.9),
        "hsv_v": tune.uniform(0.0, 0.9),
        "degrees": tune.uniform(0.0, 45.0),
        "translate": tune.uniform(0.0, 0.9),
        "scale": tune.uniform(0.0, 0.95),
        "shear": tune.uniform(0.0, 10.0),
        "perspective": tune.uniform(0.0, 0.001),
        "flipud": tune.uniform(0.0, 1.0),
        "fliplr": tune.uniform(0.0, 1.0),
        "bgr": tune.uniform(0.0, 1.0),
        "mosaic": tune.uniform(0.0, 1.0),
        "mixup": tune.uniform(0.0, 1.0),
        "cutmix": tune.uniform(0.0, 1.0),
        "copy_paste": tune.uniform(0.0, 1.0),
        "close_mosaic": tune.randint(0, 11),
    }
    task = model.task
    model_in_store = ray.put(model)
    base_name = train_args.get("name", "tune")

    def _tune(config):
        model_to_train = ray.get(model_in_store)
        model_to_train.trainer = None
        model_to_train.reset_callbacks()
        config = _sanitize_tune_value(dict(config))
        config.update(train_args)
        try:
            trial_id = tune.get_trial_id()
            trial_suffix = trial_id.split("_")[-1] if "_" in trial_id else trial_id
            config["name"] = f"{base_name}_{trial_suffix}"
        except Exception:
            config["name"] = base_name
        results = model_to_train.train(**config)
        return results.results_dict

    if not space and (not train_args.get("resume")):
        space = default_space
        LOGGER.warning("Search space not provided, using default search space.")
    data = train_args.get("data", TASK2DATA[task])
    space["data"] = data
    if "data" not in train_args:
        LOGGER.warning(f'Data not provided, using default "data={data}".')
    resolved_search_alg, tuner_param_space, resolved_search_alg_kind = (
        _resolve_ray_search_alg(search_alg, task, space, iterations)
    )
    trainable_with_resources = tune.with_resources(
        _tune, {"cpu": NUM_THREADS, "gpu": gpu_per_trial or 0}
    )
    max_t = train_args.get("epochs") or DEFAULT_CFG_DICT["epochs"] or 100
    scheduler = ASHAScheduler(
        time_attr="epoch",
        metric=TASK2METRIC[task],
        mode="max",
        max_t=max_t,
        grace_period=min(grace_period, max_t),
        reduction_factor=3,
    )
    if resolved_search_alg_kind == "bohb":
        scheduler = HyperBandForBOHB(
            time_attr="epoch",
            metric=TASK2METRIC[task],
            mode="max",
            max_t=max_t,
            reduction_factor=3,
        )
    tune_dir = get_save_dir(
        get_cfg(
            DEFAULT_CFG, {**train_args, **{"exist_ok": train_args.pop("resume", False)}}
        ),
        name=train_args.pop("name", "tune"),
    )
    tune_dir.mkdir(parents=True, exist_ok=True)
    if tune.Tuner.can_restore(tune_dir):
        LOGGER.info(f"{colorstr('Tuner: ')} Resuming tuning run {tune_dir}...")
        tuner = tune.Tuner.restore(
            str(tune_dir), trainable=trainable_with_resources, resume_errored=True
        )
    else:
        tuner = tune.Tuner(
            trainable_with_resources,
            param_space=tuner_param_space,
            tune_config=tune.TuneConfig(
                search_alg=resolved_search_alg,
                scheduler=scheduler,
                num_samples=iterations,
                trial_name_creator=lambda trial: f"{trial.trainable_name}_{trial.trial_id}",
                trial_dirname_creator=lambda trial: f"{trial.trainable_name}_{trial.trial_id}",
            ),
            run_config=RunConfig(storage_path=tune_dir.parent, name=tune_dir.name),
        )
    tuner.fit()
    results = tuner.get_results()
    ray.shutdown()
    return results
