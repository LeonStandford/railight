from __future__ import annotations
import gc
import json
import random
import shutil
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from ultralytics.cfg import CFG_INT_KEYS, get_cfg, get_save_dir
from ultralytics.utils import (
    DEFAULT_CFG,
    LOGGER,
    YAML,
    callbacks,
    colorstr,
    remove_colorstr,
)
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.patches import torch_load
from ultralytics.utils.plotting import plot_tune_results


class Tuner:

    def __init__(self, args=DEFAULT_CFG, _callbacks: dict | None = None):
        self.space = args.pop("space", None) or {
            "lr0": (1e-05, 0.01),
            "lrf": (0.01, 1.0),
            "momentum": (0.7, 0.98, 0.3),
            "weight_decay": (0.0, 0.001),
            "warmup_epochs": (0.0, 5.0),
            "warmup_momentum": (0.0, 0.95),
            "box": (1.0, 20.0),
            "cls": (0.1, 4.0),
            "cls_pw": (0.0, 1.0),
            "dfl": (0.4, 12.0),
            "hsv_h": (0.0, 0.1),
            "hsv_s": (0.0, 0.9),
            "hsv_v": (0.0, 0.9),
            "degrees": (0.0, 45.0),
            "translate": (0.0, 0.9),
            "scale": (0.0, 0.95),
            "shear": (0.0, 10.0),
            "perspective": (0.0, 0.001),
            "flipud": (0.0, 1.0),
            "fliplr": (0.0, 1.0),
            "bgr": (0.0, 1.0),
            "mosaic": (0.0, 1.0),
            "mixup": (0.0, 1.0),
            "cutmix": (0.0, 1.0),
            "copy_paste": (0.0, 1.0),
            "close_mosaic": (0.0, 10.0),
        }
        mongodb_uri = args.pop("mongodb_uri", None)
        mongodb_db = args.pop("mongodb_db", "ultralytics")
        mongodb_collection = args.pop("mongodb_collection", "tuner_results")
        self.args = get_cfg(overrides=args)
        self.args.exist_ok = self.args.resume
        self.tune_dir = get_save_dir(self.args, name=self.args.name or "tune")
        self.args.name, self.args.exist_ok, self.args.resume = (None, False, False)
        self.tune_file = self.tune_dir / "tune_results.ndjson"
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        self.prefix = colorstr("Tuner: ")
        callbacks.add_integration_callbacks(self)
        self.mongodb = None
        if mongodb_uri:
            self._init_mongodb(mongodb_uri, mongodb_db, mongodb_collection)
        LOGGER.info(
            f"{self.prefix}Initialized Tuner instance with 'tune_dir={self.tune_dir}'\n{self.prefix}💡 Learn about tuning at https://docs.ultralytics.com/guides/hyperparameter-tuning"
        )

    def _connect(self, uri: str = "", max_retries: int = 3):
        check_requirements("pymongo")
        from pymongo import MongoClient
        from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

        for attempt in range(max_retries):
            try:
                client = MongoClient(
                    uri,
                    serverSelectionTimeoutMS=30000,
                    connectTimeoutMS=20000,
                    socketTimeoutMS=40000,
                    retryWrites=True,
                    retryReads=True,
                    maxPoolSize=30,
                    minPoolSize=3,
                    maxIdleTimeMS=60000,
                )
                client.admin.command("ping")
                LOGGER.info(
                    f"{self.prefix}Connected to MongoDB Atlas (attempt {attempt + 1})"
                )
                return client
            except (ConnectionFailure, ServerSelectionTimeoutError):
                if attempt == max_retries - 1:
                    raise
                wait_time = 2**attempt
                LOGGER.warning(
                    f"{self.prefix}MongoDB connection failed (attempt {attempt + 1}), retrying in {wait_time}s..."
                )
                time.sleep(wait_time)

    def _init_mongodb(self, mongodb_uri="", mongodb_db="", mongodb_collection=""):
        self.mongodb = self._connect(mongodb_uri)
        self.collection = self.mongodb[mongodb_db][mongodb_collection]
        self.collection.create_index([("fitness", -1)], background=True)
        LOGGER.info(f"{self.prefix}Using MongoDB Atlas for distributed tuning")

    def _get_mongodb_results(self, n: int = 5) -> list:
        try:
            return list(self.collection.find().sort("fitness", -1).limit(n))
        except Exception:
            return []

    @staticmethod
    def _json_default(x):
        return x.item() if hasattr(x, "item") else str(x)

    def _result_record(
        self,
        iteration: int,
        fitness: float,
        hyperparameters: dict[str, float],
        datasets: dict[str, dict],
        save_dirs: dict[str, str] | None = None,
    ) -> dict:
        result = {
            "iteration": iteration,
            "fitness": round(fitness, 5),
            "hyperparameters": hyperparameters,
            "datasets": datasets,
        }
        if save_dirs:
            result["save_dirs"] = save_dirs
        return result

    def _save_to_mongodb(
        self,
        fitness: float,
        hyperparameters: dict[str, float],
        metrics: dict,
        datasets: dict[str, dict],
        iteration: int,
    ):
        try:
            self.collection.insert_one(
                {
                    "fitness": fitness,
                    "hyperparameters": {
                        k: v.item() if hasattr(v, "item") else v
                        for (k, v) in hyperparameters.items()
                    },
                    "metrics": metrics,
                    "datasets": datasets,
                    "timestamp": datetime.now(),
                    "iteration": iteration,
                }
            )
        except Exception as e:
            LOGGER.warning(f"{self.prefix}MongoDB save failed: {e}")

    def _sync_mongodb_to_file(self):
        try:
            all_results = list(self.collection.find().sort("iteration", 1))
            if not all_results:
                return
            with open(self.tune_file, "w", encoding="utf-8") as f:
                for result in all_results:
                    f.write(
                        json.dumps(
                            self._result_record(
                                result["iteration"],
                                result["fitness"] or 0.0,
                                result.get("hyperparameters", {}),
                                result.get("datasets", {}),
                                result.get("save_dirs"),
                            ),
                            default=self._json_default,
                        )
                        + "\n"
                    )
        except Exception as e:
            LOGGER.warning(f"{self.prefix}MongoDB to NDJSON sync failed: {e}")

    def _load_local_results(self) -> list[dict]:
        if not self.tune_file.exists():
            return []
        with open(self.tune_file, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _local_results_to_array(
        self, results: list[dict], n: int | None = None
    ) -> np.ndarray | None:
        if not results:
            return None
        x = np.array(
            [
                [r.get("fitness", 0.0)]
                + [
                    r.get("hyperparameters", {}).get(k, getattr(self.args, k))
                    for k in self.space
                ]
                for r in results
            ],
            dtype=float,
        )
        if n is None:
            return x
        order = np.argsort(-x[:, 0])
        return x[order][:n]

    def _save_local_result(self, result: dict):
        with open(self.tune_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, default=self._json_default) + "\n")

    @staticmethod
    def _best_metrics(result: dict) -> dict | None:
        datasets = result.get("datasets", {})
        if len(datasets) == 1:
            return next(iter(datasets.values()))
        if len(datasets) > 1:
            return {k: round(v.get("fitness") or 0.0, 5) for (k, v) in datasets.items()}
        return None

    @staticmethod
    def _has_training_metrics(result: dict, require_all: bool = False) -> bool:
        datasets = result.get("datasets", {})
        return bool(datasets) and (
            all(datasets.values()) if require_all else any(datasets.values())
        )

    @classmethod
    def _best_result_index(cls, results: list[dict], fitness: np.ndarray) -> int:
        valid = [
            i for (i, result) in enumerate(results) if cls._has_training_metrics(result)
        ]
        return valid[int(fitness[valid].argmax())] if valid else int(fitness.argmax())

    @staticmethod
    def _dataset_names(data: list) -> list[str]:
        stems = [Path(str(d)).stem for d in data]
        totals, seen = (Counter(stems), Counter())
        names = []
        for stem in stems:
            seen[stem] += 1
            names.append(f"{stem}-{seen[stem]}" if totals[stem] > 1 else stem)
        return names

    @staticmethod
    def _crossover(x: np.ndarray, alpha: float = 0.2, k: int = 9) -> np.ndarray:
        k = min(k, len(x))
        weights = x[:, 0] - x[:, 0].min() + 1e-06
        if not np.isfinite(weights).all() or weights.sum() == 0:
            weights = np.ones_like(weights)
        idxs = random.choices(range(len(x)), weights=weights, k=k)
        parents_mat = np.stack([x[i][1:] for i in idxs], 0)
        lo, hi = (parents_mat.min(0), parents_mat.max(0))
        span = hi - lo
        span = np.where(span == 0, np.random.uniform(0.01, 0.1, span.shape), span)
        return np.random.uniform(lo - alpha * span, hi + alpha * span)

    def _mutate(
        self, n: int = 9, mutation: float = 0.5, sigma: float = 0.2
    ) -> dict[str, float]:
        x = None
        if self.mongodb:
            if results := self._get_mongodb_results(n):
                x = np.array(
                    [
                        [r["fitness"]]
                        + [
                            r["hyperparameters"].get(k, self.args.get(k))
                            for k in self.space.keys()
                        ]
                        for r in results
                    ]
                )
            elif (
                self.collection.name in self.collection.database.list_collection_names()
            ):
                x = np.array(
                    [[0.0] + [getattr(self.args, k) for k in self.space.keys()]]
                )
        if x is None:
            x = self._local_results_to_array(self._load_local_results(), n=n)
        if x is not None:
            rng = np.random.default_rng()
            ng = len(self.space)
            genes = self._crossover(x)
            gains = np.array(
                [v[2] if len(v) == 3 else 1.0 for v in self.space.values()]
            )
            factors = np.ones(ng)
            while np.all(factors == 1):
                mask = rng.random(ng) < mutation
                step = rng.standard_normal(ng) * (sigma * gains)
                factors = np.where(mask, np.exp(step), 1.0).clip(0.25, 4.0)
            hyp = {
                k: float(genes[i] * factors[i])
                for (i, k) in enumerate(self.space.keys())
            }
        else:
            hyp = {k: getattr(self.args, k) for k in self.space.keys()}
        for k, bounds in self.space.items():
            hyp[k] = round(min(max(hyp[k], bounds[0]), bounds[1]), 5)
        if "close_mosaic" in hyp:
            hyp["close_mosaic"] = round(hyp["close_mosaic"])
        if "epochs" in hyp:
            hyp["epochs"] = round(hyp["epochs"])
        return hyp

    def __call__(self, iterations: int = 10, cleanup: bool = True):
        t0 = time.time()
        self.tune_dir.mkdir(parents=True, exist_ok=True)
        (self.tune_dir / "weights").mkdir(parents=True, exist_ok=True)
        best_save_dirs = {}
        n_successful = 0
        if self.mongodb:
            self._sync_mongodb_to_file()
        start = 0
        if self.tune_file.exists():
            start = len(self._load_local_results())
            LOGGER.info(
                f"{self.prefix}Resuming tuning run {self.tune_dir} from iteration {start + 1}..."
            )
        for i in range(start, iterations):
            frac = min(i / 300.0, 1.0)
            sigma_i = 0.2 - 0.1 * frac
            mutated_hyp = self._mutate(sigma=sigma_i)
            LOGGER.info(
                f"{self.prefix}Starting iteration {i + 1}/{iterations} with hyperparameters: {mutated_hyp}"
            )
            train_args = {**vars(self.args), **mutated_hyp}
            data = train_args.pop("data")
            if not isinstance(data, (list, tuple)):
                data = [data]
            dataset_names = self._dataset_names(data)
            save_dir = (
                [get_save_dir(get_cfg(train_args))]
                if len(data) == 1
                else [
                    get_save_dir(get_cfg(train_args), name=name)
                    for name in dataset_names
                ]
            )
            weights_dir = [s / "weights" for s in save_dir]
            metrics = {}
            all_fitness = []
            dataset_metrics = {}
            for j, (d, dataset) in enumerate(zip(data, dataset_names)):
                metrics_i = {}
                try:
                    train_args["data"] = d
                    train_args["save_dir"] = str(save_dir[j])
                    launch = [
                        __import__("sys").executable,
                        "-m",
                        "ultralytics.cfg.__init__",
                    ]
                    cmd = [
                        *launch,
                        "train",
                        *(f"{k}={v}" for (k, v) in train_args.items()),
                    ]
                    subprocess.run(cmd, check=True)
                    ckpt_file = weights_dir[j] / (
                        "best.pt"
                        if (weights_dir[j] / "best.pt").exists()
                        else "last.pt"
                    )
                    metrics_i = torch_load(ckpt_file)["train_metrics"]
                    metrics = metrics_i
                    time.sleep(1)
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as e:
                    LOGGER.error(
                        f"training failure for hyperparameter tuning iteration {i + 1}\n{e}"
                    )
                dataset_metrics[dataset] = metrics_i
                all_fitness.append(metrics_i.get("fitness") or 0.0)
            fitness = sum(all_fitness) / len(all_fitness)
            result = self._result_record(
                i + 1,
                fitness,
                mutated_hyp,
                dataset_metrics,
                {dataset: str(s) for (dataset, s) in zip(dataset_names, save_dir)},
            )
            if self._has_training_metrics(result, require_all=True):
                n_successful += 1
            stop_after_iteration = False
            if self.mongodb:
                self._save_to_mongodb(
                    fitness, mutated_hyp, metrics, dataset_metrics, i + 1
                )
                self._sync_mongodb_to_file()
                total_mongo_iterations = self.collection.count_documents({})
                if total_mongo_iterations >= iterations:
                    stop_after_iteration = True
            else:
                self._save_local_result(result)
            results = self._load_local_results()
            x = self._local_results_to_array(results)
            fitness = x[:, 0]
            best_idx = self._best_result_index(results, fitness)
            best_result = results[best_idx]
            n_attempted = i + 1 - start
            current_best_save_dirs = best_result.get("save_dirs", {})
            best_is_current = best_idx == i
            if best_is_current:
                if cleanup:
                    for s in best_save_dirs.values():
                        if s not in current_best_save_dirs.values():
                            shutil.rmtree(s, ignore_errors=True)
                for dataset, weight_dir in zip(dataset_names, weights_dir):
                    best_weights_dir = (
                        self.tune_dir / "weights"
                        if len(data) == 1
                        else self.tune_dir / "weights" / dataset
                    )
                    best_weights_dir.mkdir(parents=True, exist_ok=True)
                    for ckpt in weight_dir.glob("*.pt"):
                        shutil.copy2(ckpt, best_weights_dir)
                best_save_dirs = current_best_save_dirs
            elif cleanup:
                for s in save_dir:
                    shutil.rmtree(s, ignore_errors=True)
                best_save_dirs = current_best_save_dirs
            plot_tune_results(str(self.tune_file))
            if n_successful == n_attempted:
                status = "complete ✅"
            elif n_successful == 0:
                status = "complete (all failed) ❌"
            else:
                status = f"complete ({n_successful}/{n_attempted} succeeded) ⚠️"
            has_valid_best = self._has_training_metrics(best_result)
            header_lines = [
                f"{self.prefix}{i + 1}/{iterations} iterations {status} ({time.time() - t0:.2f}s)",
                f"{self.prefix}Results saved to {colorstr('bold', self.tune_dir)}",
            ]
            if has_valid_best:
                header_lines.extend(
                    [
                        f"{self.prefix}Best fitness={fitness[best_idx]} observed at iteration {best_idx + 1}",
                        f"{self.prefix}Best fitness metrics are {self._best_metrics(best_result)}",
                        f"{self.prefix}Best fitness model is {(self.tune_dir / 'weights' if len(best_result.get('datasets', {})) == 1 else 'not saved for multi-dataset tuning')}",
                    ]
                )
            header = "\n".join(header_lines)
            LOGGER.info("\n" + header)
            if not has_valid_best:
                LOGGER.error(
                    f"{self.prefix}No iterations produced training metrics; skipping best_hyperparameters.yaml"
                )
            else:
                data = {
                    k: int(v) if k in CFG_INT_KEYS else float(v)
                    for (k, v) in zip(self.space.keys(), x[best_idx, 1:])
                }
                YAML.save(
                    self.tune_dir / "best_hyperparameters.yaml",
                    data=data,
                    header=remove_colorstr(header.replace(self.prefix, "# ")) + "\n",
                )
                YAML.print(self.tune_dir / "best_hyperparameters.yaml")
            if stop_after_iteration:
                LOGGER.info(
                    f"{self.prefix}Target iterations ({iterations}) reached in MongoDB ({total_mongo_iterations}). Stopping."
                )
                break
