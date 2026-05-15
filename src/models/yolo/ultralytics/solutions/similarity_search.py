from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
from ultralytics.data.utils import IMG_FORMATS
from ultralytics.utils import LOGGER, TORCH_VERSION
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.torch_utils import TORCH_2_4, select_device

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class VisualAISearch:

    def __init__(self, **kwargs: Any) -> None:
        assert (
            TORCH_2_4
        ), f"VisualAISearch requires torch>=2.4 (found torch=={TORCH_VERSION})"
        from ultralytics.nn.text_model import build_text_model

        check_requirements("faiss-cpu")
        self.faiss = __import__("faiss")
        self.faiss_index = "faiss.index"
        self.data_path_npy = "paths.npy"
        self.data_dir = Path(kwargs.get("data", "images"))
        self.device = select_device(kwargs.get("device", "cpu"))
        if not self.data_dir.exists():
            from ultralytics.utils import ASSETS_URL

            LOGGER.warning(
                f"{self.data_dir} not found. Downloading images.zip from {ASSETS_URL}/images.zip"
            )
            from ultralytics.utils.downloads import safe_download

            safe_download(url=f"{ASSETS_URL}/images.zip", unzip=True, retry=3)
            self.data_dir = Path("images")
        self.model = build_text_model("clip:ViT-B/32", device=self.device)
        self.index = None
        self.image_paths = []
        self.load_or_build_index()

    def extract_image_feature(self, path: Path) -> np.ndarray:
        return self.model.encode_image(Image.open(path)).detach().cpu().numpy()

    def extract_text_feature(self, text: str) -> np.ndarray:
        return (
            self.model.encode_text(self.model.tokenize([text])).detach().cpu().numpy()
        )

    def load_or_build_index(self) -> None:
        if Path(self.faiss_index).exists() and Path(self.data_path_npy).exists():
            LOGGER.info("Loading existing FAISS index...")
            self.index = self.faiss.read_index(self.faiss_index)
            self.image_paths = np.load(self.data_path_npy)
            return
        LOGGER.info("Building FAISS index from images...")
        vectors = []
        for file in self.data_dir.iterdir():
            if file.suffix.lower().lstrip(".") not in IMG_FORMATS:
                continue
            try:
                vectors.append(self.extract_image_feature(file))
                self.image_paths.append(file.name)
            except Exception as e:
                LOGGER.warning(f"Skipping {file.name}: {e}")
        if not vectors:
            raise RuntimeError("No image embeddings could be generated.")
        vectors = np.vstack(vectors).astype("float32")
        self.faiss.normalize_L2(vectors)
        self.index = self.faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.faiss.write_index(self.index, self.faiss_index)
        np.save(self.data_path_npy, np.array(self.image_paths))
        LOGGER.info(f"Indexed {len(self.image_paths)} images.")

    def search(
        self, query: str, k: int = 30, similarity_thresh: float = 0.1
    ) -> list[str]:
        text_feat = self.extract_text_feature(query).astype("float32")
        self.faiss.normalize_L2(text_feat)
        D, index = self.index.search(text_feat, k)
        results = [
            (self.image_paths[i], float(D[0][idx]))
            for (idx, i) in enumerate(index[0])
            if D[0][idx] >= similarity_thresh
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        LOGGER.info("\nRanked Results:")
        for name, score in results:
            LOGGER.info(f"  - {name} | Similarity: {score:.4f}")
        return [r[0] for r in results]

    def __call__(self, query: str) -> list[str]:
        return self.search(query)


class SearchApp:

    def __init__(self, data: str = "images", device: str | None = None) -> None:
        check_requirements("flask>=3.0.1")
        from flask import Flask, render_template, request

        self.render_template = render_template
        self.request = request
        self.searcher = VisualAISearch(data=data, device=device)
        self.app = Flask(
            __name__,
            template_folder="templates",
            static_folder=Path(data).resolve(),
            static_url_path="/images",
        )
        self.app.add_url_rule("/", view_func=self.index, methods=["GET", "POST"])

    def index(self) -> str:
        results = []
        if self.request.method == "POST":
            query = self.request.form.get("query", "").strip()
            results = self.searcher(query)
        return self.render_template("similarity-search.html", results=results)

    def run(self, debug: bool = False) -> None:
        self.app.run(debug=debug)
