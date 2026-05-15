from __future__ import annotations
from functools import partial
from pathlib import Path
import numpy as np
import torch
from ultralytics.nn.modules import Detect, Pose, Pose26
from ultralytics.utils import LINUX, LOGGER, MACOS
from ultralytics.utils.checks import (
    check_apt_requirements,
    check_requirements,
    check_version,
    is_sudo_available,
)
from ultralytics.utils.downloads import attempt_download_asset
from ultralytics.utils.files import spaces_in_path
from ultralytics.utils.tal import make_anchors


def tf_wrapper(model: torch.nn.Module) -> torch.nn.Module:
    for m in model.modules():
        if not isinstance(m, Detect):
            continue
        import types

        m._get_decode_boxes = types.MethodType(_tf_decode_boxes, m)
        if isinstance(m, Pose):
            m.kpts_decode = types.MethodType(
                partial(_tf_kpts_decode, is_pose26=type(m) is Pose26), m
            )
    return model


def _tf_decode_boxes(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
    shape = x["feats"][0].shape
    boxes = x["boxes"]
    if self.format != "imx" and (self.dynamic or self.shape != shape):
        self.anchors, self.strides = (
            a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5)
        )
        self.shape = shape
    grid_h, grid_w = shape[2:4]
    grid_size = torch.tensor(
        [grid_w, grid_h, grid_w, grid_h], device=boxes.device
    ).reshape(1, 4, 1)
    norm = self.strides / (self.stride[0] * grid_size)
    dbox = self.decode_bboxes(
        self.dfl(boxes) * norm, self.anchors.unsqueeze(0) * norm[:, :2]
    )
    return dbox


def _tf_kpts_decode(self, kpts: torch.Tensor, is_pose26: bool = False) -> torch.Tensor:
    ndim = self.kpt_shape[1]
    bs = kpts.shape[0]
    y = kpts.view(bs, *self.kpt_shape, -1)
    grid_h, grid_w = self.shape[2:4]
    grid_size = torch.tensor([grid_w, grid_h], device=y.device).reshape(1, 2, 1)
    norm = self.strides / (self.stride[0] * grid_size)
    a = (
        y[:, :, :2] + self.anchors
        if is_pose26
        else y[:, :, :2] * 2.0 + (self.anchors - 0.5)
    ) * norm
    if ndim == 3:
        a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
    return a.view(bs, self.nk, -1)


def onnx2saved_model(
    onnx_file: str,
    output_dir: Path | str,
    int8: bool = False,
    images: np.ndarray | None = None,
    disable_group_convolution: bool = False,
    prefix: str = "",
):
    cuda = torch.cuda.is_available()
    try:
        import tensorflow as tf
    except ImportError:
        check_requirements("tensorflow>=2.0.0,<=2.19.0")
        import tensorflow as tf
    check_requirements(
        (
            "tf_keras<=2.19.0",
            "sng4onnx>=1.0.1",
            "onnx_graphsurgeon>=0.3.26",
            "ai-edge-litert>=1.2.0" + (",<1.4.0" if MACOS else ""),
            "onnx>=1.12.0,<2.0.0",
            "onnx2tf>=1.26.3,<1.29.0",
            "onnxslim>=0.1.71",
            "onnxruntime-gpu" if cuda else "onnxruntime",
            "protobuf>=5",
        ),
        cmds="--extra-index-url https://pypi.ngc.nvidia.com",
    )
    LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")
    check_version(
        tf.__version__,
        ">=2.0.0",
        name="tensorflow",
        verbose=True,
        msg="https://github.com/ultralytics/ultralytics/issues/5161",
    )
    output_dir = Path(output_dir)
    onnx2tf_file = Path("calibration_image_sample_data_20x128x128x3_float32.npy")
    if not onnx2tf_file.exists():
        attempt_download_asset(f"{onnx2tf_file}.zip", unzip=True, delete=True)
    np_data = None
    if int8:
        tmp_file = output_dir / "tmp_tflite_int8_calibration_images.npy"
        if images is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            np.save(str(tmp_file), images)
            np_data = [["images", tmp_file, [[[[0, 0, 0]]]], [[[[255, 255, 255]]]]]]
    import onnx.helper

    if not hasattr(onnx.helper, "float32_to_bfloat16"):
        import struct

        def float32_to_bfloat16(fval):
            ival = struct.unpack("=I", struct.pack("=f", fval))[0]
            return ival >> 16

        onnx.helper.float32_to_bfloat16 = float32_to_bfloat16
    import onnx2tf

    LOGGER.info(
        f"{prefix} starting TFLite export with onnx2tf {onnx2tf.__version__}..."
    )
    keras_model = onnx2tf.convert(
        input_onnx_file_path=onnx_file,
        output_folder_path=str(output_dir),
        not_use_onnxsim=True,
        verbosity="error",
        output_integer_quantized_tflite=int8,
        custom_input_op_name_np_data_path=np_data,
        enable_batchmatmul_unfold=not int8,
        output_signaturedefs=True,
        disable_group_convolution=disable_group_convolution,
    )
    if int8:
        tmp_file.unlink(missing_ok=True)
        for file in output_dir.rglob("*_dynamic_range_quant.tflite"):
            file.rename(
                file.with_name(
                    file.stem.replace("_dynamic_range_quant", "_int8") + file.suffix
                )
            )
        for file in output_dir.rglob("*_integer_quant_with_int16_act.tflite"):
            file.unlink()
    return keras_model


def keras2pb(keras_model, output_file: Path | str, prefix: str = "") -> str:
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2,
    )

    LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")
    m = tf.function(lambda x: keras_model(x))
    m = m.get_concrete_function(
        tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype)
    )
    frozen_func = convert_variables_to_constants_v2(m)
    frozen_func.graph.as_graph_def()
    output_file = Path(output_file)
    tf.io.write_graph(
        graph_or_graph_def=frozen_func.graph,
        logdir=str(output_file.parent),
        name=output_file.name,
        as_text=False,
    )
    return str(output_file)


def tflite2edgetpu(
    tflite_file: str | Path, output_dir: str | Path, prefix: str = ""
) -> str:
    import subprocess

    check_cmd = "edgetpu_compiler --version"
    help_url = "https://coral.ai/docs/edgetpu/compiler/"
    assert LINUX, f"export only supported on Linux. See {help_url}"
    if (
        subprocess.run(
            check_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True
        ).returncode
        != 0
    ):
        LOGGER.info(
            f"\n{prefix} export requires Edge TPU compiler. Attempting install from {help_url}"
        )
        sudo = "sudo " if is_sudo_available() else ""
        for c in (
            f"{sudo}mkdir -p /etc/apt/keyrings",
            f"curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | {sudo}gpg --no-tty --dearmor -o /etc/apt/keyrings/google.gpg",
            f'echo "deb [signed-by=/etc/apt/keyrings/google.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" | {sudo}tee /etc/apt/sources.list.d/coral-edgetpu.list',
        ):
            subprocess.run(c, shell=True, check=True)
        check_apt_requirements(["edgetpu-compiler"])
    ver = (
        subprocess.run(check_cmd, shell=True, capture_output=True, check=True)
        .stdout.decode()
        .rsplit(maxsplit=1)[-1]
    )
    LOGGER.info(f"\n{prefix} starting export with Edge TPU compiler {ver}...")
    cmd = f'edgetpu_compiler --out_dir "{output_dir}" --show_operations --search_delegate --delegate_search_step 30 --timeout_sec 180 "{tflite_file}"'
    LOGGER.info(f"{prefix} running '{cmd}'")
    subprocess.run(cmd, shell=True)
    return str(Path(output_dir) / f"{Path(tflite_file).stem}_edgetpu.tflite")


def pb2tfjs(
    pb_file: str,
    output_dir: str,
    half: bool = False,
    int8: bool = False,
    prefix: str = "",
) -> str:
    import subprocess

    check_requirements("tensorflowjs")
    import tensorflow as tf
    import tensorflowjs as tfjs

    LOGGER.info(f"\n{prefix} starting export with tensorflowjs {tfjs.__version__}...")
    gd = tf.Graph().as_graph_def()
    with open(pb_file, "rb") as f:
        gd.ParseFromString(f.read())
    outputs = ",".join(gd_outputs(gd))
    LOGGER.info(f"\n{prefix} output node names: {outputs}")
    quantization = "--quantize_float16" if half else "--quantize_uint8" if int8 else ""
    with spaces_in_path(pb_file) as fpb_, spaces_in_path(output_dir) as f_:
        cmd = f'tensorflowjs_converter --input_format=tf_frozen_model {quantization} --output_node_names={outputs} "{fpb_}" "{f_}"'
        LOGGER.info(f"{prefix} running '{cmd}'")
        subprocess.run(cmd, shell=True)
    if " " in output_dir:
        LOGGER.warning(
            f"{prefix} your model may not work correctly with spaces in path '{output_dir}'."
        )
    return str(output_dir)


def gd_outputs(gd):
    name_list, input_list = ([], [])
    for node in gd.node:
        name_list.append(node.name)
        input_list.extend(node.input)
    return sorted(
        (
            f"{x}:0"
            for x in list(set(name_list) - set(input_list))
            if not x.startswith("NoOp")
        )
    )
