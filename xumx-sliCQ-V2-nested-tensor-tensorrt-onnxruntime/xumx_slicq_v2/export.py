from typing import Optional, Union, Tuple
import sys
from tqdm import trange
from pathlib import Path
import torch
import json
from torch import Tensor
import torch.nn as nn
from .model import Unmix
from .transforms import (
    ComplexNorm,
    NSGTBase,
    make_filterbanks,
)
import torch_tensorrt


if __name__ == '__main__':
    sample_rate=44100
    device=torch.device("cuda")

    for model_path in [
        "/xumx-sliCQ-V2/pretrained_model",
        "/xumx-sliCQ-V2/pretrained_model_realtime"
    ]:
        #for target in ['tensorrt', 'onnxruntime']:
        for target in ['onnxruntime']:
            model_path = Path(model_path)

            model_name = "xumx_slicq_v2"
            model_path = Path(model_path).expanduser()

            # load model from disk
            with open(Path(model_path, f"{model_name}.json"), "r") as stream:
                results = json.load(stream)

            # need to configure an NSGT object to peek at its params to set up the neural network
            # e.g. M depends on the sllen which depends on fscale+fmin+fmax
            nsgt_base = NSGTBase(
                results["args"]["fscale"],
                results["args"]["fbins"],
                results["args"]["fmin"],
                fs=sample_rate,
                device=device,
            )

            nb_channels = 2

            seq_dur = results["args"]["seq_dur"]

            target_model_path = Path(model_path, f"{model_name}.pth")
            state = torch.load(target_model_path, map_location=device)

            jagged_slicq, _ = nsgt_base.predict_input_size(1, nb_channels, seq_dur)
            cnorm = ComplexNorm().to(device)

            nsgt, insgt = make_filterbanks(
                nsgt_base, sample_rate
            )
            encoder = (nsgt, insgt, cnorm)

            nsgt = nsgt.to(device)
            insgt = insgt.to(device)

            jagged_slicq_cnorm = cnorm(jagged_slicq)

            xumx_model = Unmix(
                jagged_slicq_cnorm,
                realtime=results["args"]["realtime"],
            )

            xumx_model.load_state_dict(state, strict=False)
            xumx_model.freeze()
            xumx_model.to(device)

            if target == 'tensorrt':
                dest_path = Path(f'{model_name}_tensorrt.ts')

                inputs = [
                    torch_tensorrt.Input(
                        min_shape=[1, 1, 16, 16],
                        opt_shape=[1, 1, 32, 32],
                        max_shape=[1, 1, 64, 64],
                        dtype=torch.float32,
                    )
                ]
                enabled_precisions = {torch.float, torch.half}  # Run with fp16

                trt_ts_module = torch_tensorrt.compile(
                    xumx_model, inputs=inputs, enabled_precisions=enabled_precisions
                )

                input_data = input_data.to("cuda").half()
                result = trt_ts_module(input_data)
                torch.jit.save(trt_ts_module, dest_path)
            elif target == 'onnxruntime':
                dest_path = Path(f'{model_name}.onnx')

                torch.onnx.export(xumx_model,
                    jagged_slicq.to(device),
                    dest_path,
                    input_names = ['input'],              # the model's input names
                    output_names = ['output'])            # the model's output names
