import torch
from omegaconf import DictConfig, OmegaConf
import hydra
import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import contextlib
from tuning_batch import Pipeline


# context manager to help keep track of ranges of time, using NVTX
@contextlib.contextmanager
def nvtx_range(msg):
    depth = torch.cuda.nvtx.range_push(msg)
    try:
        yield depth
    finally:
        torch.cuda.nvtx.range_pop()


@hydra.main(version_base=None, config_path="conf", config_name="tuning_fp16")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    Gst.init()
    pipeline: Pipeline = hydra.utils.instantiate(cfg.pipeline)
    print("precision", pipeline.model_precision)
    pipeline.set_playing_state()
    pipeline.loop()


if __name__ == "__main__":
    my_app()
