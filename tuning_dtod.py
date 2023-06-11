import math
import itertools
import torch
import torchvision
from omegaconf import DictConfig, OmegaConf
import hydra
import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import os
import sys
import contextlib
import time
import tuning_batch
import ghetto_nvds


# context manager to help keep track of ranges of time, using NVTX
@contextlib.contextmanager
def nvtx_range(msg):
    depth = torch.cuda.nvtx.range_push(msg)
    try:
        yield depth
    finally:
        torch.cuda.nvtx.range_pop()


class Pipeline(tuning_batch.Pipeline):
    def __init__(self, *args, **kwargs):
        super(Pipeline, self).__init__(*args, **kwargs)
        print("dtod __init__")

    def create_pipeline(self, input_src, num_buffers, frame_format, sink_name):
        print("dtod create_pipeline")
        return Gst.parse_launch(f'''
            filesrc location={input_src} num-buffers={num_buffers} !
            decodebin !
            nvvideoconvert !
            video/x-raw(memory:NVMM),format={frame_format} !
            fakesink name={sink_name}
        ''')

    def buffer_to_image_tensor(self, buf, caps):
        with nvtx_range('buffer_to_image_tensor'):
            caps_structure = caps.get_structure(0)
            height, width = caps_structure.get_value('height'), caps_structure.get_value('width')

            is_mapped, map_info = buf.map(Gst.MapFlags.READ)
            if is_mapped:
                try:
                    source_surface = ghetto_nvds.NvBufSurface(map_info)
                    torch_surface = ghetto_nvds.NvBufSurface(map_info)

                    dest_tensor = torch.zeros(
                        (torch_surface.surfaceList[0].height, torch_surface.surfaceList[0].width, 4),
                        dtype=torch.uint8,
                        device=self.device
                    )

                    torch_surface.struct_copy_from(source_surface)
                    assert(source_surface.numFilled == 1)
                    assert(source_surface.surfaceList[0].colorFormat == 19) # RGBA

                    # make torch_surface map to dest_tensor memory
                    torch_surface.surfaceList[0].dataPtr = dest_tensor.data_ptr()

                    # copy decoded GPU buffer (source_surface) into Pytorch tensor (torch_surface -> dest_tensor)
                    torch_surface.mem_copy_from(source_surface)
                finally:
                    buf.unmap(map_info)
                
                return dest_tensor[:, :, :3]


@hydra.main(version_base=None, config_path="conf", config_name="tuning_dtod")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    Gst.init()
    pipeline: Pipeline = hydra.utils.instantiate(cfg.pipeline)
    print("precision", pipeline.model_precision)
    pipeline.set_playing_state()
    pipeline.loop()


if __name__ == "__main__":
    my_app()
