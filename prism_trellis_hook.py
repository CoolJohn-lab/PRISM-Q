"""Routes layers whose comfy_quant format ComfyUI does not know to this pack's loaders, so its formats load through
the stock "Load Diffusion Model" node. Wraps comfy.ops._load_quantized_module once; any other wrapper (another
node pack doing the same) stays in the chain because unknown formats fall through to the previous function."""
import json

import comfy.ops

LOADERS = {}  # comfy_quant format -> loader(module, super_load, state_dict, prefix, layer_conf, ...)
_previous = comfy.ops._load_quantized_module


def _load_quantized_module(module, super_load, state_dict, prefix, local_metadata, strict,
                           missing_keys, unexpected_keys, error_msgs, load_extra_params=False):
    conf = state_dict.get(f'{prefix}comfy_quant')
    if conf is not None and f'{prefix}weight' in state_dict:
        layer_conf = json.loads(conf.numpy().tobytes())
        loader = LOADERS.get(layer_conf.get('format'))
        if loader is not None:
            return loader(module, super_load, state_dict, prefix, layer_conf, local_metadata, strict,
                          missing_keys, unexpected_keys, error_msgs)
    return _previous(module, super_load, state_dict, prefix, local_metadata, strict,
                     missing_keys, unexpected_keys, error_msgs, load_extra_params=load_extra_params)


_load_quantized_module.prism_trellis_hook = True
if not getattr(comfy.ops._load_quantized_module, 'prism_trellis_hook', False):
    comfy.ops._load_quantized_module = _load_quantized_module
