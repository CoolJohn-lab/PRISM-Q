"""PRISM-Q for ComfyUI: loads packed PRISM-Q trellis checkpoints (format prism-q-trellis-v1) through the stock
"Load Diffusion Model" node. Adds no nodes; it registers the prism_trellis quantization format."""
from . import prism_trellis  # noqa: F401  registers the prism_trellis quantization format

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
