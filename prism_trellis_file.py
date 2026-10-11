"""Copy of src/prism4/trellis_file.py in the PRISM-Q research repository (keep in sync). Packed PRISM-Q trellis checkpoints: the shippable form of a QTIP-INT8 model (see trellis.py for the code).

A packed file is a ConvRot int8_tensorwise checkpoint in which each trellis-coded linear stores its bitstream:
  <layer>.weight        uint8 (n, k*K/8)    the tile payloads, K bits per weight: (n/16, k/16, 32*K) for 16x16 tiles
                                            row-major, stored flattened to n rows like ComfyUI's packed int4 weights
                                            (model detection reads layer widths from weight.shape[0])
  <layer>.weight_level  uint8 (n/16, k/16)  codebook level of each tile
  <layer>.weight_scale  unchanged (FP32 row scale)
  <layer>.comfy_quant   the int8_tensorwise config with "format": "prism_trellis" (convrot settings kept)
Layers on the plain int8 grid and every other tensor are unchanged; header metadata 'format' = FORMAT. With the
ComfyUI-PRISM-Q node pack the stock loader keeps the payload in VRAM and decodes each layer just in time
(prism_trellis.py); expand() instead turns a packed state dict back into the exact ConvRot state dict (bit-identical
int8 weights). Torch only, so the node pack vendors this file."""
import json

import torch

FORMAT = 'prism-q-trellis-v1'
LEGACY_FORMATS = ('prism4-trellis-v1',)  # files packed before the PRISM-Q rename
QUANT_FORMAT = 'prism_trellis'
BASE_FORMAT = 'int8_tensorwise'
L = 16
LEVELS = 32
G0 = 127.0 / 510.0


def level_mults(device='cpu'):
    l = torch.arange(LEVELS, dtype=torch.float64)
    return torch.round(G0 * torch.pow(2.0, -l / 8) * 65536).to(torch.int64).to(device)


def decode_tiles(payload, levels, mults=None):
    """payload (tiles, 32*K) uint8, levels (tiles,) -> int8 values (tiles, 256). State t is the 16 payload bits
    from bit K*t, cyclic; value = (r(state) * mult[level] + 32768) >> 16 with the QTIP 1MAD codebook r."""
    tiles, nbytes = payload.shape
    K = nbytes // 32
    dev = payload.device
    mults = level_mults(dev) if mults is None else mults
    bit = torch.arange(256, device=dev) * K
    byte, shift = bit // 8, (bit % 8)
    b = payload.long()
    word = (b[:, byte] << 16) | (b[:, (byte + 1) % nbytes] << 8) | b[:, (byte + 2) % nbytes]  # 24 bits from byte
    states = (word >> (8 - shift)) & 0xFFFF
    m = (states * 34038481 + 76625530) & 0xFFFFFFFF
    r = (m & 255) + ((m >> 8) & 255) + ((m >> 16) & 255) + ((m >> 24) & 255) - 510
    return ((r * mults[levels.long()][:, None] + 32768) >> 16).to(torch.int8)


def to_file(payload):
    """(n/16, k/16, 32K) tile payloads -> the stored (n, k*K/8) weight (a view of the same bytes)."""
    tn, tk, nbytes = payload.shape
    return payload.reshape(tn * 16, tk * nbytes // 16)


def from_file(weight, levels):
    """Stored (n, k*K/8) weight and its (n/16, k/16) levels -> (n/16, k/16, 32K) tile payloads."""
    return weight.reshape(*levels.shape, -1)


def decode_layer(payload, levels, chunk=1 << 16):
    """Payload ((n/16, k/16, 32K) tiles or the stored (n, k*K/8) form) and (n/16, k/16) levels -> int8 weight (n, k),
    decoded chunk tiles at a time."""
    payload = from_file(payload, levels) if payload.dim() == 2 else payload
    tn, tk, nbytes = payload.shape
    flat, lv = payload.reshape(-1, nbytes), levels.reshape(-1)
    mults = level_mults(payload.device)
    out = torch.empty(flat.shape[0], 256, dtype=torch.int8, device=payload.device)
    for i in range(0, flat.shape[0], chunk):
        out[i:i + chunk] = decode_tiles(flat[i:i + chunk], lv[i:i + chunk], mults)
    return out.view(tn, tk, 16, 16).permute(0, 2, 1, 3).reshape(tn * 16, tk * 16)


def read_conf(tensor):
    return json.loads(tensor.numpy().tobytes())


def write_conf(conf):
    return torch.frombuffer(bytearray(json.dumps(conf).encode()), dtype=torch.uint8)


def with_format(conf_tensor, fmt):
    """comfy_quant tensor with its 'format' replaced (key order kept, so the base config round-trips exactly)."""
    conf = read_conf(conf_tensor)
    conf['format'] = fmt
    return write_conf(conf)


def is_packed(metadata):
    return bool(metadata) and metadata.get('format') in (FORMAT, *LEGACY_FORMATS)


def packed_layers(sd):
    """Names of the trellis layers of a packed state dict."""
    return [k[:-len('.comfy_quant')] for k in sd
            if k.endswith('.comfy_quant') and read_conf(sd[k]).get('format') == QUANT_FORMAT]


def expand(sd, device=None):
    """Packed state dict -> ConvRot state dict, in place (decodes on device, default CUDA if available)."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    for name in packed_layers(sd):
        payload, levels = sd[name + '.weight'], sd.pop(name + '.weight_level')
        sd[name + '.weight'] = decode_layer(payload.to(device), levels.to(device)).cpu()
        sd[name + '.comfy_quant'] = with_format(sd[name + '.comfy_quant'], BASE_FORMAT)
    return sd
