import math
import os

import torch
from safetensors import safe_open


# Bytes per MXFP4 block: 32 FP4 numbers packed in 16 bytes
BYTES_PER_BLOCK = 16

# 4-bit 浮点查找表（MXFP4 的离散数值集合）。
# 每个 nibble（0~15）映射到一个近似浮点值，后续再乘 2^scale 还原动态范围。
FP4_VALUES = [
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]

# Map the names assumed in this implementation to the checkpoint names.
# 说明：
# - 普通参数（bias、非 MoE 权重）通常可直接按同名读取；
# - MoE 的两个大权重（mlp1_weight/mlp2_weight）在 checkpoint 里拆成
#   `*.blocks`（4-bit 主体）与 `*.scales`（按块指数）两份，需要组合解码。
PARAM_NAME_MAP = {
    f"block.{n}.mlp.mlp1_bias": f"block.{n}.mlp.mlp1_bias" for n in range(36)
} | {
    f"block.{n}.mlp.mlp1_weight": (f"block.{n}.mlp.mlp1_weight.blocks", f"block.{n}.mlp.mlp1_weight.scales") for n in range(36)
} | {
    f"block.{n}.mlp.mlp2_bias": f"block.{n}.mlp.mlp2_bias" for n in range(36)
} | {
    f"block.{n}.mlp.mlp2_weight": (f"block.{n}.mlp.mlp2_weight.blocks", f"block.{n}.mlp.mlp2_weight.scales") for n in range(36)
}


class Checkpoint:
    def __init__(self, path: str, device: torch.device):
        # safetensors 需要形如 "cuda:0" 的设备字符串。
        device_str = (
            device.type
            if device.index is None
            else device.type + ":" + str(device.index)
        )
        self.device_str = device_str

        # Read from all files ending with .safetensors in the checkpoint directory
        # 一个 checkpoint 可能由多个 safetensors 分片组成。
        safetensor_files = [
            os.path.join(path, fname)
            for fname in os.listdir(path)
            if fname.endswith(".safetensors")
        ]
        # Build a mapping from tensor name to (file, key)
        # 先建立“张量名 -> 所在文件”索引，后续按需懒加载具体张量。
        tensor_name_to_file = {}
        for safetensor_file in safetensor_files:
            with safe_open(safetensor_file, framework="pt", device=device_str) as f:
                for key in f.keys():
                    tensor_name_to_file[key] = safetensor_file

        self.tensor_name_to_file = tensor_name_to_file

    def get(self, name: str) -> torch.Tensor:
        # PARAM_NAME_MAP 支持两类返回：
        # 1) str: 直接读取普通 tensor；
        # 2) (blocks, scales): 表示 MXFP4 权重，需要走专门解码。
        match PARAM_NAME_MAP.get(name, name):
            case (blocks_name, scales_name):
                # MoE weights: are in block-based MXFP4 format
                return self._get_mxfp4_tensor(blocks_name, scales_name, dtype=torch.bfloat16)
            case tensor_name:
                # MoE biases and other weights
                return self._get_tensor(tensor_name)

    def _get_tensor(self, name: str) -> str:
        # 基础读取：从对应 safetensors 文件按 key 提取 tensor。
        assert name in self.tensor_name_to_file, f"Tensor {name} not found in checkpoint."
        with safe_open(
            self.tensor_name_to_file[name], framework="pt", device=self.device_str
        ) as f:
            return f.get_tensor(name)

    def _get_mxfp4_tensor(
        self,
        blocks_name: str,
        scales_name: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        rows_per_chunk: int = 16384 * 512,
    ) -> torch.Tensor:
        # blocks/scales 必须同时存在，且形状前缀匹配。
        assert blocks_name in self.tensor_name_to_file, (
            f"Blocks tensor {blocks_name} not found in checkpoint."
        )
        assert scales_name in self.tensor_name_to_file, (
            f"Scales tensor {scales_name} not found in checkpoint."
        )

        # blocks: uint8 packed nibbles；scales: 带 127 偏移的指数。
        blocks = self._get_tensor(blocks_name)
        scales = self._get_tensor(scales_name).to(torch.int32) - 127

        assert blocks.shape[:-1] == scales.shape, (
            f"{blocks.shape=} does not match {scales.shape=}"
        )

        # LUT 放在目标 dtype 上，后续索引得到近似 mantissa。
        lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

        # blocks shape 末两维通常是 [G, B]:
        # - G: 组/块维（来自原权重分组）；
        # - B: 每组里打包后的字节数（每字节含 2 个 FP4）。
        *prefix_shape, G, B = blocks.shape
        rows_total   = math.prod(prefix_shape) * G

        # 展平成二维，便于分块流水处理，减少峰值显存。
        blocks = blocks.reshape(rows_total, B)
        scales = scales.reshape(rows_total, 1)

        # 输出展开后每行长度为 B*2（每字节解出两个 nibble）。
        out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

        # 分块解码，避免一次性展开超大 tensor 造成 OOM。
        for r0 in range(0, rows_total, rows_per_chunk):
            r1 = min(r0 + rows_per_chunk, rows_total)

            blk = blocks[r0:r1]
            exp = scales[r0:r1]

            # nibble indices -> int64
            # 低 4 位 / 高 4 位分别是两个 FP4 编码索引。
            idx_lo = (blk & 0x0F).to(torch.long)
            idx_hi = (blk >> 4).to(torch.long)

            sub = out[r0:r1]
            # LUT 查表还原 mantissa，并交错写回（为 SwiGLU 双分支排列准备）。
            sub[:, 0::2] = lut[idx_lo]
            sub[:, 1::2] = lut[idx_hi]

            # 乘以 2^exp（逐行广播）还原真实数值量级。
            torch.ldexp(sub, exp, out=sub)
            del idx_lo, idx_hi, blk, exp

        # 还原回原前缀维，并把末两维并成线性层期望的输入维。
        return out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)

    def _get_mxfp4_tensor_copy(self, blocks_name: str, scales_name: str, dtype: torch.dtype = torch.bfloat16):
        "short version that uses a lot of memory"
        # 这是更直观但更吃显存的参考实现，便于对照与验证。

        loaded_blocks = self._get_tensor(blocks_name)
        # Split it into low and high nibbles, upcast to bytes, and interleave (for swiglu)
        loaded_blocks_lo = loaded_blocks & 0x0F
        loaded_blocks_hi = loaded_blocks >> 4
        loaded_blocks = torch.stack((loaded_blocks_lo, loaded_blocks_hi), dim=-1)
        loaded_blocks = loaded_blocks.view(*loaded_blocks.shape[:-2], loaded_blocks.shape[-2] * 2)

        loaded_scales = self._get_tensor(scales_name)
        # Upcast to int32 and subtract bias
        loaded_scales = loaded_scales.int() - 127

        # Convert MXFP4 numbers into target dtype
        fp4_values = torch.tensor(FP4_VALUES, dtype=dtype, device=self.device_str)
        loaded_tensor = torch.ldexp(fp4_values[loaded_blocks.int()], loaded_scales.unsqueeze(-1))
        loaded_tensor = loaded_tensor.view(*loaded_tensor.shape[:-2], -1)
        return loaded_tensor
