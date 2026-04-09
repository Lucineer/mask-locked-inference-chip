#!/usr/bin/env python3
"""Technical architecture for mask-locked inference chips.

Based on SuperInstance core_documents/01_Technical_Architecture.md.
"""
import json, math
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class ChipArchitecture(Enum):
    WEIGHT_LOCKED = "weight_locked"
    ARCHITECTURE_LOCKED = "architecture_locked"
    HYBRID = "hybrid"


class ComputeFabric(Enum):
    SYSTOLIC_ARRAY = "systolic_array"
    TLMM = "tlmm"  # Table-Lookup MatMul
    NEUROMORPHIC = "neuromorphic"
    ANALOG = "analog"


@dataclass
class LayerSpec:
    name: str
    type: str  # "attention", "ffn", "embed", "norm"
    rows: int
    cols: int
    precision_bits: int
    weight_bytes: int = 0
    mac_count: int = 0

    def __post_init__(self):
        self.mac_count = self.rows * self.cols
        self.weight_bytes = math.ceil(self.mac_count * self.precision_bits / 8)


@dataclass
class ChipSpec:
    name: str
    architecture: ChipArchitecture
    fabric: ComputeFabric
    process_nm: int
    die_area_mm2: float
    clock_mhz: int
    power_budget_w: float
    layers: List[LayerSpec] = field(default_factory=list)
    kv_cache_bytes: int = 0
    context_length: int = 2048

    @property
    def total_macs(self) -> int:
        return sum(l.mac_count for l in self.layers)

    @property
    def total_weight_bytes(self) -> int:
        return sum(l.weight_bytes for l in self.layers)

    @property
    def ops_per_cycle(self) -> int:
        return self.total_macs * 2  # multiply + add

    @property
    def peak_gops(self) -> float:
        return self.ops_per_cycle * self.clock_mhz * 1e6 / 1e9

    def summary(self) -> Dict:
        return {
            "name": self.name,
            "architecture": self.architecture.value,
            "fabric": self.fabric.value,
            "process_nm": self.process_nm,
            "die_area_mm2": self.die_area_mm2,
            "clock_mhz": self.clock_mhz,
            "power_budget_w": self.power_budget_w,
            "total_macs": self.total_macs,
            "total_weight_mb": self.total_weight_bytes / 1e6,
            "peak_gops": round(self.peak_gops, 1),
            "layers": len(self.layers),
        }


class ArchitectureGenerator:
    """Generate chip architectures based on model parameters."""

    @staticmethod
    def scout() -> ChipSpec:
        """1B parameter chip."""
        layers = []
        hidden = 512
        heads = 8
        head_dim = hidden // heads
        for i in range(12):
            layers.append(LayerSpec(f"layer{i}_qkv", "attention", hidden, hidden * 3, 4))
            layers.append(LayerSpec(f"layer{i}_out", "attention", hidden, hidden, 4))
            layers.append(LayerSpec(f"layer{i}_ffn_up", "ffn", hidden, hidden * 4, 4))
            layers.append(LayerSpec(f"layer{i}_ffn_down", "ffn", hidden * 4, hidden, 4))
            layers.append(LayerSpec(f"layer{i}_ln1", "norm", 1, hidden, 32))
            layers.append(LayerSpec(f"layer{i}_ln2", "norm", 1, hidden, 32))
        layers.append(LayerSpec("embed", "embed", 32000, hidden, 8))
        layers.append(LayerSpec("lm_head", "embed", hidden, 32000, 8))
        layers.append(LayerSpec("final_norm", "norm", 1, hidden, 32))

        return ChipSpec(
            name="Scout",
            architecture=ChipArchitecture.WEIGHT_LOCKED,
            fabric=ComputeFabric.SYSTOLIC_ARRAY,
            process_nm=28,
            die_area_mm2=48,
            clock_mhz=500,
            power_budget_w=1.0,
            layers=layers,
            kv_cache_bytes=2048 * heads * 2 * head_dim * 2,  # INT16
        )

    @staticmethod
    def messenger() -> ChipSpec:
        """3B parameter chip."""
        spec = ArchitectureGenerator.scout()
        spec.name = "Messenger"
        spec.die_area_mm2 = 48  # Same die size, higher utilization
        spec.power_budget_w = 3.0
        # Scale layers: 24 layers, 768 hidden
        spec.layers = []
        hidden = 768
        heads = 12
        head_dim = hidden // heads
        for i in range(24):
            spec.layers.append(LayerSpec(f"layer{i}_qkv", "attention", hidden, hidden * 3, 4))
            spec.layers.append(LayerSpec(f"layer{i}_out", "attention", hidden, hidden, 4))
            spec.layers.append(LayerSpec(f"layer{i}_ffn_up", "ffn", hidden, hidden * 4, 4))
            spec.layers.append(LayerSpec(f"layer{i}_ffn_down", "ffn", hidden * 4, hidden, 4))
            spec.layers.append(LayerSpec(f"layer{i}_ln1", "norm", 1, hidden, 32))
            spec.layers.append(LayerSpec(f"layer{i}_ln2", "norm", 1, hidden, 32))
        spec.layers.append(LayerSpec("embed", "embed", 32000, hidden, 8))
        spec.layers.append(LayerSpec("lm_head", "embed", hidden, 32000, 8))
        spec.layers.append(LayerSpec("final_norm", "norm", 1, hidden, 32))
        spec.kv_cache_bytes = 2048 * heads * 2 * head_dim * 2
        return spec

    @staticmethod
    def navigator() -> ChipSpec:
        """7B parameter chip."""
        spec = ArchitectureGenerator.scout()
        spec.name = "Navigator"
        spec.die_area_mm2 = 100
        spec.power_budget_w = 6.0
        spec.clock_mhz = 500
        hidden = 1024
        heads = 16
        head_dim = hidden // heads
        spec.layers = []
        for i in range(32):
            spec.layers.append(LayerSpec(f"layer{i}_qkv", "attention", hidden, hidden * 3, 4))
            spec.layers.append(LayerSpec(f"layer{i}_out", "attention", hidden, hidden, 4))
            spec.layers.append(LayerSpec(f"layer{i}_ffn_up", "ffn", hidden, hidden * 4, 4))
            spec.layers.append(LayerSpec(f"layer{i}_ffn_down", "ffn", hidden * 4, hidden, 4))
            spec.layers.append(LayerSpec(f"layer{i}_ln1", "norm", 1, hidden, 32))
            spec.layers.append(LayerSpec(f"layer{i}_ln2", "norm", 1, hidden, 32))
        spec.layers.append(LayerSpec("embed", "embed", 32000, hidden, 8))
        spec.layers.append(LayerSpec("lm_head", "embed", hidden, 32000, 8))
        spec.layers.append(LayerSpec("final_norm", "norm", 1, hidden, 32))
        spec.kv_cache_bytes = 2048 * heads * 2 * head_dim * 2
        return spec

    @staticmethod
    def captain() -> ChipSpec:
        """13B parameter chip."""
        spec = ArchitectureGenerator.scout()
        spec.name = "Captain"
        spec.die_area_mm2 = 200
        spec.power_budget_w = 12.0
        spec.clock_mhz = 500
        hidden = 1536
        heads = 24
        head_dim = hidden // heads
        spec.layers = []
        for i in range(40):
            spec.layers.append(LayerSpec(f"layer{i}_qkv", "attention", hidden, hidden * 3, 4))
            spec.layers.append(LayerSpec(f"layer{i}_out", "attention", hidden, hidden, 4))
            spec.layers.append(LayerSpec(f"layer{i}_ffn_up", "ffn", hidden, hidden * 4, 4))
            spec.layers.append(LayerSpec(f"layer{i}_ffn_down", "ffn", hidden * 4, hidden, 4))
            spec.layers.append(LayerSpec(f"layer{i}_ln1", "norm", 1, hidden, 32))
            spec.layers.append(LayerSpec(f"layer{i}_ln2", "norm", 1, hidden, 32))
        spec.layers.append(LayerSpec("embed", "embed", 32000, hidden, 8))
        spec.layers.append(LayerSpec("lm_head", "embed", hidden, 32000, 8))
        spec.layers.append(LayerSpec("final_norm", "norm", 1, hidden, 32))
        spec.kv_cache_bytes = 2048 * heads * 2 * head_dim * 2
        return spec


class ArchitectureAnalyzer:
    """Analyze chip architectures."""

    @staticmethod
    def compute_density(spec: ChipSpec) -> float:
        """Compute density (MACs/mm2)."""
        return spec.total_macs / spec.die_area_mm2

    @staticmethod
    def power_efficiency(spec: ChipSpec) -> float:
        """Power efficiency (GOPS/W)."""
        return spec.peak_gops / spec.power_budget_w

    @staticmethod
    def memory_bandwidth_requirement(spec: ChipSpec, tokens_per_sec: float) -> float:
        """Required memory bandwidth (GB/s)."""
        # Bytes per token = weights + KV cache
        weight_bytes_per_token = spec.total_weight_bytes / spec.context_length
        kv_bytes_per_token = spec.kv_cache_bytes / spec.context_length
        total_bytes_per_token = weight_bytes_per_token + kv_bytes_per_token
        return total_bytes_per_token * tokens_per_sec / 1e9

    @staticmethod
    def compare(specs: List[ChipSpec]) -> Dict:
        """Compare multiple architectures."""
        results = []
        for spec in specs:
            density = ArchitectureAnalyzer.compute_density(spec)
            efficiency = ArchitectureAnalyzer.power_efficiency(spec)
            results.append({
                "name": spec.name,
                "macs_mm2": round(density / 1e6, 1),  # Millions/mm2
                "gops_w": round(efficiency, 1),
                "peak_gops": round(spec.peak_gops, 1),
                "power_w": spec.power_budget_w,
                "die_mm2": spec.die_area_mm2,
                "process_nm": spec.process_nm,
            })
        return {"comparison": results}


def demo():
    print("=== Mask-Locked Inference Chip: Technical Architecture ===\n")

    gen = ArchitectureGenerator()
    specs = [gen.scout(), gen.messenger(), gen.navigator(), gen.captain()]

    print("--- Vessel Classes ---")
    for spec in specs:
        s = spec.summary()
        print(f"  {s['name']:12s}: {s['total_macs']/1e9:5.1f}B MACs, {s['die_area_mm2']:5.0f}mm2, "
              f"{s['clock_mhz']:4d}MHz, {s['power_budget_w']:4.1f}W, {s['peak_gops']:6.1f} GOPS")
    print()

    print("--- Architecture Analysis ---")
    analyzer = ArchitectureAnalyzer()
    comp = analyzer.compare(specs)
    for c in comp["comparison"]:
        print(f"  {c['name']:12s}: {c['macs_mm2']:5.1f}M MACs/mm2, {c['gops_w']:5.1f} GOPS/W, "
              f"{c['peak_gops']:6.1f} GOPS peak")
    print()

    print("--- Memory Bandwidth Requirements ---")
    for spec, tok_s in [(specs[0], 100), (specs[1], 80), (specs[2], 50), (specs[3], 30)]:
        bw = analyzer.memory_bandwidth_requirement(spec, tok_s)
        print(f"  {spec.name:12s} @ {tok_s:3d} tok/s: {bw:5.1f} GB/s")
    print()

    print("--- Key Architectural Principles ---")
    principles = [
        "1. Weight-locked: Model weights are hardwired into metal interconnect",
        "2. No CPU/OS: Pure dataflow architecture, no software stack",
        "3. Mixed precision: LayerNorm=FP32, Embed=INT8, Attention/FFN=INT4",
        "4. Systolic arrays: 2D grid of MAC units with pipelined data flow",
        "5. On-chip KV cache: SRAM for attention key/value states",
        "6. Thermal-aware: Hardware DVFS and clock gating",
        "7. Zero boot time: Powers on and immediately processes tokens",
        "8. Deterministic latency: Fixed throughput independent of input",
    ]
    for p in principles:
        print(f"  {p}")
    print()

    print("--- Comparison vs Traditional AI Accelerators ---")
    comparison = [
        ("NVIDIA Jetson", "GPU", "Flexible", "High", "10-30W", "$$$"),
        ("Google Coral", "TPU", "Vision-focused", "Medium", "2-5W", "$$"),
        ("Hailo-8", "NPU", "Vision/LLM", "Medium", "3-8W", "$$"),
        ("Mask-Locked", "Weight-locked", "Fixed model", "Extreme", "1-12W", "$"),
    ]
    print(f"  {'Chip':20s} {'Architecture':15s} {'Flexibility':12s} {'Efficiency':10s} {'Power':8s} {'Cost':6s}")
    print("  " + "-" * 75)
    for name, arch, flex, eff, power, cost in comparison:
        print(f"  {name:20s} {arch:15s} {flex:12s} {eff:10s} {power:8s} {cost:6s}")


if __name__ == "__main__":
    demo()
