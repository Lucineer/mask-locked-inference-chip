#!/usr/bin/env python3
"""Tapeout cost estimator and timeline planner for mask-locked inference chips."""
import json, math
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class Foundry(Enum):
    TSMC_28 = "tsmc_28"
    TSMC_40 = "tsmc_40"
    SMIC_28 = "smic_28"
    SMIC_40 = "smic_40"
    GF_65 = "gf_65"
    SKYWATER_130 = "skywater_130"


class PackageType(Enum):
    QFN = "qfn"
    BGA = "bga"
    WLCSP = "wlscp"
    SIP = "sip"


# Cost data (approximate 2025-2026 pricing)
FOUNDRY_COSTS = {
    "tsmc_28": {"mask_set_k": 2500, "per_wafer_k": 3.5, "wafer_mm": 300, "nre_k": 500, "min_lot": 25},
    "tsmc_40": {"mask_set_k": 1500, "per_wafer_k": 2.0, "wafer_mm": 300, "nre_k": 300, "min_lot": 25},
    "smic_28": {"mask_set_k": 1800, "per_wafer_k": 2.5, "wafer_mm": 300, "nre_k": 400, "min_lot": 25},
    "smic_40": {"mask_set_k": 1000, "per_wafer_k": 1.5, "wafer_mm": 300, "nre_k": 200, "min_lot": 25},
    "gf_65":   {"mask_set_k": 600,  "per_wafer_k": 0.8, "wafer_mm": 200, "nre_k": 100, "min_lot": 25},
    "skywater_130": {"mask_set_k": 15, "per_wafer_k": 0.01, "wafer_mm": 200, "nre_k": 10, "min_lot": 1},
}

PACKAGE_COSTS = {
    "qfn": {"per_unit_cents": 5, "min_order": 10000},
    "bga": {"per_unit_cents": 15, "min_order": 5000},
    "wlscp": {"per_unit_cents": 3, "min_order": 100000},
    "sip": {"per_unit_cents": 50, "min_order": 1000},
}

TIMELINE_MONTHS = {
    "tsmc_28": {"tapeout_to_die": 3, "packaging": 1, "testing": 1, "total": 5},
    "tsmc_40": {"tapeout_to_die": 2.5, "packaging": 1, "testing": 1, "total": 4.5},
    "smic_28": {"tapeout_to_die": 3, "packaging": 1, "testing": 1.5, "total": 5.5},
    "smic_40": {"tapeout_to_die": 2.5, "packaging": 1, "testing": 1.5, "total": 5},
    "gf_65":   {"tapeout_to_die": 2, "packaging": 0.5, "testing": 1, "total": 3.5},
    "skywater_130": {"tapeout_to_die": 4, "packaging": 1, "testing": 1, "total": 6},
}

MPW_OPTIONS = {
    "tsmc_28": {"shuttles_per_year": 4, "cost_per_slot_k": 50, "slots_available": 10, "area_limit_mm2": 50},
    "skywater_130": {"shuttles_per_year": 6, "cost_per_slot_k": 0.3, "slots_available": 100, "area_limit_mm2": 10},
}


@dataclass
class ChipSpec:
    name: str
    die_area_mm2: float
    foundry: Foundry
    package: PackageType
    target_volume: int  # annual units
    yield_pct: float = 0.85

    @property
    def dies_per_wafer(self) -> int:
        fc = FOUNDRY_COSTS[self.foundry.value]
        wafer_area = math.pi * (fc["wafer_mm"] / 2) ** 2
        die_side = math.sqrt(self.die_area_mm2)
        gross = int((wafer_area / self.die_area_mm2) * 0.85)
        return gross

    @property
    def good_dies_per_wafer(self) -> int:
        return int(self.dies_per_wafer * self.yield_pct)


class TapeoutPlanner:
    """Plan and cost a chip tapeout."""

    def __init__(self, spec: ChipSpec):
        self.spec = spec
        self.fc = FOUNDRY_COSTS[spec.foundry.value]
        self.pkg = PACKAGE_COSTS[spec.package.value]

    def nre_cost(self) -> Dict:
        return {
            "mask_set_k": self.fc["mask_set_k"],
            "design_nre_k": self.fc["nre_k"],
            "total_nre_k": self.fc["mask_set_k"] + self.fc["nre_k"],
        }

    def per_unit_cost(self, volume: Optional[int] = None) -> Dict:
        v = volume or self.spec.target_volume
        fc = self.fc
        wafers_needed = math.ceil(v / self.spec.good_dies_per_wafer / 12)  # 12 months
        die_cost = (fc["per_wafer_k"] * 1000 / self.spec.good_dies_per_wafer) / 100
        pkg_cost = self.pkg["per_unit_cents"] / 100
        test_cost = 0.05  # ~$0.05 per unit for basic test
        total = die_cost + pkg_cost + test_cost
        return {"die_cost": round(die_cost, 3), "package_cost": pkg_cost,
                "test_cost": test_cost, "total_cost": round(total, 3),
                "wafers_per_year": wafers_needed}

    def annual_cost(self, volume: Optional[int] = None) -> Dict:
        v = volume or self.spec.target_volume
        per = self.per_unit_cost(v)
        nre = self.nre_cost()
        manufacturing = round(per["total_cost"] * v / 1000, 0)
        return {
            "nre_k": nre["total_nre_k"],
            "manufacturing_k": manufacturing,
            "total_year1_k": round(nre["total_nre_k"] + manufacturing, 0),
            "per_unit_cents": round(per["total_cost"] * 100, 1),
            "break_even_units": int(nre["total_nre_k"] * 1000 / per["total_cost"]) if per["total_cost"] > 0 else 0,
        }

    def mpw_cost(self) -> Optional[Dict]:
        if self.spec.foundry.value not in MPW_OPTIONS:
            return None
        mpw = MPW_OPTIONS[self.spec.foundry.value]
        if self.spec.die_area_mm2 > mpw["area_limit_mm2"]:
            return {"feasible": False, "reason": f"Die {self.spec.die_area_mm2}mm2 > limit {mpw['area_limit_mm2']}mm2"}
        return {
            "feasible": True,
            "cost_per_run_k": mpw["cost_per_slot_k"],
            "shuttles_per_year": mpw["shuttles_per_year"],
            "area_limit_mm2": mpw["area_limit_mm2"],
            "area_utilization_pct": round(self.spec.die_area_mm2 / mpw["area_limit_mm2"] * 100, 1),
        }

    def timeline(self) -> Dict:
        tl = TIMELINE_MONTHS[self.spec.foundry.value]
        return {
            "foundry": self.spec.foundry.value,
            "tapeout_to_die_months": tl["tapeout_to_die"],
            "packaging_months": tl["packaging"],
            "testing_months": tl["testing"],
            "total_months": tl["total"],
        }

    def full_report(self) -> Dict:
        nre = self.nre_cost()
        per = self.per_unit_cost()
        annual = self.annual_cost()
        timeline = self.timeline()
        mpw = self.mpw_cost()
        return {
            "chip": self.spec.name,
            "die_area_mm2": self.spec.die_area_mm2,
            "dies_per_wafer": self.spec.dies_per_wafer,
            "good_dies_per_wafer": self.spec.good_dies_per_wafer,
            "foundry": self.spec.foundry.value,
            "package": self.spec.package.value,
            "nre": nre,
            "per_unit": per,
            "annual": annual,
            "timeline": timeline,
            "mpw": mpw,
        }


def demo():
    print("=== Mask-Locked Inference Chip: Tapeout Planner ===\n")

    products = [
        ("Scout", 48, Foundry.GF_65, PackageType.QFN, 10000, 0.90),
        ("Messenger", 48, Foundry.SMIC_40, PackageType.QFN, 50000, 0.85),
        ("Navigator", 100, Foundry.SMIC_28, PackageType.BGA, 20000, 0.80),
        ("Captain", 200, Foundry.SMIC_28, PackageType.BGA, 5000, 0.75),
    ]

    for name, area, foundry, pkg, volume, yield_pct in products:
        spec = ChipSpec(name=name, die_area_mm2=area, foundry=foundry,
                       package=pkg, target_volume=volume, yield_pct=yield_pct)
        planner = TapeoutPlanner(spec)
        r = planner.full_report()

        print(f"--- {name} ({area}mm2, {foundry.value}, {pkg.value}) ---")
        print(f"  Dies/wafer: {r['dies_per_wafer']} gross, {r['good_dies_per_wafer']} good")
        print(f"  NRE: ${r['nre']['total_nre_k']}K (mask ${r['nre']['mask_set_k']}K + design ${r['nre']['design_nre_k']}K)")
        print(f"  Per unit: ${r['per_unit']['total_cost']:.3f} (die ${r['per_unit']['die_cost']:.3f} + pkg ${r['per_unit']['package_cost']:.2f} + test ${r['per_unit']['test_cost']:.2f})")
        a = r["annual"]
        print(f"  Year 1 ({volume:,} units): ${a['total_year1_k']}K total, ${a['per_unit_cents']}c/unit")
        print(f"  Break-even: {a['break_even_units']:,} units")
        print(f"  Timeline: {r['timeline']['total_months']} months")
        if r["mpw"]:
            m = r["mpw"]
            if m.get("feasible"):
                print(f"  MPW: {m['cost_per_run_k']}K/run, {m['shuttles_per_year']} shuttles/year, {m['area_utilization_pct']}% area")
            else:
                print(f"  MPW: NOT FEASIBLE ({m['reason']})")
        print()

    print("=== Cost Comparison Table ===\n")
    print(f"{'Product':<12} {'Foundry':<12} {'Die mm2':>8} {'NRE $K':>8} {'$/unit':>8} {'Break-even':>10} {'Timeline':>8}")
    print("-" * 70)
    for name, area, foundry, pkg, volume, yield_pct in products:
        spec = ChipSpec(name=name, die_area_mm2=area, foundry=foundry,
                       package=pkg, target_volume=volume, yield_pct=yield_pct)
        p = TapeoutPlanner(spec)
        a = p.annual_cost()
        t = p.timeline()
        print(f"{name:<12} {foundry.value:<12} {area:>8.0f} {a['nre_k']:>8.0f} {a['per_unit_cents']/100:>8.2f} {a['break_even_units']:>10,} {t['total_months']:>7.1f}mo")

    print()
    print("=== SkyWater 130nm MPW (free shuttle) ===")
    spec_mp = ChipSpec("Prototype", 8, Foundry.SKYWATER_130, PackageType.QFN, 100, 0.95)
    p = TapeoutPlanner(spec_mp)
    r = p.full_report()
    print(f"  Die: {r['die_area_mm2']}mm2, NRE: ${r['nre']['total_nre_k']}K, MPW: ${r['mpw']['cost_per_run_k']}K/run")
    print(f"  Good dies/wafer: {r['good_dies_per_wafer']}")
    print(f"  Perfect for: first tapeout, proof of concept, weight-to-metal validation")


if __name__ == "__main__":
    demo()
