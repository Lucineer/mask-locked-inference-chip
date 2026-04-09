# Mask-Locked Inference Chip

**Serverless Silicon for Edge AI**

A new category of semiconductor that physically embeds neural network weights into silicon. Unlike existing AI chips that load models into memory, this approach bakes model parameters directly into the hardware fabric — achieving unprecedented efficiency for edge AI deployment.

## The Core Idea

Send text in, get text out. No OS, no drivers, no software stack. The chip IS the model.

## Key Specs

| Product | Model | Performance | Power | Price |
|---------|-------|-------------|-------|-------|
| Nano | 1B params | 100 tok/s | <1W | $15 |
| Micro | 3B params | 80 tok/s | 2-3W | $35 |
| Standard | 7B params | 50 tok/s | 4-6W | $60 |
| Pro | 13B params | 30 tok/s | 8-12W | $120 |

## How It Works

Traditional chips store weights in memory — fetching them for every computation. A mask-locked chip encodes weights directly into the silicon's metal interconnect layers. The weights become permanent physical structures. Zero access latency. Zero access energy. Infinite bandwidth.

The tradeoff: changing weights requires fabricating a new chip. For edge inference with stable models, this tradeoff is enormously favorable.

## Documents

- [Developer Plan](docs/developer_plan.md) — Complete 10-page ASIC development roadmap
- [Deep Dive Analysis](docs/mask_locked_deep_dive.md) — Technical deep-dive with competitive analysis
- [Customer Validation](docs/customer-validation-report.md) — Market research with personas and pricing analysis

## Market

- Edge AI chip market: $3.67B (2025) → $11.54B (2030)
- 70M Raspberry Pi owners as primary TAM
- $35-50 sweet spot for maker segment
- One-time purchase strongly preferred (85%+)

## Competitive Gap

| Approach | Weight Storage | Flexibility | Efficiency |
|----------|---------------|-------------|------------|
| Traditional NPU | External DRAM | Full | Low |
| SRAM-Based (Groq) | On-chip SRAM | Model-swappable | Medium |
| Architecture-Specific (Etched) | External HBM | Architecture-fixed | High |
| **Mask-Locked** | **Metal layers** | **None** | **Extreme** |

## Tech Stack

- Process: 28nm or 40nm (mature, cost-effective)
- Quantization: INT4 baseline, INT2 optional
- Interface: USB / PCIe / M.2
- HDL: Chisel (Scala-based)

## Implementation Timeline

- Phase 1 (Months 1-6): Feasibility & Simulation
- Phase 2 (Months 7-18): RTL Design & Verification
- Phase 3 (Months 19-24): Physical Design & Tapeout
- Phase 4 (Months 25-30): Validation & Production

Part of the [Lucineer ecosystem](https://github.com/Lucineer/the-fleet).
