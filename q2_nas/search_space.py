"""
Searchable CNN architecture space for edge devices.

An architecture is a plain JSON-serialisable dict, e.g.

    {"resolution": 32, "stem_channels": 24,
     "stages": [ {"depth": 2, "channels": 32, "kernel": 3, "block": "dwsep",  "activation": "relu"},
                 {"depth": 1, "channels": 64, "kernel": 5, "block": "mbconv", "activation": "hswish"},
                 {"depth": 3, "channels": 128,"kernel": 3, "block": "conv",   "activation": "relu6"} ]}

Searchable dimensions (all required by the assignment):
  * variable convolutional depth        - number of stages (2-4) and blocks per stage (1-3)
  * variable numbers of filters         - per-stage channel width + stem width
  * different kernel sizes              - 3 / 5 / 7 per stage
  * depthwise-separable conv blocks     - block type: conv | dwsep | mbconv
  * different activation functions      - relu | relu6 | hswish | silu per stage
  * variable input resolutions          - 24 / 32 / 40 pixels

Stage 0 keeps the resolution, every later stage halves it with a stride-2 first block.
"""
import copy
import json
import random

RESOLUTIONS = [24, 32, 40]
STEM_CHANNELS = [16, 24, 32]
NUM_STAGES = [2, 3, 4]
DEPTHS = [1, 2, 3]
KERNELS = [3, 5, 7]
BLOCKS = ["conv", "dwsep", "mbconv"]
ACTIVATIONS = ["relu", "relu6", "hswish", "silu"]
STAGE_CHANNELS = [            # channel choices per stage index (wider deeper in the net)
    [16, 24, 32, 48],
    [32, 48, 64, 96],
    [64, 96, 128, 160],
    [128, 160, 192, 256],
]

SPACE_DESCRIPTION = {
    "resolution": RESOLUTIONS, "stem_channels": STEM_CHANNELS, "num_stages": NUM_STAGES,
    "depth_per_stage": DEPTHS, "kernel": KERNELS, "block": BLOCKS, "activation": ACTIVATIONS,
    "stage_channels": STAGE_CHANNELS,
}


def sample_stage(i, rng):
    return {"depth": rng.choice(DEPTHS), "channels": rng.choice(STAGE_CHANNELS[i]),
            "kernel": rng.choice(KERNELS), "block": rng.choice(BLOCKS),
            "activation": rng.choice(ACTIVATIONS)}


def sample_architecture(rng=random):
    n = rng.choice(NUM_STAGES)
    return {"resolution": rng.choice(RESOLUTIONS), "stem_channels": rng.choice(STEM_CHANNELS),
            "stages": [sample_stage(i, rng) for i in range(n)]}


def mutate(arch, rng=random):
    """Return a copy of `arch` with exactly one randomly chosen decision re-sampled
    (guaranteed to differ from the parent)."""
    for _ in range(100):
        child = copy.deepcopy(arch)
        gene = rng.choice(["resolution", "stem_channels", "num_stages", "stage", "stage", "stage"])
        if gene == "resolution":
            child["resolution"] = rng.choice(RESOLUTIONS)
        elif gene == "stem_channels":
            child["stem_channels"] = rng.choice(STEM_CHANNELS)
        elif gene == "num_stages":
            n = rng.choice(NUM_STAGES)
            st = child["stages"][:n]
            while len(st) < n:
                st.append(sample_stage(len(st), rng))
            child["stages"] = st
        else:
            i = rng.randrange(len(child["stages"]))
            key = rng.choice(["depth", "channels", "kernel", "block", "activation"])
            choices = {"depth": DEPTHS, "channels": STAGE_CHANNELS[i], "kernel": KERNELS,
                       "block": BLOCKS, "activation": ACTIVATIONS}[key]
            child["stages"][i][key] = rng.choice(choices)
        if arch_key(child) != arch_key(arch):
            return child
    return child


def arch_key(arch):
    """Canonical string used to de-duplicate architectures."""
    return json.dumps(arch, sort_keys=True)


def arch_summary(arch):
    """Compact human-readable encoding, e.g. r32-s24|2x32k3dwsep-relu|..."""
    st = "|".join(f"{s['depth']}x{s['channels']}k{s['kernel']}{s['block']}-{s['activation']}"
                  for s in arch["stages"])
    return f"r{arch['resolution']}-s{arch['stem_channels']}|{st}"


def search_space_size():
    per_stage = [len(DEPTHS) * len(c) * len(KERNELS) * len(BLOCKS) * len(ACTIVATIONS)
                 for c in STAGE_CHANNELS]
    total = 0
    for n in NUM_STAGES:
        p = 1
        for i in range(n):
            p *= per_stage[i]
        total += p
    return total * len(RESOLUTIONS) * len(STEM_CHANNELS)
