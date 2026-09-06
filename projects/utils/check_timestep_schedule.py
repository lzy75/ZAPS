"""检查论文 30-NFE 的 15/10/5 时间步是否与 guided-diffusion 完全一致。"""

import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from modules.zaps_algorithm import build_irregular_timesteps


EXPECTED = [
    0, 24, 48, 71, 95, 119, 143, 166, 190, 214, 238, 262, 285, 309, 333,
    334, 371, 408, 445, 482, 518, 555, 592, 629, 666,
    667, 750, 833, 916, 999,
]


def main():
    actual = build_irregular_timesteps(
        total_steps=1000,
        schedule=(15, 10, 5),
        spacing="linear",
    ).tolist()
    print("期望时间步（升序）:", EXPECTED)
    print("实际时间步（升序）:", actual)
    print("实际反向采样顺序  :", list(reversed(actual)))
    if actual != EXPECTED:
        raise SystemExit("FAIL：当前 15/10/5 时间步与 guided-diffusion 不一致")
    print("PASS：30 个时间步与论文图示及 guided-diffusion space_timesteps 一致")


if __name__ == "__main__":
    main()
