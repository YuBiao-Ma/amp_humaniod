from humanoidGym import GYM_ROOT_DIR, GYM_ENVS_DIR

from .long.long_uneven_ori_config import LongOriUnevenRoughCfg,LongOriUnevenRoughCfgPPO
from .g1.g1_config import G1AmpCfg,G1AmpCfgPPO
from .g1.g1_uneven_config import G1UnevenAmpCfg,G1UnevenAmpCfgPPO


from .lite.lite_config import LiteAmpCfg, LiteAmpCfgPPO

from .long.long_uneven_env import LongUnevenRobot
from .g1.g1_amp_env import AmpG1Robot
from .lite.lite_amp_env import LiteRobot

from .base.legged_robot import LeggedRobot

from humanoidGym.utils.task_registry import task_registry

task_registry.register("long_uneven_ori",LongUnevenRobot,LongOriUnevenRoughCfg(),LongOriUnevenRoughCfgPPO())
task_registry.register("g1_amp",AmpG1Robot,G1AmpCfg(),G1AmpCfgPPO())
task_registry.register("g1_uneven_amp",AmpG1Robot,G1UnevenAmpCfg(),G1UnevenAmpCfgPPO())
task_registry.register("lite_amp",LiteRobot,LiteAmpCfg(),LiteAmpCfgPPO())

