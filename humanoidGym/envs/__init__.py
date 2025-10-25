from humanoidGym import GYM_ROOT_DIR, GYM_ENVS_DIR

from .long.long_uneven_ori_config import LongOriUnevenRoughCfg,LongOriUnevenRoughCfgPPO
from .g1.g1_config import G1AmpCfg,G1AmpCfgPPO
from .g1.g1_uneven_config import G1UnevenAmpCfg,G1UnevenAmpCfgPPO


from .lite.lite_config import LiteAmpCfg, LiteAmpCfgPPO
from .lite.lite_lstm_config import LiteAmpLSTMCfg, LiteAmpCfgLSTMPPO

from .long.long_uneven_env import LongUnevenRobot
from .g1.g1_amp_env import AmpG1Robot
from .g1.g1_height_env import AmpG1HeightRobot
from .g1.g1_height_config import G1HeightAmpCfgPPO,G1HeightAmpCfg
from .g1.g1_image_env import AmpG1ImageRobot
from .lite.lite_amp_env import LiteRobot
from .g1.g1_uneven_with_stair_config import G1UnevenStairAmpCfg,G1UnevenStairAmpCfgPPO

from .base.legged_robot import LeggedRobot

from humanoidGym.utils.task_registry import task_registry

task_registry.register("long_uneven_ori",LongUnevenRobot,LongOriUnevenRoughCfg(),LongOriUnevenRoughCfgPPO())
task_registry.register("g1_amp",AmpG1Robot,G1AmpCfg(),G1AmpCfgPPO())
task_registry.register("g1_image_amp",AmpG1ImageRobot,G1UnevenAmpCfg(),G1UnevenAmpCfgPPO())
task_registry.register("g1_uneven_stair_amp",AmpG1Robot,G1UnevenStairAmpCfg(),G1UnevenStairAmpCfgPPO())
task_registry.register("g1_height_amp",AmpG1HeightRobot,G1HeightAmpCfg(),G1HeightAmpCfgPPO())
task_registry.register("lite_amp",LiteRobot,LiteAmpCfg(),LiteAmpCfgPPO())
task_registry.register("lite_lstm_amp",LiteRobot,LiteAmpLSTMCfg(),LiteAmpCfgLSTMPPO())


