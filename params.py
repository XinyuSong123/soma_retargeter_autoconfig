"""Minimal robot registration table for the local soma-retargeter workspace.

Users only need to register four things here:
1. ROBOT_XML_DICT
2. ROBOT_URDF_DICT
3. RETARGETER_CONFIG_DICT
4. POSE_PAIR_JSON_DICT

Other runtime JSON files are auto-generated on first use beside the
registered retargeter config.
"""

import pathlib


HERE = pathlib.Path(__file__).parent
CONFIG_ROOT = HERE / "soma_retargeter" / "configs"
ROBOT_ROOT = HERE / "assets" / "robots"
RPO_POSE_PAIR_ROOT = CONFIG_ROOT / "roboparty_rpo" / "pose_pairs"

# xml路径
ROBOT_XML_DICT = {
    "e3_v2": ROBOT_ROOT / "e3_v2_synthetic" / "mjcf" / "e3_v2_synthetic.xml",
    "roboparty_rpo": ROBOT_ROOT / "atom01" / "mjcf" / "atom01.xml",
    # unitree_g1 can use Newton's built-in asset when left unregistered here.
}

# urdf路径
ROBOT_URDF_DICT = {
    "roboparty_rpo": ROBOT_ROOT / "atom01" / "urdf" / "atom01.urdf",
}

# 重定向配置路径，把骨架的不同部位映射到机器人对应的link
RETARGETER_CONFIG_DICT = {
    "e3_v2": CONFIG_ROOT / "e3_v2" / "soma_to_e3_v2_retargeter_config.json",
    "unitree_g1": CONFIG_ROOT / "unitree_g1" / "soma_to_g1_retargeter_config.json",
    "unitree_g1_23dof": CONFIG_ROOT / "unitree_g1_23dof" / "soma_to_g1_23dof_retargeter_config.json",
    "roboparty_rpo": CONFIG_ROOT / "roboparty_rpo" / "soma_to_rpo_retargeter_config.json",
}



'''
用于优化config的机器人姿态对的JSON文件路径字典。每个机器人可以有多个姿态对，每个姿态对包含一对soma和目标机器人在该姿态下的对应关系。在UI中可以查看这些姿态对
请手动把机器人的姿态对JSON文件放在对应路径下，并在下面注册它们。示例格式如下：
POSE_PAIR_JSON_DICT["my_robot"] = {
    "t_pose": CONFIG_ROOT / "my_robot" / "pose_pairs" / "my_robot_t_pose.json",
    "natural_down": CONFIG_ROOT / "my_robot" / "pose_pairs" / "my_robot_natural_down.json",
}
可以先只注册T-pose，然后测试一下重定向效果，效果不好再多注册一些其他姿态对。
'''
POSE_PAIR_JSON_DICT = {
    "e3_v2": {},
    "unitree_g1": {},
    "unitree_g1_23dof": {},
    "roboparty_rpo": {
        "t_pose": RPO_POSE_PAIR_ROOT / "roboparty_rpo_t_pose.json",
        "natural_down": RPO_POSE_PAIR_ROOT / "roboparty_rpo_natural_down.json",
        "both_arms_forward": RPO_POSE_PAIR_ROOT / "roboparty_rpo_both_arms_forward.json",
        "both_elbows_forward_90": RPO_POSE_PAIR_ROOT / "roboparty_rpo_both_elbows_forward_90.json",
        "arms_forward_squat_hip_yaw_out_45": (
            RPO_POSE_PAIR_ROOT / "roboparty_rpo_arms_forward_squat_hip_yaw_out_45.json"
        ),
    },
}

# Example:
#
# ROBOT_XML_DICT["my_robot"] = ROBOT_ROOT / "my_robot" / "mjcf" / "world.xml"
# ROBOT_URDF_DICT["my_robot"] = ROBOT_ROOT / "my_robot" / "urdf" / "my_robot.urdf"
# RETARGETER_CONFIG_DICT["my_robot"] = CONFIG_ROOT / "my_robot" / "soma_to_my_robot_retargeter_config.json"
# POSE_PAIR_JSON_DICT["my_robot"] = {
#     "t_pose": CONFIG_ROOT / "my_robot" / "pose_pairs" / "my_robot_t_pose.json",
# }
