import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from soma_retargeter.robotics.morphology import analyze_mjcf_morphology
from soma_retargeter.robotics.reachability import (
    orthonormal_basis_from_jacobian,
    project_relative_rotation_quat_xyzw,
    quat_xyzw_to_rotation_vector,
    rotation_vector_to_quat_xyzw,
)
from soma_retargeter.robotics.retarget_profile import stable_json_dumps, validate_legacy_scaler_for_v2
from soma_retargeter.robotics.task_compiler import compile_retarget_profile


class TestRetargetProfileV2(unittest.TestCase):
    def test_legacy_scaler_validation_rejects_negative_scale_for_v2(self):
        warnings = validate_legacy_scaler_for_v2({"joint_scales": {"RightHand": -0.5, "LeftHand": 1.0}})
        self.assertEqual(warnings[0]["code"], "non_positive_scale")
        self.assertEqual(warnings[0]["joint"], "RightHand")

    def test_reachability_projection_removes_unreachable_pitch_roll(self):
        basis, _, rank = orthonormal_basis_from_jacobian(np.array([[0.0], [0.0], [2.0]]))
        self.assertEqual(rank, 1)
        source = rotation_vector_to_quat_xyzw(np.array([0.2, -0.3, 0.4]))
        projected = project_relative_rotation_quat_xyzw(source, basis)
        rotvec = quat_xyzw_to_rotation_vector(projected)
        self.assertAlmostEqual(rotvec[0], 0.0, places=7)
        self.assertAlmostEqual(rotvec[1], 0.0, places=7)
        self.assertAlmostEqual(rotvec[2], 0.4, places=7)

    def test_compile_profile_is_deterministic_and_disables_hand_orientation(self):
        mjcf = """<mujoco><worldbody><body name="base_link"><joint name="waist" type="hinge" axis="0 0 1" range="-1 1"/><body name="left_elbow_yaw_link"/></body></worldbody></mujoco>"""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "robot.xml"
            path.write_text(mjcf)
            morphology = analyze_mjcf_morphology(path)
            raw = {"ik_map": {"Hips": "base_link", "LeftHand": "left_elbow_yaw_link"}}
            first = compile_retarget_profile(robot_name="fixture", raw_config=raw, morphology=morphology)
            second = compile_retarget_profile(robot_name="fixture", raw_config=raw, morphology=morphology)

        self.assertEqual(json.loads(stable_json_dumps(first)), json.loads(stable_json_dumps(second)))
        self.assertFalse(first.semantic_sites["LeftHand"].orientation_supported)
        hand_chain = first.chains["LeftHand"]
        self.assertEqual(hand_chain.rotational_rank, 0)
        self.assertEqual(hand_chain.rotational_basis.shape, (3, 0))

    def test_chain_profile_uses_mjcf_path_lengths_and_axis_rank(self):
        mjcf = """
        <mujoco>
          <worldbody>
            <body name="base_link">
              <body name="torso_link" pos="0 0 0.25">
                <joint name="waist_pitch" type="hinge" axis="0 1 0" range="-1 1"/>
                <body name="left_arm_link" pos="0 0.2 0"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "robot.xml"
            path.write_text(mjcf)
            morphology = analyze_mjcf_morphology(path)
            raw = {"ik_map": {"Hips": "base_link", "Chest": "torso_link", "LeftArm": "left_arm_link"}}
            profile = compile_retarget_profile(robot_name="fixture", raw_config=raw, morphology=morphology)

        chest_chain = profile.chains["Chest"]
        self.assertEqual(chest_chain.joint_names, ["waist_pitch"])
        self.assertEqual(chest_chain.rotational_rank, 1)
        self.assertTrue(np.allclose(np.abs(chest_chain.rotational_basis[:, 0]), [0.0, 1.0, 0.0]))
        self.assertAlmostEqual(chest_chain.total_length, 0.25)
        self.assertEqual(profile.tasks[0].rotation_mask_or_basis, [[0.0], [1.0], [0.0]])

        arm_chain = profile.chains["LeftArm"]
        self.assertEqual(arm_chain.joint_names, [])
        self.assertAlmostEqual(arm_chain.total_length, 0.2)

    def test_middle_limb_absolute_position_is_not_default_task(self):
        morphology = analyze_mjcf_morphology(None)
        raw = {"ik_map": {"LeftForeArm": "left_forearm"}}
        profile = compile_retarget_profile(robot_name="fixture", raw_config=raw, morphology=morphology)
        self.assertEqual(profile.tasks[0].task_type, "direction")
        self.assertEqual(profile.tasks[0].reference_site, "LeftArm")
        self.assertIsNone(profile.tasks[0].position_mask_or_basis)

    def test_middle_limb_triplet_generates_pole_vector_task(self):
        mjcf = """
        <mujoco>
          <worldbody>
            <body name="left_arm">
              <body name="left_forearm" pos="1 0 0">
                <body name="left_hand" pos="0 1 0"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "robot.xml"
            path.write_text(mjcf)
            morphology = analyze_mjcf_morphology(path)
            raw = {
                "ik_map": {
                    "LeftArm": "left_arm",
                    "LeftForeArm": "left_forearm",
                    "LeftHand": "left_hand",
                }
            }
            profile = compile_retarget_profile(robot_name="fixture", raw_config=raw, morphology=morphology)

        pole = next(task for task in profile.tasks if task.name == "LeftForeArm_pole_vector")
        self.assertEqual(pole.task_type, "pole_vector")
        self.assertEqual(pole.reference_site, "LeftArm")
        self.assertEqual(pole.source_semantic, "LeftForeArm")
        self.assertEqual(pole.target_site, "LeftHand")
        self.assertTrue(pole.enabled)

    def test_collision_proxies_and_pairs_are_written_from_geoms(self):
        mjcf = """
        <mujoco>
          <worldbody>
            <body name="base">
              <body name="torso" pos="0 0 1.0">
                <geom name="torso_sphere" type="sphere" size="0.2"/>
              </body>
              <body name="left_hand" pos="0.5 0.3 0.8">
                <geom name="left_hand_sphere" type="sphere" size="0.05"/>
              </body>
              <body name="right_hand" pos="0.5 -0.3 0.8">
                <geom name="right_hand_sphere" type="sphere" size="0.05"/>
              </body>
              <body name="left_leg" pos="0 0.1 0.4">
                <geom name="left_leg_box" type="box" size="0.05 0.05 0.2"/>
              </body>
              <body name="right_leg" pos="0 -0.1 0.4">
                <geom name="right_leg_box" type="box" size="0.05 0.05 0.2"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "robot.xml"
            path.write_text(mjcf)
            morphology = analyze_mjcf_morphology(path)
            raw = {
                "ik_map": {
                    "Chest": "torso",
                    "LeftHand": "left_hand",
                    "RightHand": "right_hand",
                    "LeftLeg": "left_leg",
                    "RightLeg": "right_leg",
                }
            }
            profile = compile_retarget_profile(robot_name="fixture", raw_config=raw, morphology=morphology)

        self.assertEqual(morphology.summary()["geom_count"], 5)
        collision = profile.collision
        self.assertTrue(collision["enabled"])
        self.assertEqual(collision["margin"], 0.03)
        self.assertGreaterEqual(len(collision["proxies"]), 5)
        pairs = {(pair["a"], pair["b"]) for pair in collision["pairs"]}
        self.assertIn(("Chest", "LeftHand"), pairs)
        self.assertIn(("LeftHand", "RightLeg"), pairs)
        chest_proxy = next(proxy for proxy in collision["proxies"] if proxy["semantic"] == "Chest")
        self.assertEqual(chest_proxy["source"], "geom_bounds")
        self.assertAlmostEqual(chest_proxy["radius"], 0.2)

    def test_collision_config_disables_safely_without_proxies(self):
        morphology = analyze_mjcf_morphology(None)
        raw = {"ik_map": {"Chest": "torso"}}
        profile = compile_retarget_profile(robot_name="fixture", raw_config=raw, morphology=morphology)
        self.assertFalse(profile.collision["enabled"])
        self.assertTrue(any(warning["code"] == "collision_proxy_disabled" for warning in profile.warnings))


if __name__ == "__main__":
    unittest.main()
