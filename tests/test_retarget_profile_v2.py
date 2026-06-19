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

    def test_distal_hand_site_uses_geom_bounds_offset(self):
        mjcf = """
        <mujoco>
          <worldbody>
            <body name="left_forearm">
              <body name="left_hand" pos="0.4 0 0">
                <geom name="palm" type="box" pos="0.05 0 0" size="0.1 0.03 0.02"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "robot.xml"
            path.write_text(mjcf)
            morphology = analyze_mjcf_morphology(path)
            raw = {"ik_map": {"LeftForeArm": "left_forearm", "LeftHand": "left_hand"}}
            profile = compile_retarget_profile(robot_name="fixture", raw_config=raw, morphology=morphology)

        site = profile.semantic_sites["LeftHand"]
        self.assertEqual(site.source, "geom_bounds")
        self.assertTrue(np.allclose(site.local_position, [0.15, 0.0, 0.0]))
        self.assertAlmostEqual(profile.chains["LeftHand"].total_length, 0.55)

    def test_distal_hand_site_prefers_explicit_mjcf_site_over_geom_bounds(self):
        mjcf = """
        <mujoco>
          <worldbody>
            <body name="left_forearm">
              <body name="left_hand" pos="0.4 0 0">
                <site name="left_hand_tip" pos="0.3 0.01 0" quat="0.70710678 0 0 0.70710678"/>
                <geom name="palm" type="box" pos="0.05 0 0" size="0.1 0.03 0.02"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "robot.xml"
            path.write_text(mjcf)
            morphology = analyze_mjcf_morphology(path)
            raw = {"ik_map": {"LeftForeArm": "left_forearm", "LeftHand": "left_hand"}}
            profile = compile_retarget_profile(robot_name="fixture", raw_config=raw, morphology=morphology)

        self.assertEqual(morphology.summary()["site_count"], 1)
        site = profile.semantic_sites["LeftHand"]
        self.assertEqual(site.source, "explicit_site")
        self.assertTrue(np.allclose(site.local_position, [0.3, 0.01, 0.0]))
        self.assertTrue(np.allclose(site.local_rotation_xyzw, [0.0, 0.0, 0.70710678, 0.70710678]))
        self.assertAlmostEqual(profile.chains["LeftHand"].total_length, 0.7001666203960727)

    def test_distal_hand_site_uses_child_anchor_before_geom_bounds(self):
        mjcf = """
        <mujoco>
          <worldbody>
            <body name="left_arm">
              <body name="left_forearm" pos="0.4 0 0">
                <geom name="forearm_geom" type="box" pos="0.05 0 0" size="0.1 0.02 0.02"/>
                <body name="left_hand_tip" pos="0.3 0.02 0"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "robot.xml"
            path.write_text(mjcf)
            morphology = analyze_mjcf_morphology(path)
            raw = {"ik_map": {"LeftForeArm": "left_arm", "LeftHand": "left_forearm"}}
            profile = compile_retarget_profile(robot_name="fixture", raw_config=raw, morphology=morphology)

        site = profile.semantic_sites["LeftHand"]
        self.assertEqual(site.source, "inferred_child")
        self.assertTrue(np.allclose(site.local_position, [0.3, 0.02, 0.0]))
        self.assertAlmostEqual(profile.chains["LeftHand"].total_length, 0.7006659275674582)

    def test_distal_hand_site_uses_mjcf_mesh_bounds_offset(self):
        mesh = """
        solid hand
          facet normal 0 0 1
            outer loop
              vertex 0 0 0
              vertex 0.2 0.04 0
              vertex 0.2 -0.04 0
            endloop
          endfacet
        endsolid hand
        """
        mjcf = """
        <mujoco>
          <compiler meshdir="meshes"/>
          <asset>
            <mesh name="hand_mesh" file="hand.stl"/>
          </asset>
          <worldbody>
            <body name="left_forearm">
              <body name="left_hand" pos="0.4 0 0">
                <geom type="mesh" mesh="hand_mesh"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "meshes").mkdir()
            (root / "meshes" / "hand.stl").write_text(mesh)
            path = root / "robot.xml"
            path.write_text(mjcf)
            morphology = analyze_mjcf_morphology(path)
            raw = {"ik_map": {"LeftForeArm": "left_forearm", "LeftHand": "left_hand"}}
            profile = compile_retarget_profile(robot_name="fixture", raw_config=raw, morphology=morphology)

        self.assertEqual(morphology.summary()["geom_count"], 1)
        site = profile.semantic_sites["LeftHand"]
        self.assertEqual(site.source, "geom_bounds")
        self.assertTrue(np.allclose(site.local_position, [0.2, 0.0, 0.0]))
        self.assertAlmostEqual(profile.chains["LeftHand"].total_length, 0.6)

    def test_mesh_bounds_and_fingerprint_refresh_when_mesh_changes(self):
        mjcf = """
        <mujoco>
          <compiler meshdir="meshes"/>
          <asset>
            <mesh name="hand_mesh" file="hand.stl"/>
          </asset>
          <worldbody>
            <body name="left_forearm">
              <body name="left_hand" pos="0.4 0 0">
                <geom type="mesh" mesh="hand_mesh"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """

        def mesh_with_tip(x):
            return f"""
            solid hand
              facet normal 0 0 1
                outer loop
                  vertex 0 0 0
                  vertex {x} 0.04 0
                  vertex {x} -0.04 0
                endloop
              endfacet
            endsolid hand
            """

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "meshes").mkdir()
            mesh_path = root / "meshes" / "hand.stl"
            robot_path = root / "robot.xml"
            robot_path.write_text(mjcf)

            mesh_path.write_text(mesh_with_tip(0.2))
            first_morphology = analyze_mjcf_morphology(robot_path)
            first = compile_retarget_profile(
                robot_name="fixture",
                raw_config={"ik_map": {"LeftForeArm": "left_forearm", "LeftHand": "left_hand"}},
                morphology=first_morphology,
            )

            mesh_path.write_text(mesh_with_tip(0.4))
            second_morphology = analyze_mjcf_morphology(robot_path)
            second = compile_retarget_profile(
                robot_name="fixture",
                raw_config={"ik_map": {"LeftForeArm": "left_forearm", "LeftHand": "left_hand"}},
                morphology=second_morphology,
            )

        self.assertNotEqual(first_morphology.robot_fingerprint, second_morphology.robot_fingerprint)
        self.assertTrue(np.allclose(first.semantic_sites["LeftHand"].local_position, [0.2, 0.0, 0.0]))
        self.assertTrue(np.allclose(second.semantic_sites["LeftHand"].local_position, [0.4, 0.0, 0.0]))

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

    def test_root_ground_metadata_records_scale_and_ground_source(self):
        mjcf = """
        <mujoco>
          <worldbody>
            <body name="pelvis" pos="0 0 1.0">
              <body name="left_foot" pos="0.1 0.1 -1.0"/>
              <body name="right_foot" pos="0.1 -0.1 -1.0"/>
            </body>
          </worldbody>
        </mujoco>
        """
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "robot.xml"
            path.write_text(mjcf)
            morphology = analyze_mjcf_morphology(path)
            raw = {
                "ik_map": {"Hips": "pelvis", "LeftFoot": "left_foot", "RightFoot": "right_foot"},
                "root_motion": {"source_leg_length_m": 0.5},
                "ground_barrier": {"ground_height": -0.02},
            }
            profile = compile_retarget_profile(robot_name="fixture", raw_config=raw, morphology=morphology)

        root_motion = profile.rest_frame_alignment["root_motion"]
        self.assertEqual(root_motion["source"], "semantic_hips_feet_rest_pose")
        self.assertAlmostEqual(root_motion["robot_leg_length_m"], np.sqrt(1.01), places=6)
        self.assertAlmostEqual(root_motion["source_leg_length_m"], 0.5)
        self.assertAlmostEqual(root_motion["horizontal_scale"], np.sqrt(1.01) / 0.5, places=6)
        self.assertAlmostEqual(root_motion["robot_nominal_pelvis_height_m"], 1.0)
        self.assertAlmostEqual(root_motion["ground_height_m"], -0.02)
        self.assertEqual(root_motion["ground_height_source"], "explicit_ground_barrier")
        self.assertAlmostEqual(profile.segment_ratios["leg_length"], root_motion["horizontal_scale"])

    def test_root_ground_metadata_warns_when_leg_length_unavailable(self):
        morphology = analyze_mjcf_morphology(None)
        raw = {"ik_map": {"Hips": "pelvis"}}
        profile = compile_retarget_profile(robot_name="fixture", raw_config=raw, morphology=morphology)
        root_motion = profile.rest_frame_alignment["root_motion"]
        self.assertEqual(root_motion["source"], "fallback")
        self.assertEqual(root_motion["ground_height_source"], "default_world_z0")
        self.assertTrue(any(warning["code"] == "root_leg_length_unavailable" for warning in profile.warnings))


if __name__ == "__main__":
    unittest.main()
