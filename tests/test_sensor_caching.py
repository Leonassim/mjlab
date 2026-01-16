"""Tests for sensor per-step caching (issue #492)."""

from __future__ import annotations

import mujoco
import pytest
from conftest import get_test_device

from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg
from mjlab.sim.sim import Simulation, SimulationCfg


@pytest.fixture(scope="module")
def device():
  """Test device fixture."""
  return get_test_device()


def test_sensor_caching_returns_same_object(device, robot_with_floor_xml):
  """Multiple data accesses within a step return the same cached object."""
  entity_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(robot_with_floor_xml)
  )

  raycast_cfg = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(
      size=(0.3, 0.3), resolution=0.15, direction=(0.0, 0.0, -1.0)
    ),
    max_distance=10.0,
  )

  scene_cfg = SceneCfg(
    num_envs=2,
    env_spacing=5.0,
    entities={"robot": entity_cfg},
    sensors=(raycast_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=20)
  sim = Simulation(num_envs=2, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["terrain_scan"]
  sim.step()
  scene.update(dt=0.01)

  # Access data multiple times - should return same cached object.
  data1 = sensor.data
  data2 = sensor.data
  data3 = sensor.data

  assert data1 is data2
  assert data2 is data3


def test_sensor_cache_invalidates_on_update(device, robot_with_floor_xml):
  """After update() is called, next data access recomputes."""
  entity_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(robot_with_floor_xml)
  )

  raycast_cfg = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(
      size=(0.3, 0.3), resolution=0.15, direction=(0.0, 0.0, -1.0)
    ),
    max_distance=10.0,
  )

  scene_cfg = SceneCfg(
    num_envs=2,
    env_spacing=5.0,
    entities={"robot": entity_cfg},
    sensors=(raycast_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=20)
  sim = Simulation(num_envs=2, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["terrain_scan"]
  sim.step()
  scene.update(dt=0.01)

  # Access data (caches it).
  data1 = sensor.data

  # Call update to invalidate cache.
  sensor.update(dt=0.01)

  # Access data again - should be new object.
  data2 = sensor.data

  assert data1 is not data2


def test_sensor_cache_invalidates_on_reset(device, robot_with_floor_xml):
  """After reset() is called, next data access recomputes."""
  entity_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(robot_with_floor_xml)
  )

  raycast_cfg = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(
      size=(0.3, 0.3), resolution=0.15, direction=(0.0, 0.0, -1.0)
    ),
    max_distance=10.0,
  )

  scene_cfg = SceneCfg(
    num_envs=2,
    env_spacing=5.0,
    entities={"robot": entity_cfg},
    sensors=(raycast_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=20)
  sim = Simulation(num_envs=2, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["terrain_scan"]
  sim.step()
  scene.update(dt=0.01)

  # Access data (caches it).
  data1 = sensor.data

  # Call reset to invalidate cache.
  sensor.reset()

  # Access data again - should be new object.
  data2 = sensor.data

  assert data1 is not data2


def test_raycast_sensor_compute_called_once_per_step(device, robot_with_floor_xml):
  """RayCastSensor._perform_raycast is called once per step, not per data access."""
  entity_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(robot_with_floor_xml)
  )

  raycast_cfg = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(size=(0.5, 0.5), resolution=0.1, direction=(0.0, 0.0, -1.0)),
    max_distance=10.0,
  )

  scene_cfg = SceneCfg(
    num_envs=2,
    env_spacing=5.0,
    entities={"robot": entity_cfg},
    sensors=(raycast_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=20)
  sim = Simulation(num_envs=2, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["terrain_scan"]
  sim.step()
  scene.update(dt=0.01)

  # Patch _perform_raycast to count calls.
  original_perform_raycast = sensor._perform_raycast
  call_count = [0]

  def counting_perform_raycast():
    call_count[0] += 1
    return original_perform_raycast()

  sensor._perform_raycast = counting_perform_raycast

  # Access data 10 times.
  for _ in range(10):
    _ = sensor.data

  # Should only call _perform_raycast once.
  assert call_count[0] == 1


def test_sensor_data_fresh_after_physics_step(device, robot_with_floor_xml):
  """After physics step + update(), sensor data reflects new state."""
  entity_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(robot_with_floor_xml)
  )

  raycast_cfg = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(size=(0.0, 0.0), resolution=0.1, direction=(0.0, 0.0, -1.0)),
    max_distance=10.0,
  )

  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=5.0,
    entities={"robot": entity_cfg},
    sensors=(raycast_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=20)
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["terrain_scan"]

  # Initial step.
  sim.step()
  scene.update(dt=0.01)
  data1 = sensor.data
  distance1 = data1.distances[0, 0].item()

  # Body at z=2, floor at z=0, so distance should be ~2m.
  assert abs(distance1 - 2.0) < 0.2

  # Move body up to z=3.
  sim.data.qpos[0, 2] = 3.0
  sim.step()
  scene.update(dt=0.01)

  data2 = sensor.data
  distance2 = data2.distances[0, 0].item()

  # Distance should now be ~3m.
  assert abs(distance2 - 3.0) < 0.2
  assert distance2 > distance1


def test_contact_sensor_caching_with_update(device, falling_box_xml):
  """ContactSensor cache is invalidated by update() and works with stateful updates."""
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(falling_box_xml))

  contact_cfg = ContactSensorCfg(
    name="box_contact",
    primary=ContactMatch(mode="geom", pattern="box_geom", entity="robot"),
    secondary=ContactMatch(mode="geom", pattern="ground_geom", entity="robot"),
    fields=("found", "force"),
    track_air_time=True,
  )

  scene_cfg = SceneCfg(
    num_envs=2,
    env_spacing=3.0,
    entities={"robot": entity_cfg},
    sensors=(contact_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=75)
  sim = Simulation(num_envs=2, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["box_contact"]

  # Run a few steps to get contact.
  for _ in range(10):
    sim.step()
    scene.update(dt=0.01)

  # Access data multiple times - should return same cached object.
  data1 = sensor.data
  data2 = sensor.data
  assert data1 is data2

  # Update should invalidate cache.
  sensor.update(dt=0.01)
  data3 = sensor.data

  assert data1 is not data3


def test_contact_sensor_reset_invalidates_cache(device, falling_box_xml):
  """ContactSensor.reset() invalidates the cache."""
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(falling_box_xml))

  contact_cfg = ContactSensorCfg(
    name="box_contact",
    primary=ContactMatch(mode="geom", pattern="box_geom", entity="robot"),
    secondary=ContactMatch(mode="geom", pattern="ground_geom", entity="robot"),
    fields=("found", "force"),
    track_air_time=True,
  )

  scene_cfg = SceneCfg(
    num_envs=2,
    env_spacing=3.0,
    entities={"robot": entity_cfg},
    sensors=(contact_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=75)
  sim = Simulation(num_envs=2, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["box_contact"]

  # Run a few steps.
  for _ in range(10):
    sim.step()
    scene.update(dt=0.01)

  # Access data (caches it).
  data1 = sensor.data

  # Reset should invalidate cache.
  sensor.reset()
  data2 = sensor.data

  assert data1 is not data2
