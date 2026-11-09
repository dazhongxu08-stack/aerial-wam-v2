"""Run the restored wire, and wait for the spin wheels to center before driving."""
import os
import sys

os.environ["FASTRTPS_DEFAULT_PROFILES_FILE"] = (
    "/home/user/rt/app/experiments/aerial/deploy/fastrtps_udp_only.xml"
)
os.environ["FASTDDS_BUILTIN_TRANSPORTS"] = "UDPv4"
sys.path.insert(0, "/home/user/rt/app/experiments/aerial/deploy")

import importlib.machinery

import steer_center

_BASE = "/home/user/rt/app/experiments/aerial/deploy/yhs_wire.pyc.bak-before-face-low"
_loader = importlib.machinery.SourcelessFileLoader("yhs_wire_restored", _BASE)
_base = _loader.load_module()
steer_center.install(_base)
_orig_yolo = _base._yolo_alive


def _yolo_alive() -> bool:
    """One detector is enough. Each wire start used to leave another 2 GB process."""
    import subprocess

    try:
        found = subprocess.run(
            ["ps", "-eo", "args="],
            capture_output=True,
            text=True,
            check=False,
        )
        if any("/home/user/rt/bin/yolo_world.pyc" in line for line in found.stdout.splitlines()):
            return True
    except OSError:
        pass
    return _orig_yolo()


_base._yolo_alive = _yolo_alive

def _spin(node, executor=None):
    """The stock spin() from this thread stops delivering camera frames."""
    import rclpy
    from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
    runner = executor or SingleThreadedExecutor()
    runner.add_node(node)
    try:
        while rclpy.ok():
            try:
                runner.spin_once(timeout_sec=0.05)
            except ExternalShutdownException:
                return
    finally:
        runner.remove_node(node)
        if executor is None:
            runner.shutdown()

import rclpy
rclpy.spin = _spin
_base.main()
