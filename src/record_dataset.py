from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
#import lerobot.utils.visualization_utils as viz_utils
from lerobot.scripts.lerobot_record import record_loop
from lerobot.processor import make_default_processors

camera_index = 1
NUM_EPISODES = 20
FPS = 30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 10
TASK_DESCRIPTION = "LeCamera"
DATASET_PTH = TASK_DESCRIPTION.lower().replace(" ", "_")

leader_id = "leader_qqq"
follower_id = "follower_qqq"

leader_port  = "COM4"
follower_port = "COM3"

hf_user = "jqhisme"
# Create robot configuration
robot_config = SO101FollowerConfig(
    id=follower_id,
    cameras={
        "front": OpenCVCameraConfig(index_or_path=camera_index, width=640, height=480, fps=FPS) # Optional: fourcc="MJPG" for troubleshooting OpenCV async error.
    },
    port=follower_port,
)

teleop_config = SO101LeaderConfig(
    id=leader_id,
    port=leader_port,
)

# Initialize the robot and teleoperator
robot = SO101Follower(robot_config)
teleop = SO101Leader(teleop_config)

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id=f"{hf_user}/lerobot-{DATASET_PTH}-dataset",
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

# Initialize the keyboard listener and rerun visualization
_, events = init_keyboard_listener()
init_rerun(session_name="recording")

# Connect the robot and teleoperator
robot.connect()
teleop.connect()

# Create the required processors
teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

episode_idx = 0
while episode_idx < NUM_EPISODES and not events["stop_recording"]:
    log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
        teleop=teleop,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
    )

    # Reset the environment if not stopping or re-recording
    if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
        log_say("Reset the environment")
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            teleop=teleop,
            control_time_s=RESET_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
        )

    if events["rerecord_episode"]:
        log_say("Re-recording episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    dataset.save_episode()
    episode_idx += 1

# Clean up
log_say("Stop recording")
robot.disconnect()
teleop.disconnect()
dataset.push_to_hub()