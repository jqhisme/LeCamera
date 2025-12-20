from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.processor import make_default_processors
import os

camera_index = 1
NUM_EPISODES = 20
FPS = 30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 10
TASK_DESCRIPTION = "LeCameraEval"
DATASET_PTH = TASK_DESCRIPTION.lower().replace(" ", "_")
hf_user = "jqhisme"
pretrained_model_path = "models\ACT\pretrained_model"

# Create the robot configuration
camera_config = {"front": OpenCVCameraConfig(index_or_path=camera_index, width=640, height=480, fps=FPS)}
robot_config = SO101FollowerConfig(
    port="COM3", id="follower_qqq", cameras=camera_config
)

# Initialize the robot
robot = SO101Follower(robot_config)

# Initialize the policy
policy = ACTPolicy.from_pretrained(pretrained_model_path)

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Create the dataset
# check if C:\Users\jiang\.cache\huggingface\lerobot\jqhisme\lerobot-lecameraeval-dataset-eval exists
# if yes,remove it (this is for testing, so it does not matter)
os_path = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "lerobot", hf_user, f"lerobot-{DATASET_PTH}-dataset-eval")
if os.path.exists(os_path):
    import shutil
    shutil.rmtree(os_path)
dataset = LeRobotDataset.create(
    repo_id=f"{hf_user}/lerobot-{DATASET_PTH}-dataset-eval",
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

# Initialize the keyboard listener and rerun visualization
_, events = init_keyboard_listener()
init_rerun(session_name="recording")

# Connect the robot
robot.connect()

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy,
    pretrained_path=pretrained_model_path,
    dataset_stats=dataset.meta.stats,
)



for episode_idx in range(NUM_EPISODES):
    log_say(f"Running inference, recording eval episode {episode_idx + 1} of {NUM_EPISODES}")

    # Run the policy inference loop
    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
    )

    dataset.save_episode()

# Clean up
robot.disconnect()
dataset.push_to_hub()