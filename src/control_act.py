import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower

import sys
import random
from PIL import Image

MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 100
hf_user = "jqhisme"
taskname = "lecamera"
camera_index = 1
FPS = 30
all_actions = []


def main():
    device = torch.device("cuda")  # or "cuda" or "cpu"
    pretrained_model_path = "models\ACT\pretrained_model"
    model = ACTPolicy.from_pretrained(pretrained_model_path)

    dataset_id = "jqhisme/lerobot-lecamera-dataset-filtered"
    # This only downloads the metadata for the dataset, ~10s of MB even for large-scale datasets
    dataset_metadata = LeRobotDatasetMetadata(dataset_id)
    preprocess, postprocess = make_pre_post_processors(model.config, dataset_stats=dataset_metadata.stats)

    # # find ports using lerobot-find-port
    follower_port = 'COM3'  # something like "/dev/tty.usbmodem58760431631"

    # # the robot ids are used the load the right calibration files
    follower_id = "follower_qqq" # something like "follower_so100"

    # Robot and environment configuration
    # Camera keys must match the name and resolutions of the ones used for training!
    # You can check the camera keys expected by a model in the info.json card on the model card on the Hub
    
    robot_config = SO101FollowerConfig(
    id=follower_id,
    cameras={
        "front": OpenCVCameraConfig(index_or_path=camera_index, width=640, height=480, fps=FPS) # Optional: fourcc="MJPG" for troubleshooting OpenCV async error.
    },
    port=follower_port,
)
    robot = SO101Follower(robot_config)
    robot.connect()
    img_index = 0
    for _ in range(MAX_EPISODES):
        for _ in range(MAX_STEPS_PER_EPISODE):
            obs = robot.get_observation()
            obs_frame = build_inference_frame(
                observation=obs, ds_features=dataset_metadata.features, device=device
            )

            obs = preprocess(obs_frame)
            if random.random() < 0.2:
                captured_img =  obs['observation.images.front'].cpu().numpy()[0].transpose(1,2,0)
                # scale to 0-255 
                captured_img = (captured_img.clip(0, 1) * 255).astype('uint8')
                Image.fromarray(captured_img).save(f"results/act/act_captured_image_{img_index}.png")
                img_index += 1

            action = model.select_action(obs)
            action = postprocess(action)

            action = make_robot_action(action, dataset_metadata.features)
            all_actions.append(action)
            robot.send_action(action)

        print("Episode finished! Starting new episode...")
    robot.disconnect()

    # write all_actions to csv
    keys = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
    with open("results/act/actions.csv", "w") as f:
        f.write(",".join([key for key in keys]) + "\n")
        for action in all_actions:
            action_str = ",".join([str(action[key]) for key in keys])
            f.write(action_str + "\n")

if __name__ == "__main__":
    main()
    