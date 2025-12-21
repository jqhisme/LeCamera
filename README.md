# LeCamera
Gonna Capture Them all \
![Demo Video](./Documentation/DemoVideo.mp4) \
LeCamera is a dynamic human tracking system designed for documenting creative projects such as kinetic installations and immersive environments. It combines affordable robotics and computer vision to autonomously follow a human subject, providing diverse visual perspectives without a camera operator.

## Features

- **YOLO + Inverse Kinematics:** Analytical human detection and smooth camera tracking.
- **Imitation Learning:** End-to-end policies (ACT, SmolVLM) trained from human demonstrations for expressive and human-like motion.
- **Human-in-the-loop data collection:** Uses the LeRobot library for robot control and dataset creation.

## Usage

Clone the repository:
```bash
git clone https://github.com/yourusername/lecamera.git
cd lecamera
```

Run Inference Using Policy
```
python yolo_follow.py #yolo+inverse kinematics
python src/control_act.py # run using ACT policy
python src/control_smolvlm.py # run using smolVLM policy
```

Run on eval dataset
```
cd src
python eval.py
```

Record Dataset
```
cd src
python record_dataset.py
```

