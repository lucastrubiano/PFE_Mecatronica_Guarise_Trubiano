"""
This is the main file of the project.
It will be used to run the different programs to train, test and use the ML models.

The commands are the following:
    - python main.py train [face|eyes|mouth] [stages]
    - python main.py test [face|eyes|mouth]
    - python main.py real_time [face|eyes|mouth]
    - python main.py process_images
"""

import sys
import argparse

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

from src import train, test, real_time, process_images

# Create the argument parser
parser = argparse.ArgumentParser(
    description="Run the different programs to train, test and use the ML models."
)

# Add the arguments
parser.add_argument("command", help="The command to run. It can be train, test, real_time or process_images.")
parser.add_argument(
    "part",
    nargs="?",
    help="The part of the face to train, test or use. It can be face, eyes or mouth.",
)
parser.add_argument(
    "stages",
    nargs="?",
    help="The number of stages to train the haar cascade. It can be any integer number. It is only used when training a haar cascade.",
)

# Parse the arguments
args = parser.parse_args()

# Get the command
command = args.command

# Get the part
part = args.part

# Get the stages
stages = args.stages

# Check if the command is valid
if command not in ["train", "test", "real_time", "process_images"]:
    print("Invalid command. It must be train, test or real_time.")
    sys.exit(1)

# Check if the part is valid
if part not in ["face", "eyes", "mouth"] and command in ["train", "test"]:
    print("Invalid part. It must be face, eyes or mouth.")
    sys.exit(1)
elif part not in ["face", "eyes", "mouth", None] and command == "real_time":
    print(
        "Invalid part. It must be face, eyes or mouth. Or it can be empty to use all the parts."
    )
    sys.exit(1)

# Check if the stages is valid
if stages is not None and command == "train":
    try:
        stages = int(stages)
    except ValueError:
        print("Invalid stages. It must be an integer number.")
        sys.exit(1)

# Run the command
if command == "train":
    obj = train.Train(part, stages)
elif command == "test":
    obj = test.Test(part)
elif command == "real_time":
    obj = real_time.RealTime()
elif command == "process_images":
    obj = process_images.ProcessImages()

obj.run()
