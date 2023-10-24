"""
This is the main file of the project.
It will be used to run the different programs to train, test and use the ML models.

The commands are the following:
    - python main.py train [face|eyes|mouth] [stages]
    - python main.py test [face|eyes|mouth]
    - python main.py real_time # [face|eyes|mouth]
    - python main.py process_images
"""

import sys
import click
# Load environment variables
from dotenv import load_dotenv
#from OLD import landmarks_model

load_dotenv()

from src import train, test, real_time, process_images, utils

@click.group()
def main():
    """
     CLI for run the project
    """

@main.command()
@click.option("--display-video", is_flag=True, help="Display video")
@click.option("--head-pose", is_flag=True, help="Head pose")
def real_time_system(display_video: bool, head_pose: bool) -> None:
    """
    Run the real time system

    Parameters
    ----------
        - display_video (bool, optional): Display video. Defaults to False.
        - head_pose (bool, optional): Head pose. Defaults to False.
    """
    obj = real_time.RealTime(display_video=display_video, head_pose=head_pose)
    obj.run()

@main.command()
def set_fps_device() -> None:
    """
    Calculate FPS
    """
    utils.set_fps()

if __name__ == "__main__":
    main()