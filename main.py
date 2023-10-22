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
import src.utils as utils
# Load environment variables
from dotenv import load_dotenv
#from OLD import landmarks_model

load_dotenv()

from src import train, test, real_time, process_images

@click.group()
def main():
    """
     CLI for run the project
    """

@main.command()
def real_time_system() -> None:
    """
    Run the real time system
    """
    obj = real_time.RealTime()
    obj.run()

@main.command()
def set_fps_device() -> None:
    """
    Calculate FPS
    """
    utils.set_fps()

if __name__ == "__main__":
    main()