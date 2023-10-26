from __future__ import annotations

from os import environ as env
from os import listdir
from os import makedirs
from os import path
from os import rename

import cv2


class ProcessImages:
    IMG_WIDTH = 20
    IMG_HEIGHT = 20

    WIDTH = int(env['IMG_WIDTH']) if 'IMG_WIDTH' in env else IMG_WIDTH
    HEIGHT = int(env['IMG_HEIGHT']) if 'IMG_HEIGHT' in env else IMG_HEIGHT

    RAW_POSITIVE_DIR = env['RAW_POSITIVE_DIR']
    RAW_NEGATIVE_DIR = env['RAW_NEGATIVE_DIR']
    IMG_POSITIVE_DIR = env['IMG_POSITIVE_DIR'].format(
        width=WIDTH, heigth=HEIGHT,
    )
    IMG_NEGATIVE_DIR = env['IMG_NEGATIVE_DIR'].format(
        width=WIDTH, heigth=HEIGHT,
    )

    def __init__(self):
        # Create the directories
        self.__create_directories()

    def __create_directories(self):
        """
        Create the directories to store the processed images.
        """
        # Create the directories
        makedirs(self.RAW_POSITIVE_DIR, exist_ok=True)
        makedirs(self.RAW_NEGATIVE_DIR, exist_ok=True)
        makedirs(self.IMG_POSITIVE_DIR, exist_ok=True)
        makedirs(self.IMG_NEGATIVE_DIR, exist_ok=True)

    def __process_images(self, input_dir, output_dir):
        """
        Process the images.
        Steps are:
            - List the files in the input directory
            - Rename the files to a sequential number like '00000001.jpg'
            - Read the images
            - Crop the images to 1:1 ratio (keep the center of the image)
            - Resize the images to WIDTH x HEIGHT
            - Save the images in the output directory

        :param input_dir: The directory where the raw images are stored.
        :param output_dir: The directory where the processed images will be stored.
        """

        # List the files in the input directory
        files = [
            f for f in listdir(input_dir) if path.isfile(
                path.join(input_dir, f),
            )
        ]

        # Rename the files to a sequential number like '00000001.jpg'
        for i, file in enumerate(files):
            rename(
                path.join(input_dir, file), path.join(
                    input_dir, f'{i:08}.jpg',
                ),
            )

        files = [
            f for f in listdir(input_dir) if path.isfile(
                path.join(input_dir, f),
            )
        ]

        # Read the images
        for file in files:
            img = cv2.imread(path.join(input_dir, file))

            # Crop the images to 1:1 ratio (keep the image centered)
            height, width, channels = img.shape
            if height > width:
                img = img[
                    int((height - width) / 2): int((height + width) / 2), 0:width,
                ]
            elif width > height:
                img = img[
                    0:height, int((width - height) / 2): int((width + height) / 2),
                ]

            # Resize the images to WIDTH x HEIGHT
            img = cv2.resize(
                img, (self.WIDTH, self.HEIGHT),
                interpolation=cv2.INTER_AREA,
            )

            # Save the images in the output directory
            cv2.imwrite(path.join(output_dir, file), img)

    def run(self):
        """
        Process the images.
        """
        # Process the positive images
        self.__process_images(
            input_dir=self.RAW_POSITIVE_DIR, output_dir=self.IMG_POSITIVE_DIR,
        )

        # Process the negative images
        self.__process_images(
            input_dir=self.RAW_NEGATIVE_DIR, output_dir=self.IMG_NEGATIVE_DIR,
        )
