import numpy as np
from skimage import io
import argparse
import os

DATA_DIRECTORY = "./reports/pspnet_imagenet"
FIELD_NAMES = ["name", "size", "river", "background",
               'mask_roi_13', 'mask_roi_12', 'river_roi_13', 'river_roi_12']
# IMAGE_CROP = True


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Count Number of Pixels Assigned to Each Class."
    )
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source of data.")
    parser.add_argument("--field-names", nargs='+', default=FIELD_NAMES,
                        help="List of the headers in .csv report.")
    # parser.add_argument("--image-crop", , default=IMAGE_CROP,
    #                     help="List of the headers in .csv report.")

    return parser.parse_args()


args = get_arguments()


def csv_writer(input_list, output_path, names_list=None):
    import csv
    if names_list is None:
        names_list = list()

    csv_path = os.path.join(output_path, "output_roi.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=names_list)

        csv_writer.writeheader()
        for item in input_list:
            csv_writer.writerow(item)


def pixel_counter(masks_dir):
    items_list = list()
    for root, dirs, files in os.walk(masks_dir, topdown=True):
        for image in files:
            if image.endswith(".png"):
                mask = io.imread(os.path.join(root, image))
                mask_roi_12 = mask[:, int(mask.shape[1] - mask.shape[1] / 2):]
                mask_roi_13 = mask[:, int(mask.shape[1] - mask.shape[1] / 3):]
                items_list.append({
                    "name": image,
                    "size": mask.shape,
                    "river": np.sum(mask == 255),
                    "background": np.sum(mask == 0),
                    "mask_roi_12": mask_roi_12.shape,
                    "river_roi_12": np.sum(mask_roi_12 == 255),
                    "mask_roi_13": mask_roi_13.shape,
                    "river_roi_13": np.sum(mask_roi_13 == 255),
                })

    return items_list


def main(data_dir, names_list):
    csv_writer(pixel_counter(data_dir), data_dir, names_list)


if __name__ == "__main__":
    main(args.data_dir, args.field_names)
