import os
import sys
from predict_masks import predict_value


def process_cmd_args():
    """
    Reads CMD args.
    source_folder_name :name of the folder that would be used as a source folder
    destination_folder_name: name of the folder that would be used as a destination folder

    :return: tuple(logo_file_name, destination_folder_name, local_temp_folder, source_folder_name)
    """
    if len(sys.argv) < 3:
        print("Not enough cmd arguments.")

    source_folder_name = sys.argv[1]
    destination_folder_name = sys.argv[2]
    segmentation_model = sys.argv[3]

    return (source_folder_name, destination_folder_name, segmentation_model)


def create_folder(folder_name):
    """
    Creates folder if not it does not exists
    :param folder_name: full folder path
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def main():
    source_folder_name, destination_folder_name, segmentation_model = process_cmd_args()
    create_folder(destination_folder_name)
    predict_value(source_folder_name, destination_folder_name, segmentation_model)

    return None


if __name__ == '__main__':
    main()
