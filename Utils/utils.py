import os
import json
import shutil
import cv2

def extract_frames(video_path, output_path, interval=1):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    # Get the frames per second (fps) of the input video
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Get the width and height of the frames
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    frame_count = 0

    while True:
        # Read the next frame
        ret, frame = video_capture.read()

        # Break the loop if we have reached the end of the video
        if not ret:
            break

        # Save the frame as an image
        output_filename = os.path.join(output_path, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(output_filename, frame)

        # Display the frame every 'interval' frames
        # if frame_count % interval == 0:
        #     cv2.imshow('Frame', frame)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break

        frame_count += 1

    # Release the video capture object
    video_capture.release()

    # Close the OpenCV window
    cv2.destroyAllWindows()

def count_frames(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get the total number of frames
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Release the video file
    video.release()

    # Return the number of frames
    return total_frames


def extract_frames_2(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Save frame as an image
        frame_path = output_folder + "frame_{:04d}.jpg".format(frame_count)
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()


def create_video_from_images(image_dir, output_video, fps=10):
    # Get the list of image file names in the directory
    image_files = sorted(os.listdir(image_dir), key=lambda x: int(x.split('.')[0]))

    # Read the first image to get its dimensions
    first_image_path = os.path.join(image_dir, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, channels = first_image.shape

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Choose the desired video codec
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Iterate over the image files and write them to the video
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        video_writer.write(image)
        cv2.imshow('Video', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video writer and close the OpenCV windows
    video_writer.release()
    cv2.destroyAllWindows()

def update_annotation_file_name(json_file):

    # Read the JSON file
    with open(json_file, "r") as f:
        data = json.load(f)

    # Update the "file_name" values
    for image_info in data["images"]:
        file_name = image_info["file_name"]
        image_info["file_name"] = os.path.basename(file_name)

    # Write the updated JSON back to the file
    with open(json_file, "w") as f:
        json.dump(data, f)

    print(" file_name replaced successfully")

def update_roboflow_annos_images_file_name(image_dir_path):

    images = os.listdir(image_dir_path)

    for image in images:

        if not image.endswith(".json"):

            old_file_name = os.path.join(image_dir_path, image)
            new_file_name = os.path.join(image_dir_path, image.split('_')[0] + '.jpg')

            os.rename(old_file_name, new_file_name)

    print("done!")

def rename_files_and_update_json(json_file, image_folder):

    # Read the JSON file
    with open(json_file, "r") as f:
        data = json.load(f)

    # Update the "file_name" values in the JSON file and rename image files
    for image_info in data["images"]:

        old_file_name = image_info["file_name"]

        # Create a new file name based on the image ID
        new_file_name = old_file_name.split("_")[0] + ".jpg"

        # Rename the image file in the folder
        old_file_path = os.path.join(image_folder, old_file_name)
        new_file_path = os.path.join(image_folder, new_file_name)
        os.rename(old_file_path, new_file_path)

        # Update the "file_name" value in the JSON file
        image_info["file_name"] = new_file_name

    # Write the updated JSON back to the file
    with open(json_file, "w") as f:
        json.dump(data, f)

    print("done")

def rename_files_and_update_json_2(json_file, image_folder):

    # Read the JSON file
    with open(json_file, "r") as f:
        data = json.load(f)

    i = 1
    # Update the "file_name" values in the JSON file and rename image files
    for image_info in data["images"]:

        old_file_name = image_info["file_name"]

        # Create a new file name based on the image ID
        new_file_name = str(i) + ".jpg"

        # Rename the image file in the folder
        old_file_path = os.path.join(image_folder, old_file_name)
        new_file_path = os.path.join(image_folder, new_file_name)
        os.rename(old_file_path, new_file_path)

        # Update the "file_name" value in the JSON file
        image_info["file_name"] = new_file_name
        
        i=i+1

    # Write the updated JSON back to the file
    with open(json_file, "w") as f:
        json.dump(data, f)

    print("done")


def replace_category_ids_in_json(annotations_file, coco_categories_file):
    # Load the COCO categories
    with open(coco_categories_file, "r") as f:
        coco_categories = json.load(f)

    # Load your annotations
    with open(annotations_file, "r") as f:
        annotations = json.load(f)

    # Create a custom data mapping
    custom_data_categorie_mapping = {
        categorie['id']: categorie['name']
        for categorie in annotations["categories"]
    }

    print("+"*50)
    print(custom_data_categorie_mapping)

    # Create a mapping between your category names and COCO category IDs
    coco_category_mapping = {
        coco_category["name"]: coco_category["id"]
        for coco_category in coco_categories
    }

    print("+"*50)
    print(coco_category_mapping)

    # Replace the category IDs in annotations
    for annotation in annotations["annotations"]:
        your_category_id = annotation["category_id"]
        your_category_name = custom_data_categorie_mapping[your_category_id]
        coco_category_id = coco_category_mapping[your_category_name]
        annotation["category_id"] = coco_category_id

    # Replace the category IDs in categories
    for category in annotations["categories"]:
        your_category_id = category["id"]
        your_category_name = custom_data_categorie_mapping[your_category_id]
        coco_category_id = coco_category_mapping[your_category_name]
        category["id"] = coco_category_id

    # Save the updated annotations
    with open(annotations_file, "w") as f:
        json.dump(annotations, f)

    print("done.............!!!")



def filter_frames(input_folder, output_folder):

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok= True)

    # Get a list of all files in the input folder
    #files = os.listdir(input_folder)
    files = sorted(os.listdir(input_folder), key=lambda x: int(x.split('.')[0]))

    # iterate ove the files and copy every second frame to the output folder
    for i, file in enumerate(files):
        if i%2 != 0:
            # Construct the input and output file paths
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file)

            # Copy the file to the output folder
            shutil.copy2(input_path, output_path)

    print("Frames copied successfully!")



def skip_frames(input_folder, output_folder, frame_ranges, frames_to_skip):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over the frame ranges
    for frame_range, frames_skip in zip(frame_ranges, frames_to_skip):
        start_frame, end_frame = frame_range

        # Iterate over the frames in the range
        for frame_number in range(start_frame, end_frame + 1):
            # Check if the frame needs to be skipped
            if frame_number % (frames_skip + 1) != 0:
                continue

            # Construct the input and output file paths
            input_path = os.path.join(input_folder, f"{frame_number}.png")
            output_path = os.path.join(output_folder, f"{frame_number}.png")

            # Copy the frame to the output folder
            shutil.copy2(input_path, output_path)

    print("Frames copied successfully!")





if __name__ == "__main__":

    # # Specify the input and output folder paths
    # input_folder = "/home/yalamaku/Documents/Detectron_2/Dataset/ZED_cam_data/data/images_without_frame_split/"
    # output_folder = "/home/yalamaku/Documents/Detectron_2/Dataset/ZED_cam_data/data/skipped_frames_data/train/"

    # # Specify the frame ranges and frames to skip for each range
    # frame_ranges = [(1, 64), (65, 128), (129, 194), (195, 266), (267, 585),
    #                 (589, 735), (736, 1000), (1001, 1220), (1221, 1590), (1590, 1622)]
    # frames_to_skip = [10, 1, 2, 10, 2, 10, 2, 0, 2, 10]

    # # Call the function to skip frames and store selected frames in the output folder
    # skip_frames(input_folder, output_folder, frame_ranges, frames_to_skip)

    #filter_frames(input_folder= input_path, output_folder= output_path)

    # update_annotation_file_name("/home/yalay_dataset/sample_dataset/val/val_gt/annotations.json")

    # annotations_file = "/home/yalamaku/Documents/Detectron_2/Dataset/ZED_cam_data/annotations/panoptic_seg_annotation_with_coco_classes/coco_detection.json"
    # coco_categories_file = "/home/yalamaku/Documents/Detectron_2/Dataset/ZED_cam_data/annotations/panoptic_coco_categories.json"

    # annotations_file = "/home/yalamaku/Documents/Detectron_2/Dataset/ZED_cam_data/robo_flow_augumented_annotations/train_gt/train/coco_detection.json"
    # cityscapes_categorie_files = "/home/yalamaku/Documents/Detectron_2/Dataset/ZED_cam_data/robo_flow_augumented_annotations/train_gt/Categories.json"

    # replace_category_ids_with_coco(annotations_file, cityscapes_categorie_files)

    # update_roboflow_annos_images_file_name("/home/yalamaku/Documents/Detectron_2/Dataset/ZED_camera_dataset/Roboflow_annotated_data/CustomDataset_with_CityscapesLabels/valid")

    # json_file = "/home/yalamaku/Documents/Detectron_2/Dataset/ZED_camera_dataset/Roboflow_annotated_data/CustomDataset_With_CityscapesLabels/valid/_annotations.coco.json"
    # image_folder = "/home/yalamaku/Documents/Detectron_2/Dataset/ZED_camera_dataset/Roboflow_annotated_data/CustomDataset_With_CityscapesLabels/valid"
    # rename_files_and_update_json(json_file= json_file, image_folder= image_folder)

    for folder in ['test', 'train', 'valid']:
        json_file = f"/home/yalamaku/Documents/Thesis/Dataset_files/ZED_camera_dataset/Roboflow_annotated_data/RPTU-university-Data/{folder}/{folder}/_annotations.coco.json"
        image_folder = f"/home/yalamaku/Documents/Thesis/Dataset_files/ZED_camera_dataset/Roboflow_annotated_data/RPTU-university-Data/{folder}/{folder}"

        print(image_folder)
        print(json_file)

        rename_files_and_update_json_2(
            json_file= json_file,
            image_folder= image_folder
        )

        

        replace_category_ids_in_json(
            annotations_file= json_file,
            coco_categories_file= "/home/yalamaku/Documents/Thesis/Dataset_files/ZED_camera_dataset/Roboflow_annotated_data/Categories.json"
        )


