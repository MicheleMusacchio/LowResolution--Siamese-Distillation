import cv2
import os
from PIL import Image
import random
def images_resize(input_path, output_path, size):
    img = cv2.imread(input_path)
    downscaled_img = cv2.resize(img, (size[1], size[0]))
    cv2.imwrite(output_path, downscaled_img)

def is_image_file(filename):
    # Check if the file has a common image extension
    image_extensions = {'.jpg', '.jpeg', '.png'}
    return any(filename.lower().endswith(ext) for ext in image_extensions)

def resize_dataset(input_folder, output_folder, size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over files and subdirectories in the input folder
    for entry in os.listdir(input_folder):
        entry_path = os.path.join(input_folder, entry)
        output_entry_path = os.path.join(output_folder, entry)

        if os.path.isfile(entry_path) and is_image_file(entry):
            images_resize(entry_path, output_entry_path, size)
        #if it's not a file it goes deeper
        elif os.path.isdir(entry_path):
            #print("\n entry path when is not file", entry_path)
            resize_dataset(entry_path, output_entry_path, size)

#to ignore file like .DS_store
def is_image_file(filename):
    # Check if the file has a common image extension
    image_extensions = {'.jpg', '.jpeg', '.png'}
    return any(filename.lower().endswith(ext) for ext in image_extensions)

def downsampling(input_path, target_size, output_path):
    try:
        im = Image.open(input_path)
        target_size = target_size
        horizontal_scale = im.size[0] / target_size
        vertical_scale = im.size[1] / target_size
        scale = max(horizontal_scale, vertical_scale)
        new_size = (int(im.size[0] / scale), int(im.size[1] / scale))
        im = im.resize(new_size, Image.LANCZOS)
        im.save(output_path)
    #to see if something went wrong
    except Exception as e:
        print(f"Error processing image for {input_path}: {e}")

def process_folder(input_folder, target_size, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over files and subdirectories in the input folder
    for entry in os.listdir(input_folder):
        entry_path = os.path.join(input_folder, entry)
        output_entry_path = os.path.join(output_folder, entry)

        if os.path.isfile(entry_path) and is_image_file(entry):
            #print("entry_path ", entry_path, "\n entry: ", entry)
            downsampling(entry_path, target_size, output_entry_path)
        #if it's not a file it goes deeper
        elif os.path.isdir(entry_path):
            #print("\n entry path when is not file", entry_path)
            process_folder(entry_path, target_size, output_entry_path)

def random_process_folder(input_folder, target_size, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over files and subdirectories in the input folder
    for entry in os.listdir(input_folder):
        entry_path = os.path.join(input_folder, entry)
        output_entry_path = os.path.join(output_folder, "LR-" + entry)

        if os.path.isfile(entry_path) and is_image_file(entry):
            size = random.choice(target_size)
            downsampling(entry_path, size, output_entry_path)
        elif os.path.isdir(entry_path):
            random_process_folder(entry_path, target_size, output_entry_path)


def main():

    desired_size = (224, 224)
    final_size = 64

    input_path = '/Users/giacomo/Desktop/Face-recognition/lfw'
    output_path = '/Users/giacomo/Desktop/prova-224'
    output_path2 = '/Users/giacomo/Desktop/prova-64'

    resize_dataset(input_folder = input_path, output_folder = output_path, size = desired_size)
    process_folder(input_folder=input_path, target_size=final_size, output_folder=output_path2)


if __name__ == "__main__":
    main()
