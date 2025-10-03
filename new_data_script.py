import time
import argparse
import os
import requests
from PIL import Image
import io

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def setup_requests_session():
    """Configure a robust requests session with retries"""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session


def load_new_fires(data_dict, fire_name, t_0):
    '''
    Function to load images for new fire. It takes the t_0 timestamp corresponding to the start time of the fire.
    This function will download 40 images before t_0 and 40 images after t_0.
    Total 81 images will be downloaded.

    :param data_dict: dictionary with key as timestamp and value as image url
    :param fire_name: fire name to create save as, example- fire_name = "miramar_fire"
    :param t_0: timestamp for start of fire, example - t_0 = "2025-01-13 14:12:42"
    :return:
    '''

    start_time = time.time()

    def download_and_save_image(session, url, image_name, fire_name):
        try:
            # Add timeout and use persistent session
            response = session.get(url, timeout=10)
            response.raise_for_status()

            with Image.open(io.BytesIO(response.content)) as img:
                camera_name = url.split('/')[4]
                save_dir = os.path.join(fire_name, camera_name)
                os.makedirs(save_dir, exist_ok=True)

                save_path = os.path.join(save_dir, f"{image_name}.jpg")
                img.save(save_path)
                print(f"Saved: {save_path}")

            # Explicit cleanup
            del response
            return True

        except Exception as e:
            print(f"Failed {url}: {str(e)[:100]}")  # Truncate long error messages
            return False



    t_0_index = list(data_dict.keys()).index(t_0)

    session = setup_requests_session()
    # Read and save 40 images before t0
    for i in range(40, 0, -1):
        if t_0_index - i >= 0:
            secs = f"{i * 60:05d}"
            image_name = f"img_-{secs}"
            download_and_save_image(session,url=data_dict[list(data_dict.keys())[t_0_index - i]],
                                    image_name=image_name,
                                    fire_name=fire_name)
        time.sleep(0.1)

        # Read and save t0 image
    download_and_save_image(session,url=data_dict[list(data_dict.keys())[t_0_index]],
                            image_name="img_+00000",
                            fire_name=fire_name)


    # Read and save 40 images after t0
    for i in range(1, 41):
        if t_0_index + i < len(data_dict):
            secs = f"{i * 60:05d}"
            image_name = f"img_+{secs}"
            download_and_save_image(session,url=data_dict[list(data_dict.keys())[t_0_index + i]],
                                            image_name=image_name,
                                            fire_name=fire_name)
        time.sleep(0.1)
    end_time = time.time()

    print("========= Done ==========")
    print(f" Time taken: {end_time - start_time}")

    return


def parse_arguments():
    parser = argparse.ArgumentParser(description="Download wildfire camera images around a specified time.")

    parser.add_argument(
        "--fire_name",
        type=str,
        required=True,
        help="Name of the wildfire event (e.g., 'wildfire_2025')"
    )

    parser.add_argument(
        "--t0",
        type=str,
        required=True,
        help="Reference timestamp in format 'YYYY-MM-DD HH:MM:SS' (e.g., '2025-01-13 14:12:42')"
    )

    parser.add_argument(
        "--filename",
        type=str,
        required=True,
        help="Path to the input text file containing image URLs (e.g., 'tdlln-e-mobo-c_1736803320-1736809080.txt')"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    print(f"Fire Name: {args.fire_name}")
    print(f"t0: {args.t0}")
    print(f"Input File: {args.filename}")

    # Load data from file
    try:
        with open(args.filename, 'r') as file:
            content = file.readlines()
            data_dict = {}
            for line in content:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    data_dict[parts[0]] = parts[1]
    except FileNotFoundError:
        print(f"Error: File {args.filename} not found.")
        exit(1)

    # Process images
    try:
        t_0_index = list(data_dict.keys()).index(args.t0)
    except ValueError:
        print(f"Error: t0 timestamp {args.t0} not found in the file.")
        exit(1)

    load_new_fires(data_dict, args.fire_name, args.t0)
