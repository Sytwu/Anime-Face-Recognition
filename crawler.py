import argparse
import json
import os
import requests
import time
from tqdm import tqdm

def main(config_file, character_name):
    with open(config_file, "r", encoding="utf-8") as f:
        configs = json.load(f)

    config = configs.get(character_name)
    if not config:
        print(f"Configuration for character '{character_name}' not found.")
        return

    tag = config.get("tag")
    download_folder = config.get("download_folder")
    os.makedirs(download_folder, exist_ok=True)

    total_posts_to_retrieve = 1000  # Desired number of posts
    limit = 100  # Maximum number of posts per request
    total_pages = (total_posts_to_retrieve + limit - 1) // limit  # Calculate total pages needed

    total_downloaded = 0

    for page in range(1, total_pages + 1):
        print(f"Downloading page {page}...")
        params = {
            "tags": tag,
            "page": page,
            "limit": limit
        }
        response = requests.get("https://danbooru.donmai.us/posts.json", params=params)
        posts = response.json()

        if not posts:
            break

        for post in tqdm(posts, desc=f"Page {page}", unit="image"):
            file_url = post.get("sample_file_url") or post.get("file_url")
            if file_url:
                image_name = os.path.basename(file_url)
                image_path = os.path.join(download_folder, image_name)
                if os.path.exists(image_path):
                    continue
                response = requests.get(file_url, stream=True)
                total_size = int(response.headers.get('content-length', 0))
                with open(image_path, "wb") as f:
                    for data in tqdm(response.iter_content(1024), total=total_size//1024, unit='KB', desc=image_name):
                        f.write(data)
                time.sleep(1)  # Delay to avoid excessive requests
                total_downloaded += 1

    print(f"Total images downloaded: {total_downloaded}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Danbooru Image Downloader")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--cha", type=str, required=True, help="Character name")
    args = parser.parse_args()

    main(args.config, args.cha)
