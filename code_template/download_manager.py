import os
import subprocess
from multiprocessing import Pool

def download_file(url):
    """
    Downloads a file from a URL into a specified directory.
    """
    dest_dir = "medmnist-101/weights"
    print(f"Downloading {url} into {dest_dir}")
    # Using wget to download. -P specifies the directory.
    # -nc prevents re-downloading if the file exists.
    command = ["wget", "-nc", "-P", dest_dir, url]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Successfully downloaded {url}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to download {url}. Error: {e.stderr}")
        return False

def main():
    """
    Main function to read URLs and download them in parallel.
    """
    links_file = "medmnist-101/download_weights.py"
    dest_dir = "medmnist-101/weights"

    # Create destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Read URLs from the file
    with open(links_file, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    if not urls:
        print(f"No URLs found in {links_file}")
        return

    # Use a pool of 3 processes to download files
    num_processes = 3
    with Pool(num_processes) as p:
        results = p.map(download_file, urls)

    successful_downloads = sum(1 for r in results if r)
    print(f"\nFinished downloading.")
    print(f"{successful_downloads}/{len(urls)} files downloaded successfully.")

if __name__ == "__main__":
    main() 