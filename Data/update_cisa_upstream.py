import os
import requests

# Specify the URL of the JSON file and the location where it should be saved
url = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
file_path = "./cisa.json"

def download_or_update_json(url, file_path):
    if os.path.exists(file_path):
        print("File already exists. Checking for updates...")
        # Fetch the file from the URL
        response = requests.get(url)
        # Compare the new file content with the existing file
        with open(file_path, 'r+') as file:
            existing_content = file.read()
            if existing_content != response.text:
                print("Updating the existing file...")
                # Update the file if the content has changed
                file.seek(0)
                file.write(response.text)
                file.truncate()
            else:
                print("The file is up to date.")
    else:
        print("File does not exist. Downloading...")
        # Download the file if it does not exist
        response = requests.get(url)
        with open(file_path, 'w') as file:
            file.write(response.text)
        print("File downloaded.")

# Run the function
download_or_update_json(url, file_path)