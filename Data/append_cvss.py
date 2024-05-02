import json
import requests
import threading
import time

FILE_PATH = "./cisa.json"

def append_cvss_score(vulnerability):
    try:
        res = requests.get(f"https://cve.circl.lu/api/cve/{vulnerability['cveID']}")
        res.raise_for_status()
        cvss = res.json().get('cvss', {})
        if cvss is None:
            vulnerability['cvss'] = ""
        else:
            vulnerability['cvss'] = cvss
    except requests.exceptions.RequestException as e:
        print("Error appending CVSS score:", e)
    except AttributeError as e:
        print("Error appending CVSS score:", e)

def output_to_json(updated_vulnerabilities, output_file):
    try:
        with open(output_file, 'w') as file:
            json.dump(updated_vulnerabilities, file)
        print(f"Updated vulnerabilities saved to {output_file}")
    except IOError as e:
        print("Error writing to JSON file:", e)
        
def fetch_local_json(file_path):
    try:
        with open(file_path, 'r') as file:
            json_data = json.load(file)
            vulnerabilities = json_data.get("vulnerabilities", [])
            return vulnerabilities
    except IOError as e:
        print("Error reading local JSON file:", e)
        return []
    
if __name__ == "__main__":
    
    vulnerabilities = fetch_local_json(FILE_PATH)
    
    max_threads = 50  # Set the maximum number of threads
    timeout = 10  # Set the timeout for threads in seconds

    # Create threads for appending CVSS scores
    threads = []
    for vulnerability in vulnerabilities:
        thread = threading.Thread(target=append_cvss_score, args=(vulnerability,))
        threads.append(thread)
        thread.start()

        # Wait for the maximum number of threads to be reached
        while threading.active_count() >= max_threads:
            time.sleep(0.5)

    # Wait for all threads to complete
    for thread in threads:
        thread.join(timeout)

    output_to_json(vulnerabilities, "./cvss_vulnerabilities.json")