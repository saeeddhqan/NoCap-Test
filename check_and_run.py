import os
import time
import subprocess
from datetime import datetime, timedelta

# ==== CONFIGURATION ====
LOG_FILE = "pylog124M/44109974-4272-4478-b583-b06ed3890766.log"  # Replace with your log file path
TIMEOUT_MINUTES = 30
SCRIPT_TO_RUN = "./run.sh"
CHECK_INTERVAL_SECONDS = 300  # Check every 5 minutes

def log_modified_within(path, minutes):
    try:
        mtime = os.path.getmtime(path)
        last_modified = datetime.fromtimestamp(mtime)
        return datetime.now() - last_modified < timedelta(minutes=minutes)
    except FileNotFoundError:
        print(f"[ERROR] Log file not found: {path}")
        return True  # Avoid triggering script if file doesn't exist

def main():
    while True:
        if not log_modified_within(LOG_FILE, TIMEOUT_MINUTES):
            print(f"[INFO] No change in log for {TIMEOUT_MINUTES} minutes. Running {SCRIPT_TO_RUN}")
            subprocess.run(SCRIPT_TO_RUN, shell=True)
            break
        else:
            print(f"[INFO] Training still active. Log modified within last {TIMEOUT_MINUTES} minutes.")
        time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()