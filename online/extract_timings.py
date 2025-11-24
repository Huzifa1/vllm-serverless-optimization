import argparse
import re
from datetime import datetime

def extract_datetime(log_line):
    """extracts datetime"""
    DATE_PATTERN = re.compile(r"\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}(?:\.\d{3})?")

    match = DATE_PATTERN.search(log_line)
    if match is None:
        return None

    value = match.group()

    # Define the format string that matches the input time string
    time_format = "%m-%d %H:%M:%S.%f" if "." in value else "%m-%d %H:%M:%S"

    try:
        return datetime.strptime(value, time_format)
    except ValueError:
        raise ValueError(
            "Failed converting time value '%s' using format '%s'",
            value,
            time_format,
        )

def extract_time(line, pattern):
    """Extract time from a line using a regex pattern."""
    match = re.search(pattern, line)
    if match:
        return float(match.group(1))
    return 0
    
def extract_test_timings(test_log: str):
    with open(test_log, "r") as f:
        lines = f.readlines()
    
    startup_times = []
    generation_times = []
    wakeup_times = []
    sleep_times = []
    
    for line in lines:
        if "Text generated in" in line:
            duration = extract_time(line, r"Text generated in ([\d.]+) seconds")
            generation_times.append(duration)
            
        elif "Server is ready after" in line:
            duration = extract_time(line, r"Server is ready after ([\d.]+) seconds")
            startup_times.append(duration)
            
        elif "Server is sleeping after" in line:
            duration = extract_time(line, r"Server is sleeping after ([\d.]+) seconds")
            sleep_times.append(duration)
            
        elif "Server is wakeup after" in line:
            duration = extract_time(line, r"Server is wakeup after ([\d.]+) seconds")
            wakeup_times.append(duration)
    
    return {
        "startup_times": startup_times,
        "generation_times": generation_times,
        "wakeup_times": wakeup_times,
        "sleep_times": sleep_times
    }

def main():
    parser = argparse.ArgumentParser(description="Extract timings from server logs")

    parser.add_argument(
        "--test_log",
        type=str,
        required=True,
        help="Path to test log file"
    )
    
    args = parser.parse_args()
    
    extract_test_timings(args.test_log)

if __name__ == "__main__":
    main()