import pandas as pd
import re
import os

# Initialize the data dictionary for the DataFrame
directory = "/Users/dentira/anomaly-detection/epilepsy-dataset/physionet.org/files/chbmit/1.0.0"
directories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d)) and d.startswith('chb')]

for dir_name in directories:
    data = {
        "File_names": [],
        "Labels": [],
        "Start_time": [],
        "End_time": []
    }

    # Read the input text file (replace with your actual file path)
    input_file_path = f"/Users/dentira/anomaly-detection/epilepsy-dataset/physionet.org/files/chbmit/1.0.0/{dir_name}/{dir_name}-summary.txt"

    with open(input_file_path, "r") as file:
        input_text = file.read()

    # Split the text into individual file entries based on "File Name" occurrence
    file_entries = input_text.split("\n\n")

    for entry in file_entries:
        # Try to extract the file name
        file_name_match = re.search(r"File Name: (\S+)", entry)
        if file_name_match:
            file_name = file_name_match.group(1)
        else:
            print(f"Skipping entry due to missing or malformed 'File Name': {entry}")
            continue  # Skip this entry if no file name is found
        
        # Extract start time and end time
        start_time_match = re.search(r"File Start Time: (\S+)", entry)
        end_time_match = re.search(r"File End Time: (\S+)", entry)
        if start_time_match and end_time_match:
            start_time_str = start_time_match.group(1)
            end_time_str = end_time_match.group(1)
        else:
            print(f"Skipping entry due to missing or malformed start/end time: {entry}")
            continue
        
        # Convert start and end time to seconds (for simplicity)
        start_time = sum(int(x) * 60 ** i for i, x in enumerate(reversed(start_time_str.split(":"))))
        end_time = sum(int(x) * 60 ** i for i, x in enumerate(reversed(end_time_str.split(":"))))
        
        # Check if there are seizures
        num_seizures_match = re.search(r"Number of Seizures in File: (\d+)", entry)
        if num_seizures_match:
            num_seizures = int(num_seizures_match.group(1))
        else:
            print(f"Skipping entry due to missing 'Number of Seizures' data: {entry}")
            continue
        
        
        if num_seizures > 1:
            # For each seizure, extract start and end times
            for i in range(1, num_seizures + 1):
                seizure_start_time_match = re.search(rf"Seizure {i} Start Time: (\d+) seconds", entry)
                seizure_end_time_match = re.search(rf"Seizure {i} End Time: (\d+) seconds", entry)
                
                if seizure_start_time_match and seizure_end_time_match:
                    seizure_start_time = int(seizure_start_time_match.group(1))
                    seizure_end_time = int(seizure_end_time_match.group(1))
                    label = 1  # This file has a seizure
                else:
                    print(f"Skipping entry due to missing seizure time data: {entry}")
                    continue
                
                # Add the parsed data to the DataFrame for each seizure
                data["File_names"].append(file_name)
                data["Labels"].append(label)
                data["Start_time"].append(seizure_start_time)
                data["End_time"].append(seizure_end_time)

        elif num_seizures == 1:
            seizure_start_time_match = re.search(rf"Seizure\s?1? Start Time: (\d+) seconds", entry)
            seizure_end_time_match = re.search(rf"Seizure\s?1? End Time: (\d+) seconds", entry)
            
            if seizure_start_time_match and seizure_end_time_match:
                seizure_start_time = int(seizure_start_time_match.group(1))
                seizure_end_time = int(seizure_end_time_match.group(1))
                label = 1  # This file has a seizure
        
            else:
                print(f"Skipping entry due to missing seizure time data: {entry}")
                continue
            
            # Add the parsed data to the DataFrame for each seizure
            data["File_names"].append(file_name)
            data["Labels"].append(label)
            data["Start_time"].append(seizure_start_time)
            data["End_time"].append(seizure_end_time)
            # If there are no seizures, we still need to add an entry, but with a label of 0
        else :
            data["File_names"].append(file_name)
            data["Labels"].append(0)  # No seizures
            data["Start_time"].append(0)
            data["End_time"].append(0)

    # Create a DataFrame from the parsed data
    df = pd.DataFrame(data)

    # Ensure the columns are of integer type where necessary
    df["Labels"] = df["Labels"].astype(int)
    df["Start_time"] = df["Start_time"].astype(int)
    df["End_time"] = df["End_time"].astype(int)

    # Save the DataFrame to a CSV file
    output_file_path = f"/Users/dentira/anomaly-detection/epilepsy-dataset/physionet.org/files/chbmit/1.0.0/parsed_labels/{dir_name}_labels.csv"
    df.to_csv(output_file_path, index=False)

    print(f"CSV file for {dir_name} has been created successfully.")
