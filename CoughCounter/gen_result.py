import os
import re
import pandas as pd

def generate_result_txt(dataframe, output_file='result.txt'):
    with open(output_file, 'w') as f:
        previous_file = None
        total_cough_counter = 0

        for idx in range(len(dataframe)):
            # Extract the audio filename from the path
            audio_path = dataframe.iloc[idx, 0]
            audio_filename = os.path.basename(audio_path).split('_')
            current_file = audio_filename[0]

            # If the file changes, reset the total cough counter and update previous_file
            if previous_file != current_file:
                if previous_file:
                    f.write(f"\nTotal Coughs for {previous_file}: {total_cough_counter}\n\n")
                previous_file = current_file
                total_cough_counter = 0

            # Extract the audio timestamp using regular expressions
            match = re.search(r'(.*?)\.[^.]+$', audio_filename[-1])
            if match:
                audio_timestamp = float(match.group(1))
            else:
                # Handle the case if regex pattern doesn't match
                print(f"Error: Couldn't extract timestamp from '{audio_filename[-1]}'")
                continue

            # Write the audio file name in the first row if it's a new file
            if previous_file != current_file:
                f.write(f"{current_file}\n\n")

            # Write the column headers 'cough number' and 'time stamp' if it's a new file
            if previous_file != current_file:
                f.write("cough number,time stamp\n\n")

            # Extract the cough timestamps
            cough_timestamps = [float(x) for x in dataframe.iloc[idx, -1].strip('[]').split(',')]

            # Increment the total cough counter
            total_cough_counter += len(cough_timestamps)

            # Write the cough numbers and timestamps for the current file
            for i, cough_time in enumerate(cough_timestamps, start=1):
                f.write(f"cough{i}\t{audio_timestamp + cough_time}\n")

        # Write the total coughs for the last file in the dataframe
        if previous_file:
            f.write(f"\nTotal Coughs for {previous_file}: {total_cough_counter}\n\n")

# d = {'Path':['vad_audio/test_10.396999999999998.wav','vad_audio/test_100.4.wav','vad_audio/audio2_0.4.wav','vad_audio/audio2_15.wav','vad_audio/audio3_1.55.wav']
#      ,'Coughs':['[1.1,2.2]','[1.1,2.2,3.3]','[4.0]','[1,2,3,4,5]','[1,2,3]']}
# df = pd.DataFrame(data=d)

df = pd.read_csv('a.csv')



# Call the function with the dataframe containing cough labels
generate_result_txt(df)