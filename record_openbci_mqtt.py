import paho.mqtt.client as mqtt
import time
import json
import os
import numpy as np
import base64
import struct
import sys

BASE64_BYTES_PER_SAMPLE = 4
SCALE_FACTOR_EMG = 4500000/24/(2**23-1)  # uV/count
EMG_FREQUENCY = 1000  # Hz


def base64_to_list_of_channels(encoded_data, bytes_per_sample):
    """Convert base64 string of samples (on each channel) to list of integers
    """
    output = list()
    for i in range(int(len(encoded_data) / bytes_per_sample)):
        # decode base64
        sample_start = i * bytes_per_sample
        decoded_bytes = base64.b64decode(encoded_data[sample_start:sample_start+bytes_per_sample])

        # convert 24-bit signed int to 32-bit signed int
        decoded = struct.unpack('>i', (b'\0' if decoded_bytes[0] < 128 else b'\xff') + decoded_bytes)
        output.append(decoded)
    return output


def on_message(client, userdata, message):
    """Callback function
    """
    global file_created
    global script_start_time
    global last_sampling_rate_print_time
    global samples_counter
    global gestures_classes
    global current_gesture_idx
    global current_gesture_start_time
    global gest_duration
    global time_for_gest_change

    # Calculate duration of the current gesture
    curr_gest_dur = time.time() - current_gesture_start_time

    # Change a gesture if time for it elapsed and say it
    if curr_gest_dur > gest_duration:
        current_gesture_idx += 1
        current_gesture_start_time = time.time()
        if current_gesture_idx >= len(gestures_classes):
            # Stop subscriber loop
            print("Recording ends, unsubscribing...")
            client.loop_stop()

            print("Program ends...")
            sys.exit()

        # Tell user about the gesture
        os.system(f'spd-say "{gestures_classes[current_gesture_idx]}"')

    # Save data if time for transition elapsed
    if curr_gest_dur > time_for_gest_change:

        message_as_string = str(message.payload.decode("utf-8"))
        message_as_string = ''.join(message_as_string.split())  # remove all white characters

        # Decode the JSON
        packet = json.loads(message_as_string)

        # Read number of channels
        num_channels = packet["channels"]

        # Add a number of samples to the counter
        samples_counter += packet["packets"]

        for encoded_channels in packet["data"]:
            if len(encoded_channels) == num_channels * BASE64_BYTES_PER_SAMPLE:
                channels_list = base64_to_list_of_channels(encoded_channels, BASE64_BYTES_PER_SAMPLE)
                scaled_data = np.asarray(channels_list).T * SCALE_FACTOR_EMG
                # TODO add to the data info about gesture
                with open(f"{data_directory}/{int(script_start_time)}/X.csv", 'a') as f:
                    np.savetxt(f, scaled_data, delimiter=',')
                    #print("Packet saved to file, time: " + str(time.time() - script_start_time))
                with open(f"{data_directory}/{int(script_start_time)}/y.csv", 'a') as f:
                    one_hot_label = np.asarray(
                        [g == gestures_classes[current_gesture_idx] for g in gestures_classes])
                    np.savetxt(f, one_hot_label.reshape(-1, len(gestures_classes)), delimiter=',')

        # Print the number of samples received
        time_elapsed = time.time() - last_sampling_rate_print_time
        if time_elapsed >= 1.0:
            print("***************************")
            print("Samples received: " + str(samples_counter))
            print("Time from the last print: " + str(time_elapsed))
            print("***************************")
            last_sampling_rate_print_time = time.time()
            samples_counter = 0


# MQTT data
broker_data = {
  "broker_address": "192.168.9.100"
}
topic = "sensors/emg/data"

# File created flag
file_created = False

# Script start time
script_start_time = time.time()

# The time of last sampling rate print
last_sampling_rate_print_time = script_start_time

# List of gestures
gestures_classes = ['idle', 'fist', 'flexion', 'extension', 'pinch_index', 'pinch_middle',
                    'pinch_ring', 'pinch_small']

# Current gesture
current_gesture_idx = 0
os.system(f'spd-say "{gestures_classes[current_gesture_idx]}"')

# Time of current gesture beginning
current_gesture_start_time = time.time()

# Duration of one gesture and time for change
gest_duration = 5
time_for_gest_change = 1

# Samples counter
samples_counter = 0

# Create a dir for data
session_time = time.localtime()
data_directory = "recorded_signals"
if not os.path.exists(data_directory):
    os.mkdir(data_directory)
if not os.path.exists(f"{data_directory}/{int(script_start_time)}"):
    os.mkdir(f"{data_directory}/{int(script_start_time)}")

# Create a client
print("Creating new subscriber instance")
client = mqtt.Client("emg_sub")

# Type username and password before connecting to broker
# client.username_pw_set(broker_data['username'], broker_data['password'])

# Connect with broker
print("Connecting to broker")
client.connect(broker_data['broker_address'])

# Subscribe to topic
print("Subscribing to topic")
client.subscribe(topic)

# Attach function to callback
client.on_message = on_message

try:
    # Start the loop
    client.loop_forever()

except KeyboardInterrupt:
    # Stop subscriber loop
    client.loop_stop()
