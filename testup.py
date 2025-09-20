import cv2
import sys
import os
import sys
import cv2
import numpy as np
import time
from collections import deque, defaultdict
from datetime import datetime, timedelta
# from ultralytics import YOLO
import threading
import pyaudio
import audioop

# Add DLL paths before import
os.add_dll_directory(r"C:/openpose/build/x64/Release")
os.add_dll_directory(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin")
sys.path.append(r"C:/openpose/build/python/openpose/Release")
# from openpose import pyopenpose as op
# from . import pyopenpose as op


import sys
import cv2
import os
from sys import platform
import argparse

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release');
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    # params = dict()
    # params["model_folder"] = "../../../models/"
    # opWrapper = op.WrapperPython(op.ThreadManagerMode.Synchronous)
    # opWrapper.configure(params)
    # opWrapper.execute()
except Exception as e:
    print(e)
    sys.exit(-1)

# Set up OpenPose parameters
params = dict()
params["model_folder"] = "models/"  # Adjust path if needed
params["net_resolution"] = "-1x160"  # Smaller network size = faster

# Start OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Start webcam
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    sys.exit()
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Error: Failed to read frame from camera.")
        break


    frame_count += 1
    if frame_count % 2 != 0:  # Skip every other frame
        continue

    datum = op.Datum()
    datum.cvInputData = frame

    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    if datum.cvOutputData is None:
        print("❌ Warning: OpenPose did not return output data.")
        continue

    cv2.imshow("OpenPose - Webcam", datum.cvOutputData)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



# import cv2

# for i in range(5):
#     cap = cv2.VideoCapture(i)
#     if cap.read()[0]:
#         print(f"Camera index {i} works!")
#         cap.release()
