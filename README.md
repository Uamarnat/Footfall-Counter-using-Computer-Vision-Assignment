# Footfall-Counter-using-Computer-Vision-Assignment
The assignment was to build an AI based footfall counter that can detect, track, and count people as they enter and exit a specific area in a video, such as a doorway or hallway. In addition to counting, the system also highlights the regions inside that area where the most movement occurs using a simple activity heatmap.  

Steps-
Run the script.  
Draw the ROI manually with mouse, press space.  
Example video is shown in Demo_Video.webm.  

<img src="Screenshot1.png" alt="Screenshot" width="600">  

Tools-  
OpenCV- pip install opencv-python  
YOLOv8-pip install ultralytics  
Weight file yolv8n.pt- https://docs.ultralytics.com/models/yolov8/  
NumPy- pip install numpy  

Working- 
A downloaded YouTube video(https://youtu.be/3FXUw98rrUY?si=oV4icf6k42LDWE7A) is passed through OpenCVs video capture function and using the OpenCVs select ROI function the user can manually draw a rectangular Region of Interest (ROI) on the first frame. This ROI defines the specific area where movement detection, counting, and analysis take place.A NumPy array of zeros of size equal to the ROI is initialized to represent the heatmap. YOLOv8 model is then used to detect and track people in each frame, assigning a unique ID to every detected individual. Only detections labeled as persons (class ID 0) are processed. For each person, the program calculates the bottom-center point of their bounding box, which represents the approximate position of their feet. This point is used to determine whether the person is inside or outside the ROI. A dictionary keeps track of each person’s previous inside/outside status, when someone crosses into the ROI, the entry counter increases and when they leave, the exit counter increases. The number of people currently within the ROI is updated in real time. To track the motion, the YOLO track function is used and it stores the most recent 30 foot positions for each person and connects them with yellow lines, forming short path trails that indicate recent movement. Each time a person’s feet are inside the ROI, a small circle is added to the heatmap, and with each new frame, the heatmap values are multiplied by a decay factor (0.99) to gradually fade older movements. The resulting colorized heatmap is overlaid onto the ROI, where red areas indicate regions of high activity and blue areas indicate low movement. In every frame, the program displays the blue ROI boundary, green bounding boxes around detected people, yellow motion trails, the blended heatmap overlay, and on-screen text showing the total number of entries, exits, and people currently within the ROI.

This code is partially adapted for this specific video, where the camera angle places the footpath or “hallway” at the bottom of the frame. Because of that, the program counts the position of a person’s feet (the bottom of the bounding box) to determine if they are inside the ROI. For videos captured from different camera angles, this can be adjusted to use the center of the bounding box or another suitable point.

One limitation of this approach is that the system registers multiple entries for the same person. This happens when the model temporarily loses detection due to occlusion or noise and then detects the person again, treating them as a new entry even though they were already inside the ROI. This issue can be mitigated by introducing a frame buffer to handle short term missed detections, or by using a more advanced tracking method that assigns permanent IDs to individuals, such as models with reidentification capability.
