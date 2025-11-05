import cv2
from ultralytics import YOLO
import numpy as np

cap = cv2.VideoCapture("People Walking Past the Camera - Free Stock Footage For Commercial Projects.mp4")
ret, frame = cap.read()
#Define roi manully 
roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True )
cv2.destroyWindow("Select ROI")
#heatmap roi
roi_heatmap = np.zeros((int(roi[3]), int(roi[2])), dtype = np.float32)
#Load model
model = YOLO("yolov8n.pt")
#variables
entries =0
exits=0
last_status={}
path={}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    #draw roi
    cv2.rectangle(frame, (int(roi[0]), int(roi[1])), (int(roi[0]+roi[2]), int(roi[1]+roi[3])), (255, 0, 0), 2)
    #detect people
    people =0
    #heatmap decay
    roi_heatmap *= 0.99
    #tracking from yolo model
    results = model.track(frame, persist=True)
    for box in results[0].boxes:
        cls=int(box.cls[0])
        x1,y1,x2, y2 = map(int, box.xyxy[0])
        
        if cls ==0: #class id 0 for person
            #get persion id
            person_id = int(box.id[0]) if box.id else None
            #get previous frame status
            prev_inside = last_status.get(person_id, False)
            #count the number of people
            cx = (x1+x2)//2 #calculate the centerpoint of the bottom of the box to count feet
            cy = y2
            if person_id not in path:
                path[person_id] = []

            path[person_id].append((cx, cy))
            path[person_id] = path[person_id][-30:]

            if len(path[person_id]) > 1:
                cv2.polylines(frame, [np.int32(path[person_id])], isClosed=False, color=(0, 255, 255), thickness=2)
            
            is_inside = roi[0]<cx<roi[0] +roi[2] and roi[1]<cy<roi[1]+roi[3]
            
            if is_inside:
                #coordinates within the roi
                roi_x = int(cx-roi[0])
                roi_y = int(cy - roi[1])
                cv2.circle(roi_heatmap,(roi_x, roi_y),8,1,-1) #update heatmap
                #count the people inside roi
                people +=1
            
            if not prev_inside and is_inside:
                #count the entries
                entries +=1    
            
            elif prev_inside and not is_inside:
                #count the exits
                exits +=1
                #remove people from inside roi count 
                people -=1
            
            last_status[person_id] = is_inside            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
    #adding heatmap color
    if np.max(roi_heatmap)>0:
        roi_color = cv2.applyColorMap(np.uint8(255 *roi_heatmap / np.max(roi_heatmap)), cv2.COLORMAP_JET)
        x, y, w, h = map(int, roi)
        heatmapOnRoi= cv2.addWeighted(frame[y:y+h, x:x+w], 0.6, roi_color, 0.4, 0)
        frame[y:y+h, x:x+w] = heatmapOnRoi
    
    cv2.putText(frame, f"Entries: {entries}", (20,90),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    cv2.putText(frame, f"Exits: {exits}", (20,130),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(frame, f"People in ROI:{people}",(20,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Example", frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"People Remaining in ROI: {people}")
print(f"Total Entries: {entries}")
print(f"Total Exits: {exits}")
