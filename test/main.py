import numpy as np
import cv2
import time
from mss import mss
from PIL import Image

championsInGame = ["Viego"]
frame_cnt = 0
min_threshold = 0.15
max_threshold = 0.8

# Load champ icons
icons = []
for champion in championsInGame:
    template = cv2.resize(cv2.imread("../champion_images_cropped/" + champion + ".png"), (30, 30), interpolation=cv2.INTER_LINEAR)
    icons.append((champion, template))

championBBox = {}
for champion in championsInGame:
    championBBox[champion] = None

cap = cv2.VideoCapture("minimap.mp4")
fps, st, frames_to_count, cnt = (0, 0, 20, 0)
while cap.isOpened():
    ret, frame = cap.read()

    if cnt == frames_to_count:
        try:
            fps = round(frames_to_count / (time.time() - st))
            st = time.time()
            cnt = 0
        except:
            pass
    cnt += 1

    frame = cv2.putText(frame, 'FPS: ' + str(fps), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Threshold of blue in BGR space
    #lower_color_threshold = np.array([150, 100, 0])
    #upper_color_threshold = np.array([255, 165, 90])

    # Threshold of red in BGR space
    lower_color_threshold = np.array([10, 0, 130])
    upper_color_threshold = np.array([100, 100, 230])

    mask = cv2.inRange(frame, lower_color_threshold, upper_color_threshold)
    filtered_frame = cv2.bitwise_and(frame, frame, mask=mask)

    kernel = np.ones((5, 5), np.uint8)
    blurred = cv2.medianBlur(filtered_frame, 1)
    closing = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("closing", closing)
    gray = cv2.cvtColor(closing, cv2.COLOR_BGR2GRAY)

    champFound = 0
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=30, param2=10, minRadius=14, maxRadius=18)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            x, y = (i[0], i[1])
            radius = i[2]
            cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)

            c_x1 = int(max(x - float(radius) - 5, 0))
            c_y1 = int(max(y - float(radius) - 5, 0))
            c_x2 = min(c_x1 + 2 * radius + 5, 405)
            c_y2 = min(c_y1 + 2 * radius + 5, 405)

            cropped_frame = frame[c_y1:c_y2, c_x1:c_x2]

            threshold = max_threshold
            find = False
            while not find and threshold > min_threshold:
                for champion, template in icons:
                    res = cv2.matchTemplate(cropped_frame, template, cv2.TM_CCOEFF_NORMED)
                    loc = np.where(res >= threshold)
                    if len(loc[0]) > 0:
                        find = True
                        champFound += 1
                    for pt in zip(*loc[::-1]):
                        championBBox[champion] = ((c_x1, c_y1), (c_x2, c_y2), frame_cnt)
                threshold -= 0.05

    if champFound > 0:
        if champFound > 5 and min_threshold < max_threshold:
            #min_threshold += 0.02
            print("min_threshold :", min_threshold)

    for champion in championBBox.keys():
        if championBBox[champion] is not None:
            if frame_cnt - championBBox[champion][2] < 5:
                cv2.rectangle(frame, championBBox[champion][0], championBBox[champion][1],
                              (255, 0, 0, 255), 2)
                cv2.putText(frame, champion, championBBox[champion][0], cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 0, 0, 255), 2)

    frame_cnt += 1

    # Display the resulting frame
    cv2.imshow('Minimap Viewer', frame)
    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cv2.destroyAllWindows()
