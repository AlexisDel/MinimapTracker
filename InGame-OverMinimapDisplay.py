import time
import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk
from mss import mss


class App:
    def __init__(self, championsInGame):
        self.window = tk.Tk()
        self.window.overrideredirect(True)
        self.window.geometry('405x405+1516+676')
        self.window.lift()
        self.window.wm_attributes("-topmost", True)
        self.window.wm_attributes("-disabled", True)
        self.window.wm_attributes("-transparentcolor", "white")

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(championsInGame)

        # Create a canvas that can fit the above video source size
        self.window.image = tk.PhotoImage(file='background.png')
        self.labelImage = tk.Label(self.window, image=self.window.image, bg='white')
        self.labelImage.pack()

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 1
        self.update()
        self.window.mainloop()

    def update(self):
        # Get a frame from the video source
        frame = self.vid.get_frame()
        img = Image.fromarray(frame, 'RGBA')
        self.photo = ImageTk.PhotoImage(image=img)
        self.labelImage.configure(image=self.photo)
        self.window.after(self.delay, self.update)


class MyVideoCapture:

    def __init__(self, championsInGame):
        self.frame = 0
        self.fps = 0
        self.st = 0
        self.frames_to_count = 20
        self.cnt = 0
        self.min_threshold = 0.2
        self.max_threshold = 0.8

        # Load champ icons
        self.icons = []
        for champion in championsInGame:
            template = cv2.resize(cv2.imread("champion_images_cropped/" + champion + ".png"), (30, 30),
                                  interpolation=cv2.INTER_LINEAR)
            self.icons.append((champion, template))

        self.background = Image.open("background.png")

        self.championBBox = {}
        for champion in championsInGame:
            self.championBBox[champion] = None

    def get_frame(self):
        screenShot = mss().grab({'left': 1515, 'top': 675, 'width': 405, 'height': 405})
        img = Image.frombytes('RGB', (screenShot.width, screenShot.height), screenShot.rgb, )
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

        if self.cnt == self.frames_to_count:
            try:
                self.fps = round(self.frames_to_count / (time.time() - self.st))
                self.st = time.time()
                self.cnt = 0
            except:
                pass
        self.cnt += 1

        # Returned Image
        minimap = np.array(self.background.copy())

        # minimap = cv2.putText(minimap, 'FPS: ' + str(self.fps), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0, 255), 1)

        # Threshold of blue in BGR space
        lower_blue = np.array([150, 100, 0])
        upper_blue = np.array([255, 165, 90])
        mask = cv2.inRange(frame, lower_blue, upper_blue)
        filtered_frame = cv2.bitwise_and(frame, frame, mask=mask)

        kernel = np.ones((5, 5), np.uint8)
        blurred = cv2.medianBlur(filtered_frame, 1)
        closing = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
        gray = cv2.cvtColor(closing, cv2.COLOR_BGR2GRAY)

        champFound = 0
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=30, param2=10, minRadius=14, maxRadius=18)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                x, y = (i[0], i[1])
                radius = i[2]
                cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)

                c_x1 = int(max(x - float(radius), 0))
                c_y1 = int(max(y - float(radius), 0))
                c_x2 = min(c_x1 + 2 * radius, 405)
                c_y2 = min(c_y1 + 2 * radius, 405)

                cropped_frame = frame[c_y1:c_y2, c_x1:c_x2]

                threshold = self.max_threshold
                find = False
                while not find and threshold > self.min_threshold:
                    for champion, template in self.icons:
                        res = cv2.matchTemplate(cropped_frame, template, cv2.TM_CCOEFF_NORMED)
                        loc = np.where(res >= threshold)
                        if len(loc[0]) > 0:
                            find = True
                            champFound += 1
                        for pt in zip(*loc[::-1]):
                            self.championBBox[champion] = ((c_x1, c_y1), (c_x2, c_y2), self.frame)
                    threshold -= 0.05

        if champFound > 0:
            if champFound > 5 and self.min_threshold < self.max_threshold:
                self.min_threshold += 0.02
                print("min_threshold :", self.min_threshold)

        for champion in self.championBBox.keys():
            if self.championBBox[champion] is not None:
                if self.frame - self.championBBox[champion][2] < 5:
                    cv2.rectangle(minimap, self.championBBox[champion][0], self.championBBox[champion][1],
                                  (255, 0, 0, 255), 2)
                    cv2.putText(minimap, champion, self.championBBox[champion][0], cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 0, 0, 255), 2)

        self.frame += 1

        return minimap


if __name__ == '__main__':
    championsInGame = ["Sejuani", "Qiyana", "Ahri", "Kalista", "Amumu"]
    App(championsInGame)
