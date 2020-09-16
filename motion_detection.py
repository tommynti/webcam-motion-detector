import cv2, time, pandas
from datetime import datetime

first_frame = None  # none data type ths python
status_list =[None,None]
times = []
df = pandas.DataFrame(columns=["Start Time","End Time"])
start_time = time.perf_counter()
video=cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    check, frame = video.read()
    status = 0
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame,(21,21),0)

    camera_wait_open = time.perf_counter()-start_time
    if first_frame is None:
        if camera_wait_open > 2:
            first_frame = gray_frame

        continue

    delta_frame = cv2.absdiff(first_frame,gray_frame)
    th_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1] # pixel poy exoun diafora > 30 pernoun timh 255 --> white
    th_frame = cv2.dilate(th_frame, None, iterations=2)

    contours,hierachy = cv2.findContours(th_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    for contour in contours:
        if cv2.contourArea(contour) <10000:
            continue

        status = 1
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)

    status_list.append(status)

    status_list=status_list[-2:]

    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())

    cv2.imshow("Recording", gray_frame)
    cv2.imshow("Delta Frame",delta_frame)
    cv2.imshow("Threshold",th_frame)
    cv2.imshow("Motion Detection",frame)

    key = cv2.waitKey(1)

    if key==ord('q') or key==ord('Q'):
        if status==1:  
            times.append(datetime.now())
        break

print(status_list)
print(times)

for i in range(0,len(times),2):
    df = df.append({"Start Time":times[i],"End Time":times[i+1]},ignore_index=True)

df.to_csv("Motion detection times.csv")

video.release()
cv2.destroyAllWindows()
