import cv2
import sys

# 0번 카메라를 지정한다.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    sys.exit()

index = 0

while True: # 카메라실행 무한루프 돌린다.
    ret, frame = cap.read() # 카메라에서 1프레임의 데이터를 받아온다.
    if not ret:
        break
    
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.imshow('frame', frame)
    cv2.putText(frame, 'Hello', (100, 100), font, 1, (255, 0, 0), 2)

    key = cv2.waitKey(1)
    if key == ord('q'): # 키값이 0이면 들어오는값이 q일때까지 멈춤
        break
    elif key == ord('s'):
        cv2.imwrite(f'my_pic_{index}.png', frame) # s값을 누르면 사진을 저장한다.
        index = index + 1


    # frame2 = cv2.Canny(frame, 50, 150) # canny를 쓰면 라인 edge만 따온다

    # cv2.imshow('canny',frame2)
cap.release()
cv2.destroyAllWindows()

# python app.py