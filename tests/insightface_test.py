import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis


app_l = FaceAnalysis(name='buffalo_l', 
                     root='./', 
                    #  providers=['CPUExecutionProvider'], 
                    #  providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], 
                     )

app_l.prepare(ctx_id=0, det_size=(640, 640))


app_sc = FaceAnalysis(name='buffalo_sc', 
                     root='./', 
                    #  providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], 
                    )

app_sc.prepare(ctx_id=0, det_size=(640, 640))

# img = cv2.imread('./test_image_1.jpg')
img = cv2.imread('./test_image_2.jpg')

results_l = app_l.get(img)

print(results_l)

print(len(results_l))

print(results_l[0].keys())

print(results_l[0]['det_score'])

img_copy = img.copy()

gender_encode = ["Male", "Female"]

# get bounding box for each face
for face in results_l:
    
    x1, y1, x2, y2 = face["bbox"].astype(int)

    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

    kps = face["kps"].astype(int)
    for k1, k2 in kps:
        cv2.circle(img_copy, (k1, k2), 3, (0, 0, 255), -1)

    score = f"score: {int(face["det_score"] * 100)}%"
    cv2.putText(img_copy, score, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255),2)

    gender = gender_encode[face["gender"]]
    age = face["age"]

    age_gender = f"{age}::{gender}"

    cv2.putText(img_copy, age_gender, (x1, y2+10), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255),2)


cv2.imshow('bbox', img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()



# from insightface.data import get_image as ins_get_image

# app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# app.prepare(ctx_id=0, det_size=(640, 640))
# img = ins_get_image('t1')
# faces = app.get(img)
# rimg = app.draw_on(img, faces)
# cv2.imwrite("./t1_output.jpg", rimg)