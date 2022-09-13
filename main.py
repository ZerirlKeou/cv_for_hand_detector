import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import yaml
import socket
import math
import numpy as np
import pandas as pd


def cmd(i):
    file = pd.read_csv("cmd.csv")
    df = pd.DataFrame(file)
    # print(df)
    docu = df[i:i + 1]
    # print(docu)
    # print(docu['output'][i])
    list_cmd = [docu['input'][i], docu['output'][i]]
    return list_cmd


def loop_data():
    while True:
        success, img = cap.read()
        # img = cv2.flip(img, 1)
        # hands, img = detector.findHands(img, flipType=False)
        hands, img = detector.findHands(img)
        # 创建21个手部坐标（x，y，z）列表传入unity
        data = []
        if hands:
            # 对于检测的第一只手
            hand = hands[0]
            lmList = hand['lmList']
            # print(lmList)
            for lm in lmList:
                # unity中y坐标与cv中相反，传入列表翻转y轴
                data.extend([lm[0], datay['settings'][3]['height'] - lm[1], lm[2]])
            # print(data)
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((datay['cap_img_setting'][1]['imgSize'], datay['cap_img_setting'][1]['imgSize'], 3),
                               np.uint8) * 255

            imgCrop = img[y - datay['cap_img_setting'][0]['offset']:y + h + datay['cap_img_setting'][0]['offset'],
                      x - datay['cap_img_setting'][0]['offset']:x + w + datay['cap_img_setting'][0]['offset']]
            # height.width,channel
            imgCropShape = imgCrop.shape

            aspectRatio = h / w
            try:
                if aspectRatio > 1:
                    k = datay['cap_img_setting'][1]['imgSize'] / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, datay['cap_img_setting'][1]['imgSize']))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((datay['cap_img_setting'][1]['imgSize'] - wCal) / 2)
                    # height,width
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite)
                    sock.sendto(str.encode(str(cmd(index))), serverAddressPort_camera)
                    # print(prediction, cmd(index))
                else:
                    k = datay['cap_img_setting'][1]['imgSize'] / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (datay['cap_img_setting'][1]['imgSize'], hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((datay['cap_img_setting'][1]['imgSize'] - hCal) / 2)
                    # height,width
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite)
                    sock.sendto(str.encode(str(cmd(index))), serverAddressPort_camera)
                    # print(prediction, cmd(index))
            except:
                pass
            sock.sendto(str.encode(str(data)), serverAddressPort_data)

        # img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
        cv2.imshow("hand_detector", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    with open("enviro.yaml", encoding="utf-8") as yaml_file:
        datay = yaml.safe_load(yaml_file)
    # print(type(data['settings'][3]['height']))
    # print(data['settings'][3]['height'])
    cap = cv2.VideoCapture(0)
    cap.set(datay['settings'][0]['id3'], datay['settings'][2]['width'])
    cap.set(datay['settings'][1]['id4'], datay['settings'][3]['height'])

    detector = HandDetector(maxHands=datay['hands'][0]['max_num'], detectionCon=datay['hands'][1]['confidence'])
    classifier = Classifier(datay['path'][0]['model_h5'], datay['path'][1]['model_txt'])

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    serverAddressPort_data = (datay['server_data'][0]['address'], datay['server_data'][1]['port'])
    serverAddressPort_camera = (datay['server_camera'][0]['address'], datay['server_camera'][1]['port'])
    loop_data()