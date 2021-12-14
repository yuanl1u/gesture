import pyrealsense2 as rs
import numpy as np
import cv2
import copy
import tools
import Learn
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
import torch.nn as nn

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# Start streaming
pipeline.start(config)

show = [cv2.imread('number/0.png'),
        cv2.imread('number/1.png'),
        cv2.imread('number/2.png'),
        cv2.imread('number/3.png'),
        cv2.imread('number/4.png'),
        cv2.imread('number/5.png')]

def calcAndDrawHist(image, color, max=255):
    hist = cv2.calcHist([image], [0], None, [max+1], [0.0, max])

    a = 0.8
    for i in range(2, len(hist)):
        hist[i] = a * hist[i - 1] + (1- a) * hist[i]

    for i in range(len(hist) - 2, 1, -1):
        hist[i] = a * hist[i + 1] + (1 - a) * hist[i]

    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
    histImg = np.zeros([256, 256, 3], np.uint8)
    hpt = int(0.9 * 256)

    for h in range(256):
        intensity = int(hist[h] * hpt / maxVal)
        cv2.line(histImg, (h, 256), (h, 256 - intensity), color)

    return hist, histImg


net = torchvision.models.resnet18(pretrained=True)
net.fc = nn.Linear(512, 5)
model = torch.load('model_parameters.zip')
net.load_state_dict(model)
net.cuda()
net.eval()
try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        show2 = copy.deepcopy(color_image)
        pipe_line_data = copy.deepcopy(color_image)

        #1 FILFER
        # pipe_line_data = cv2.GaussianBlur(pipe_line_data, (9, 9), 42)

        #2 thresholding 1
        HSV = cv2.cvtColor(pipe_line_data, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(HSV)
        hist_s, hist_sImg = calcAndDrawHist(S, [0, 255, 0])
        mask = (S > 100)
        pipe_line_data[~mask] = (0, 0, 0)

        # thresholding 2
        B, G, R = cv2.split(pipe_line_data)
        mask = mask & (B > 65)

        # closing
        temp = np.zeros_like(mask, dtype=np.uint8)
        temp[mask] = 255
        temp = cv2.dilate(temp, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
        temp = cv2.erode(temp, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
        mask = temp > 0

        mask_stage_1 = mask
        # align
        mask = tools.newMask(mask, -17)

        # thresholding on depth image
        mask = mask & (depth_image < 1000)

        # display mask by temp
        temp = np.zeros_like(mask, dtype=np.uint8)
        temp[mask] = 255

        # apply mask on depth map
        depth_image[~mask] = 0


        depth_pipe = copy.deepcopy(depth_colormap)
        d1 = copy.deepcopy(depth_pipe)
        depth_pipe[~mask] = 0
        d2 = depth_pipe

        show2[~mask_stage_1] = (0, 0, 0)
        images = np.hstack((color_image, show2))
        d = np.hstack((d1, d2))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.imshow('depth', d)

        cv2.imshow('hist', hist_sImg)
        cv2.imshow('color_image', color_image)
        # cv2.imshow('temp', temp)



        # cv2.imshow("histv", hist_vImg)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF == ord('s'):
            n = 5
            tools.saveFile(depth_pipe, name='g'+ str(n))
            tools.saveFile(depth_colormap, name='d'+ str(n))

        # if cv2.waitKey(1) & 0xFF == ord('e'):

        data = depth_pipe
        data = np.array([data])
        data = data[:, np.newaxis, :, :]
        data = torch.from_numpy(data)
        data = data.permute(0, 4, 2, 3, 1)
        data = data.view(-1, 3, 480, 640).float().cuda()
        data = F.adaptive_max_pool2d(data, 224)
        out = net(data)
        pred_choice = out.data.max(1)[1]
        pred_choice = int(pred_choice.cpu())+1
        print(pred_choice)
        cv2.imshow('show', show[pred_choice])


finally:

    # Stop streaming
    pipeline.stop()
