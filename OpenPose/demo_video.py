import os
import sys
import argparse
import cv2
import math, scipy
import time
import numpy as np
import util
from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model.cmu_model import get_testing_model

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

currentDT = time.localtime()
start_datetime = time.strftime("-%m-%d-%H-%M-%S", currentDT)

# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
           [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
           [1, 16], [16, 18], [3, 17], [6, 18]]

# the middle joints heatmap correpondence
mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], \
          [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], \
          [55, 56], [37, 38], [45, 46]]

# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


def process(input_image, params, model_params, old_peaks, scale_search):
    oriImg = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

    #     scale_search = [0.7]#[1, .5, 1.5, 2] # [.5, 1, 1.5, 2]
    scale_search = scale_search[0:process_speed]

    multiplier = [np.round(x, 2) * np.round(model_params['boxsize'], 2) / np.round(oriImg.shape[1], 2) for x in
                  scale_search]

    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

    for m in range(len(multiplier)):
        scale = multiplier[m]
        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        # imageToTest = scipy.misc.imresize(oriImg, int(scale*100), interp='cubic')
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'],
                                                          model_params['padValue'])

        input_img = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]),
                                 (3, 0, 1, 2))  # required shape (1, width, height, channels)

        output_blobs = model.predict(input_img)

        # extract outputs, resize, and remove padding
        heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3],
                  :]
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
        paf = cv2.resize(paf, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)

    all_peaks = []
    peak_counter = 0

    for part in range(18):
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > params['thre1']))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse

        #         print('peaks=',peaks)

        peaks_with_score = []

        ### check old peaks
        x_old_peaks = None
        if old_peaks is not None:
            if old_peaks[part] != []:
                x_old_peaks = old_peaks[part][0][0:2]
        #                 print('x_old_peak_part=',x_old_peaks)

        if len(peaks) == 0:
            if x_old_peaks is not None:
                peaks2 = [x_old_peaks]
            else:
                peaks2 = []
        elif len(peaks) == 1:
            peaks2 = peaks
        else:
            mindis = np.inf
            maxscore = 0.6
            for i, x in enumerate(peaks):
                if x_old_peaks is not None:
                    dis = np.power(x[1] - x_old_peaks[1], 2) + np.power(x[1] - x_old_peaks[1], 2)
                    if dis < mindis:
                        peaks2 = [x]
                        mindis = dis
                else:
                    score = map_ori[x[1], x[0]]
                    if score > maxscore:
                        peaks2 = [x]
                        maxscore = score
                    else:
                        peaks2 = []

        #         print('peaks2=',peaks2)
        #         print('map_ori[x[1], x[0]]=',[map_ori[x[1], x[0]] for x in peaks2])
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks2]

        #         ## Remove uncertained points
        #         peaks2 = []
        #         for x in peaks:
        #             if (map_ori[x[1], x[0]] > 0.7):
        #                 peaks_with_score.append(x + (map_ori[x[1], x[0]],))
        #                 peaks2.append(x)

        id = range(peak_counter, peak_counter + len(peaks2))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks2)

    #     for i,x in enumerate(all_peaks):
    #         if not x:
    #             if old_peaks is not None:
    #                 all_peaks[i] = old_peaks[i]

    connection_all = []
    special_k = []
    mid_num = 10

    for k in range(len(mapIdx)):
        score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if (nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    # failure case when 2 body parts overlaps
                    if norm == 0:
                        continue
                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                        np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    vec_x = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                         for I in range(len(startend))])
                    vec_y = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                         for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                        0.5 * oriImg.shape[0] / norm - 1, 0)
                    criterion1 = len(np.nonzero(score_midpts > params['thre2'])[0]) > 0.8 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior,
                                                     score_with_dist_prior + candA[i][2] + candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if (len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    #     print(all_peaks)

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if (subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + \
                              connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur
    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    canvas = input_image

    #     print len(all_peaks)
    #     print all_peaks
    #     print '='*100

    text = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist',
             'RHip','RAnkle', 'RKnee', 'LHip',
            'LAnkle', 'LKnee','REye',  'LEye', 'REar', 'LEar']

    for i in range(18):
        for j in range(len(all_peaks[i])):
            cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)
            cv2.putText(canvas, text[i], all_peaks[i][j][0:2], cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2, cv2.LINE_AA)

    stickwidth = 2

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0,
                                       360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas, all_peaks


def plotdf(df_timeseries_x,df_timeseries_y,ax1,ax2,fig):
    df_timeseries_x_plot = df_timeseries_x.copy()
    df_timeseries_y_plot = df_timeseries_y.copy()
    df_timeseries_x_plot = df_timeseries_x_plot.rename(
        columns={0: 'Nose', 1: 'Neck', 2: 'RShoulder', 3: 'RElbow', 4: 'RWrist', 5: 'LShoulder',6: 'LElbow', 7: 'LWrist',
                  8: 'RHip', 9: 'RAnkle', 10: 'RKnee', 11: 'LHip',
                 12: 'LAnkle', 13: 'LKnee', 14: 'REye', 15: 'LEye', 16: 'REar', 17: 'LEar'})

    df_timeseries_y_plot = df_timeseries_y_plot.rename(
        columns={0: 'Nose', 1: 'Neck', 2: 'RShoulder', 3: 'RElbow', 4: 'RWrist', 5: 'LShoulder',6: 'LElbow', 7: 'LWrist',
                  8: 'RHip', 9: 'RAnkle', 10: 'RKnee', 11: 'LHip',
                 12: 'LAnkle', 13: 'LKnee', 14: 'REye', 15: 'LEye', 16: 'REar', 17: 'LEar'})

    ## plot x
    lengends = df_timeseries_x_plot.columns
    # fig = plt.figure(figsize=(10, 10))
    # ax1 = plt.subplot(2, 1, 1)

    style = ['b-', 'b--', 'b:', 'r-', 'r--', 'r:', 'g-', 'g--', 'g:', 'm-', 'm--', 'm:', 'c-', 'c--', 'c:', 'k-', 'k--',
             'k:']

    # df_timeseries_x.plot(ax=ax)

    # x = np.arange(10)
    for i in np.arange(len(lengends)):
        line, = ax1.plot(df_timeseries_x_plot.iloc[:, i], style[i], label=lengends[i])

    # # Put a legend below current axis
    # ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #           fancybox=True, shadow=True, ncol=5)


    ## plot y
    lengends = df_timeseries_y_plot.columns
    # ax2 = plt.subplot(2, 1, 2)

    style = ['b-', 'b--', 'b:', 'r-', 'r--', 'r:', 'g-', 'g--', 'g:', 'm-', 'm--', 'm:', 'c-', 'c--', 'c:', 'k-', 'k--',
             'k:']

    # df_timeseries_x.plot(ax=ax)

    # x = np.arange(10)
    for i in np.arange(len(lengends)):
        line, = ax2.plot(df_timeseries_y.iloc[:, i], style[i], label=lengends[i])

    # Put a legend below current axis
    ax1.legend(labelspacing=0.5, loc='center left', bbox_to_anchor=(1, 0.5))

    fig.canvas.draw()

    # plt.show()


def makebgr(cropped,CANNY_THRESH_1,CANNY_THRESH_2,MASK_DILATE_ITER,MASK_ERODE_ITER,BLUR):
    ## Grayscale
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # -- Edge detection
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    # -- Find contours in edges, sort by area
    contour_info = []
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]

    # -- Create empty mask, draw filled polygon on it corresponding to largest contour ----
    # Mask is black, polygon is white

    # Change to
    mask = np.zeros(edges.shape)
    for c in contour_info:
        cv2.fillConvexPoly(mask, c[0], 255)

    # mask = np.zeros(edges.shape)
    # cv2.fillConvexPoly(mask, max_contour[0], 255)

    # -- Smooth mask, then blur it
    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    mask_stack = np.dstack([mask] * 3)  # Create 3-channel alpha mask

    # -- Blend masked img into MASK_COLOR background
    mask_stack = mask_stack.astype('float32') / 255.0
    img = cropped.astype('float32') / 255.0
    masked = (mask_stack * img) + ((1 - mask_stack) * MASK_COLOR)
    masked = (masked * 255).astype('uint8')

    return masked

def rotateImage(image, angle):
    image0 = image
    # if hasattr(image, 'shape'):
    image_center = tuple(np.array(image.shape)[:2]/2)
    shape = tuple(np.array(image.shape)[:2])
    # elif hasattr(image, 'width') and hasattr(image, 'height'):
    #     image_center = tuple(np.array((image.width/2, image.height/2)))
    #     shape = (image.width, image.height)
    # else:
    #     raise Exception, 'Unable to acquire dimensions of image for type %s.' % (type(image),)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle,1.0)
    image = np.asarray( image[:,:] )

    rotated_image = cv2.warpAffine(image, rot_mat, shape, flags=cv2.INTER_LINEAR)

    # Copy the rotated data back into the original image object.
    # cv2.SetData(image0, rotated_image.tostring())

    return rotated_image

def set_params(params):
    # params = dict()
    params["logging_level"] = 3
    params["output_resolution"] = "-1x-1"
    params["net_resolution"] = "-1x368"
    params["model_pose"] = "BODY_25"
    params["alpha_pose"] = 0.6
    params["scale_gap"] = 0.3
    params["scale_number"] = 1
    params["thres1"] = 0.5
    params["thres2"] = 0.9
    # If GPU version is built, and multiple GPUs are available, set the ID here
    params["num_gpu_start"] = 0
    params["disable_blending"] = False
    # Ensure you point to the correct path where models are located
    return params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='input video file name')
    parser.add_argument('--model', type=str, default='model/keras/model.h5', help='path to the weights file')
    parser.add_argument('--frame_ratio', type=int, default=1, help='analyze every [n] frames')
    parser.add_argument('--process_speed', type=int, default=4, help='Int 1 (fastest, lowest quality) to 4 (slowest, highest quality)')
    parser.add_argument('--end', type=int, default=None, help='Last video frame to analyze')

    args = parser.parse_args()

    keras_weights_file = args.model
    frame_rate_ratio = args.frame_ratio
    process_speed = args.process_speed
    ending_frame = args.end

    print('start processing...')

    # Video input
    video = args.video
    video_path = ''
    video_file = video_path + video

    # Output location
    output_path = 'videos/outputs/'
    output_format = '.mp4'
    video_output = output_path + video + str(start_datetime) + output_format

    # load model
    # authors of original model don't use
    # vgg normalization (subtracting mean) on input images
    model = get_testing_model()
    model.load_weights(keras_weights_file)

    # load config

    params, model_params = config_reader()
    # params = set_params(params)
    # model_params = set_params(model_params)
    # Video reader
    cam = cv2.VideoCapture(video_file)
    # cam.set(cv2.CAP_PROP_POS_FRAMES, 45000-1)

    length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    print(length)

    input_fps = cam.get(cv2.CAP_PROP_FPS)
    ret_val, input_image = cam.read()
    video_length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

    if ending_frame == None:
        ending_frame = video_length

        # Crop image
    width = input_image.shape[1]
    height = input_image.shape[0]
    factor = 0.0
    factor2 = 0.0
    factor3 = 0.0
    resize_fac = 1

    cropped = input_image[:, int(width * factor):int(width * (1 - factor))]
    cropped = cropped[int(height * factor2):int(height * (1 - factor3)), :]
    cropped = cv2.resize(cropped, (0, 0), fx=1.0 / resize_fac, fy=1.0 / resize_fac,
                         interpolation=cv2.INTER_AREA)

    cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    cropped_gray = cv2.GaussianBlur(cropped_gray, (5, 5), 0)

    # Video writer
    output_fps = input_fps / frame_rate_ratio
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_output, fourcc, output_fps, (cropped.shape[1], cropped.shape[0]))

    i = 0  # default is 0


    # plots:
    import matplotlib.pyplot as plt

    grid = plt.GridSpec(2, 5, wspace=0.2, hspace=0.2)
    fig = plt.figure(figsize=(10, 5))
    # ax1 = fig.add_subplot(grid[0:, 0:2])
    ax2 = fig.add_subplot(grid[0, 0:])
    ax3 = fig.add_subplot(grid[1, 0:])

    fig.show()

    # fig.show()
    old_peaks = None
    df_timeseries_x = pd.DataFrame([], columns=np.arange(18))
    df_timeseries_y = pd.DataFrame([], columns=np.arange(18))

    scale_search = [0.5]
    while (cam.isOpened()) and ret_val == True and i < ending_frame:
        if i % frame_rate_ratio == 0:
            tic = time.time()

            # Crop image
            width = input_image.shape[1]
            height = input_image.shape[0]

            cropped = input_image[:, int(width * factor):int(width * (1 - factor))]
            cropped = cv2.resize(cropped, (0, 0), fx=1.0 / resize_fac, fy=1.0 / resize_fac,
                                 interpolation=cv2.INTER_CUBIC)

            # generate image with body parts
            # == Parameters
            # BLUR = 21
            # CANNY_THRESH_1 = 200
            # CANNY_THRESH_2 = 15
            # MASK_DILATE_ITER = 3
            # MASK_ERODE_ITER = 25
            # MASK_COLOR = (1.0, 1.0, 1.0)  # In BGR format
            # cropped = makebgr(cropped, CANNY_THRESH_1, CANNY_THRESH_2, MASK_DILATE_ITER, MASK_ERODE_ITER, BLUR)

            canvas, all_peaks = process(cropped, params, model_params, old_peaks, scale_search)



            old_peaks = all_peaks
            print(video + ' :::: Processing frame: ', i)
            toc = time.time()
            print('processing time is %.5f' % (toc - tic))
            out.write(canvas)

            df_timeseries_x = df_timeseries_x.append(pd.DataFrame([e[0] for each in all_peaks for e in each]).T,
                                                     ignore_index=True)
            df_timeseries_y = df_timeseries_y.append(pd.DataFrame([e[1] for each in all_peaks for e in each]).T,
                                                     ignore_index=True)

            ax2.clear()
            ax3.clear()
            plotdf(df_timeseries_x, df_timeseries_y, ax2, ax3, fig)

            cv2.imshow('frame', canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        ret_val, input_image = cam.read()
        i += 1

    cv2.destroyAllWindows()
    out.release()

    df_timeseries_y = df_timeseries_y.rename(
        columns={0: 'Nose', 1: 'Neck', 2: 'RShoulder', 3: 'RElbow', 4: 'RWrist', 5: 'LShoulder',6: 'LElbow', 7: 'LWrist',
                  8: 'RHip', 9: 'RAnkle', 10: 'RKnee', 11: 'LHip',
                 12: 'LAnkle', 13: 'LKnee', 14: 'REye', 15: 'LEye', 16: 'REar', 17: 'LEar'})

    df_timeseries_x = df_timeseries_x.rename(
        columns={0: 'Nose', 1: 'Neck', 2: 'RShoulder', 3: 'RElbow', 4: 'RWrist', 5: 'LShoulder',6: 'LElbow', 7: 'LWrist',
                  8: 'RHip', 9: 'RAnkle', 10: 'RKnee', 11: 'LHip',
                 12: 'LAnkle', 13: 'LKnee', 14: 'REye', 15: 'LEye', 16: 'REar', 17: 'LEar'})

    # plotdf(df_timeseries_x, df_timeseries_y)
    df_output_x = output_path + video + str(start_datetime) + '_x.csv'
    df_timeseries_x.to_csv(df_output_x,sep=',')
    df_output_y = output_path + video + str(start_datetime) + '_y.csv'
    df_timeseries_y.to_csv(df_output_y, sep=',')