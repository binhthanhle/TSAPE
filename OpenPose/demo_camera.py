import os
import sys
import argparse
import cv2
import math
import time
import numpy as np
import util
from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
            maxscore = 0.7
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
                    criterion1 = len(np.nonzero(score_midpts > params['thre2'])[0]) > 0.8 * len(
                        score_midpts)
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
            'RHip', 'RAnkle', 'RKnee',
            'LHip', 'LAnkle', 'LKnee',
            'REye', 'LEye',
            'REar', 'LEar']

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--video', type=str, required=True, help='input video file name')
    parser.add_argument('--model', type=str, default='model/keras/model.h5', help='path to the weights file')
    parser.add_argument('--frame_ratio', type=int, default=7, help='analyze every [n] frames')
    # --process_speed changes at how many times the model analyzes each frame at a different scale
    parser.add_argument('--process_speed', type=int, default=1, help='Int 1 (fastest, lowest quality) to 4 (slowest, highest quality)')
    #parser.add_argument('--start', type=int, default=0, help='Video frame to start with')
    parser.add_argument('--end', type=int, default=None, help='Last video frame to analyze')

    args = parser.parse_args()
    #input_image = args.image
    #output = args.output
    keras_weights_file = args.model
    frame_rate_ratio = args.frame_ratio
    process_speed = args.process_speed
    #starting_frame = args.start
    ending_frame = args.end

    print('start processing...')

    # Video input
    video = 'webcam'
    video_path = 'videos/'
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

    # Video reader
    #cam = cv2.VideoCapture(video_file)
    cam = cv2.VideoCapture(0)
    #CV_CAP_PROP_FPS
    cam.set(cv2.CAP_PROP_FPS, 10)
    # cam.set(cv2.CAP_PROP_FPS, 10)
    input_fps = cam.get(cv2.CAP_PROP_FPS)
    print("Running at {} fps.".format(input_fps))
    ret_val, input_image = cam.read()
    video_length = 1000 #int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

    # Plotting
    import matplotlib.pyplot as plt
    grid = plt.GridSpec(2, 5, wspace=0.2, hspace=0.2)
    fig = plt.figure(figsize=(10, 5))
    # ax1 = fig.add_subplot(grid[0:, 0:2])
    ax2 = fig.add_subplot(grid[0, 0:])
    ax3 = fig.add_subplot(grid[1, 0:])

    fig.show()
    old_peaks = None
    df_timeseries_x = pd.DataFrame([], columns=np.arange(18))
    df_timeseries_y = pd.DataFrame([], columns=np.arange(18))

    if ending_frame == None:
        ending_frame = video_length

    # Video writer
    output_fps = input_fps / frame_rate_ratio
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #out = cv2.VideoWriter(video_output,fourcc, output_fps, (input_image.shape[1], input_image.shape[0]))

    i = 0 # default is 0
    resize_fac = 3
    scale_search = [0.3]

    width = input_image.shape[1]
    height = input_image.shape[0]
    factor = 0.2

    while(cam.isOpened()) and ret_val == True and i < ending_frame:
        #if i%frame_rate_ratio == 0:
        while True:
            cv2.waitKey(10)
            ret_val, orig_image = cam.read()
            tic = time.time()

            width = orig_image.shape[1]
            height = orig_image.shape[0]

            cropped = orig_image[:,int(width*factor):int(width*(1-factor))]
            # generate image with body parts

            input_image = cv2.resize(cropped, (0, 0), fx=1.0/resize_fac, fy=1.0/resize_fac, interpolation=cv2.INTER_CUBIC)
            canvas,all_peaks = process(input_image, params, model_params, old_peaks,scale_search)

            print('Processing frame: ', i)
            toc = time.time()
            print ('processing time is %.5f' % (toc - tic))

            #out.write(canvas)
            #canvas = cv2.resize(canvas, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

            cv2.imshow('frame',canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            df_timeseries_x = df_timeseries_x.append(pd.DataFrame([e[0] for each in all_peaks for e in each]).T,
                                                     ignore_index=True)
            df_timeseries_y = df_timeseries_y.append(pd.DataFrame([e[1] for each in all_peaks for e in each]).T,
                                                     ignore_index=True)

            # ## plot
            # ax1.clear()
            # ax1.imshow(canvas, aspect='auto')
            # #         fig.canvas.draw()

            ax2.clear()
            ax2.plot(df_timeseries_x[0], 's-')

            ax3.clear()
            ax3.plot(df_timeseries_y[0], 's--')

            fig.canvas.draw()

            #         ax4.clear()
            #         ax4.scatter(x=df_timeseries_x[5],y=df_timeseries_y[5])

            #         plt.legend(loc='best')
            # fig.canvas.draw()
            # time.sleep(0.1)

            # canvas.get_tk_widget().grid(row=0, column=0)

        #elif i % 3 == 0:
            #cv2.imshow('frame',input_image)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
                #break
        ret_val, input_image = cam.read()
        i += 1
    cv2.destroyAllWindows()
    out.release()
