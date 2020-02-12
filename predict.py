import numpy as np
import cv2
import os
import time
import keras
import model
from util import ufunc_4, scale_expand_kernels, fit_minarearectange, fit_boundingRect_2, text_porposcal


checkpoint_path = './resnet_train/'
test_data_path = '/Users/apple/PycharmProjects/design/venvnew/Data/demo/'

psenet = model.PSEnet().model
psenet.load_weights(checkpoint_path)

MIN_LEN = 640
MAX_LEN = 1240


def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def predict(images):
    h, w = images.shape[0:2]
    if (w < h):
        if (h < MIN_LEN):
            scale = 1.0 * MIN_LEN / h
            w = w * scale
            h = MIN_LEN
        elif (h > MAX_LEN):
            scale = 1.0 * MAX_LEN / h
            w = w * scale
            h = MAX_LEN
    elif (h <= w):
        if (w < MIN_LEN):
            scale = 1.0 * MIN_LEN / w
            h = scale * h
            w = MIN_LEN
        elif (w > MAX_LEN):
            scale = 1.0 * MAX_LEN / w
            h = scale * w
            h = MAX_LEN

    w = int(w // 32 * 32)
    h = int(h // 32 * 32)

    scalex = images.shape[1] / w
    scaley = images.shape[0] / h

    images = cv2.resize(images, (w, h), cv2.INTER_AREA)
    images = np.reshape(images, (1, h, w, 3))

    timer = {'net': 0, 'pse': 0}
    start = time.time()
    res = psenet.predict(images[0:1, :, :, :])
    timer['net'] = time.time() - start

    res1 = res[0]
    res1[res1 > 0.9] = 1
    res1[res1 <= 0.9] = 0
    newres1 = []
    for i in [2, 4]:
        n = np.logical_and(res1[:, :, 5], res1[:, :, i]) * 255
        newres1.append(n)
    newres1.append(res1[:, :, 5] * 255)

    start = time.time()
    num_label, labelimage = scale_expand_kernels(newres1, filter=False)
    rects = fit_boundingRect_2(num_label, labelimage)
    timer['pse'] = time.time()-start
    print(' net {:.0f}ms, pse {:.0f}ms'.format(
        timer['net'] * 1000, timer['pse'] * 1000))

    im = np.copy(images[0])
    for rt in rects:
        cv2.rectangle(im, (rt[0] * 2, rt[1] * 2), (rt[2] * 2, rt[3] * 2), (0, 255, 0), 2)

    g = text_porposcal(rects, res1.shape[1], max_dist=8, threshold_overlap_v=0.3)
    rects = g.get_text_line()

    for rt in rects:
        cv2.rectangle(im, (rt[0] * 2, rt[1] * 2 - 2), (rt[2] * 2, rt[3] * 2), (0, 0, 255), 2)
    cv2.imwrite('/Users/apple/PycharmProjects/design/venvnew/Data/demo/test_out.png', im)

    return rects


im_fn_list = get_images()
for im_fn in im_fn_list:
    im = cv2.imread(im_fn)[:, :, ::-1]
    rect = predict(im)
