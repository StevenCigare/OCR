from PIL import Image
import os
import numpy as np
import cv2
import time


def save_to_npz(path_to_files, to_save):
    vectorized_images = []
    mylist = []
    for file in os.listdir(path_to_files):
        a = []
        for i in file:
            if i == '.':
                break
            a.append(i)
        a = ''.join(a)
        mylist.append(int(a))

    mylist = sorted(mylist)
    mylist2 = [f'{num}.png' for num in mylist]

    for file in mylist2:
        image = Image.open(path_to_files + file)
        image_array = np.array(image)
        vectorized_images.append(image_array)
    np.savez(to_save, DataX=vectorized_images)


def bytes_to_int(byte_data):
    return int.from_bytes(byte_data, 'big')


def flatten_list(l):
    return [pixel for sublist in l for pixel in sublist]


def extract_features(X):
    return [flatten_list(sample) for sample in X]


def dist(x, y):
    return sum(
        [
            (bytes_to_int(x_i) - bytes_to_int(y_i)) ** 2
            for x_i, y_i in zip(x, y)
        ]
    ) ** (0.5)


def get_training_distances_for_test_sample(X_train, test_sample):
    return [dist(train_sample, test_sample) for train_sample in X_train]


def get_most_frequent_element(l):
    return max(l, key=l.count)


def knn(X_train, y_train, X_test, k=1):
    y_pred = []
    for test_sample_idx, test_sample in enumerate(X_test):
        training_distances = get_training_distances_for_test_sample(
            X_train, test_sample
        )

        sorted_distance_indices = [
            pair[0]
            for pair in sorted(
                enumerate(training_distances),
                key=lambda x: x[1]
            )
        ]
        candidates = [
            y_train[idx]
            for idx in sorted_distance_indices[:k]
        ]
        top_candidate = get_most_frequent_element(candidates)

        y_pred.append(top_candidate)
    return y_pred


# image magick

def load_data(path):
    with np.load(path) as data:
        train_data = data['DataX']
    return train_data


def avg_distance(contours):
    distances = []
    for i in range(len(contours) - 1):
        [x, _, w, _] = cv2.boundingRect(contours[i])
        [x_next, _, _, _] = cv2.boundingRect(contours[i + 1])
        distances.append(x_next - (x + w))
    return sum(distances) / len(distances), distances


def add_spaces(y_pred, distances, avg):
    count = 0
    for i in range(len(distances)):
        if distances[i] >= 2.0 * avg:
            y_pred.insert(i + 1 + count, ' ')
            count += 1


def remove_inside_contours(contours):
    fixed_contours = [contours[0]]
    for i in range(len(contours) - 1):
        [x, y, w, h] = cv2.boundingRect(contours[i])
        [x_next, y_next, w_next, h_next] = cv2.boundingRect(contours[i + 1])
        if not (x < x_next < x_next + w_next < x + w and y < y_next < y_next + h_next < y + h):
            fixed_contours.append(contours[i + 1])
    return fixed_contours


def find_all_lines(contours, y_res):
    max_val = 0
    temp_val = 0
    temp_max_y = 0
    line_pos = []
    for y in range(y_res):
        for contour in contours:
            [_, y_c, _, h] = cv2.boundingRect(contour)
            if y_c <= y <= y_c + h:
                temp_val += 1

        if temp_val > max_val:
            max_val = temp_val
            temp_max_y = y
        if temp_val == 0 and max_val != 0:
            line_pos.append(temp_max_y)
            max_val = 0

        temp_val = 0
    lines = []

    for y in line_pos:
        contours_in_line = []
        for contour in contours:
            [_, y_c, _, h] = cv2.boundingRect(contour)
            if y_c <= y <= y_c + h:
                contours_in_line.append(contour)
        lines.append(contours_in_line)

    return lines


def main():

    index = 0

    img = cv2.imread("./full_text_images/innyfont.png")
    width, height, _ = img.shape
    img = cv2.resize(img, (10*width, 10*height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    contours = [contour for contour in contours if cv2.contourArea(contour) > 10]

    contours.pop(0)
    print(len(contours))
    contours = remove_inside_contours(contours)

    print(len(contours))
    for contour in contours:
        if cv2.contourArea(contour) > 10:
            [X, Y, W, H] = cv2.boundingRect(contour)
            cv2.rectangle(img, (X, Y), (X + W, Y + H), (0, 255, 255), 2)
            # crop_img = img[Y:Y+H, X:X+W]1
    cv2.imshow('siemano', img)
    cv2.waitKey(0)
    lines = find_all_lines(contours, 10*height)
    lines = [sorted(line, key=lambda x: (x[0][0][0])) for line in lines]
    total_avg = 0

    for idx, line in enumerate(lines):
        for contour in line:
            [X, Y, W, H] = cv2.boundingRect(contour)
            crop_img = img[Y:Y + H, X:X + W]
            crop_img = cv2.resize(crop_img, (28, 28))
            cv2.imwrite(f'./test_images/{index}.png', crop_img)
            index += 1
        avg, _ = avg_distance(line)
        total_avg += avg
        save_to_npz("./test_images/", f"./testdata{idx}.npz")
        filelist = [f for f in os.listdir("./test_images/") if f.endswith(".png")]
        for f in filelist:
            os.remove(os.path.join("./test_images/", f))
    total_avg /= len(lines)
    print(total_avg)
    y_train = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
               'v', 'w', 'x', 'y', 'z','A', 'B', 'C', 'D', 'E', 'F','G','H','I','J']
    x_train = load_data("./traindata.npz")
    x_test = load_data("./testdata.npz")
    y_test = ['g', 'j', 'k']
    x_train = extract_features(x_train)
    for idx, line in enumerate(lines):
        x_test = extract_features(load_data(f'./testdata{idx}.npz'))
        y_pred = knn(x_train, y_train, x_test)
        _, distances = avg_distance(line)
        add_spaces(y_pred, distances, total_avg)
        y_pred.append('\n')
        print(y_pred)

if __name__ == '__main__':
    main()