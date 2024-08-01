from matplotlib import pyplot as plt, patches
import numpy as np
import cv2


IMAGE_PATH = "./sample-2.png"
THRESHOLD_PATH = "./setting-thresholds.txt"


def find_ball(image_path, threshold_path):
    image = cv2.imread(image_path)
    thresholds = np.loadtxt(threshold_path, delimiter=',', dtype=int)
    r_min, r_max, g_min, g_max, b_min, b_max = thresholds.T
    lower_orange = (int(b_min[1]), int(g_min[1]), int(r_min[1]))
    upper_orange = (int(b_min[1]), int(g_max[1]), int(r_max[1]))
    orange_mask = cv2.inRange(image, lower_orange, upper_orange)
    nonzero_indices = np.argwhere(orange_mask)
    min_x, min_y = nonzero_indices.min(axis=0)
    max_x, max_y = nonzero_indices.max(axis=0)
    min_x -= 10
    min_y -= 10
    max_x += 5
    max_y += 5

    return image, min_x, min_y, max_x, max_y


def classify_image(image_path, thresholds_path):
    image = plt.imread(image_path)
    image_file = image.copy()
    with open(thresholds_path, 'r') as file:
        thresholds = np.loadtxt(file, delimiter=',', dtype=int)
    r_min, r_max, g_min, g_max, b_min, b_max = thresholds.T

    yellow_mask = ((image_file[:, :, 0] >= r_min[0]) & (image_file[:, :, 0] <= r_max[0]) &
                   (image_file[:, :, 1] >= g_min[0]) & (image_file[:, :, 1] <= g_max[0]) &
                   (image_file[:, :, 2] >= b_min[0]) & (image_file[:, :, 2] >= b_max[0]))

    orange_mask = ((image_file[:, :, 0] >= r_min[1]) & (image_file[:, :, 0] <= r_max[1]) &
                   (image_file[:, :, 1] >= g_min[1]) & (image_file[:, :, 1] <= g_max[1]) &
                   (image_file[:, :, 2] >= b_min[1]) & (image_file[:, :, 2] >= b_max[1]))

    pink_mask = ((image_file[:, :, 0] >= r_min[2]) & (image_file[:, :, 0] <= r_max[2]) &
                 (image_file[:, :, 1] >= g_min[2]) & (image_file[:, :, 1] <= g_max[2]) &
                 (image_file[:, :, 2] >= b_min[2]) & (image_file[:, :, 2] >= b_max[2]))

    green_mask = ((image_file[:, :, 0] >= r_min[3]) & (image_file[:, :, 0] <= r_max[3]) &
                  (image_file[:, :, 1] >= g_min[3]) & (image_file[:, :, 1] <= g_max[3]) &
                  (image_file[:, :, 2] >= b_min[3]) & (image_file[:, :, 2] >= b_max[3]))

    yellow_filter = image_file.copy()
    yellow_filter[:, :, 0] *= ~yellow_mask
    yellow_filter[:, :, 1] *= ~yellow_mask
    yellow_filter[:, :, 2] *= ~yellow_mask

    orange_filter = image_file.copy()
    orange_filter[:, :, 0] *= ~orange_mask
    orange_filter[:, :, 1] *= ~orange_mask
    orange_filter[:, :, 2] *= ~orange_mask

    pink_filter = image_file.copy()
    pink_filter[:, :, 0] *= ~pink_mask
    pink_filter[:, :, 1] *= ~pink_mask
    pink_filter[:, :, 2] *= ~pink_mask

    green_filter = image_file.copy()
    green_filter[:, :, 0] *= ~green_mask
    green_filter[:, :, 1] *= ~green_mask * 0.8
    green_filter[:, :, 2] *= ~green_mask

    new_image = image_file.copy()[:, :, :3]
    # new_image[:, :, 0] *= ~yellow_mask
    # new_image[:, :, 1] *= ~orange_mask
    # new_image[:, :, 2] *= ~pink_mask

    new_image[yellow_mask] = image_file[yellow_mask]
    new_image[pink_mask] = image_file[pink_mask]
    new_image[orange_mask] = image_file[orange_mask]

    image_combined = new_image * green_filter
    return image_file, image_combined


def ssl_detect(image_path, threshold_path):
    image, image_combined = classify_image(image_path, threshold_path)
    cv2_image, min_x, min_y, max_x, max_y = find_ball(image_path, threshold_path)
    width = max_y - min_y
    height = max_x - min_x
    rectangle = patches.Rectangle((min_y, min_x), width, height, linewidth=1, edgecolor='b', facecolor='none')

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(image_combined)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
    plt.gca().add_patch(rectangle)
    plt.axis('off')

    plt.figure(figsize=(30, 5))
    plt.show()


ssl_detect(IMAGE_PATH, THRESHOLD_PATH)


