import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# --------------------------------------
# LENET_V2
# --------------------------------------

wrongseeds_LeNet_v2 =[80, 494, 635, 902, 2426, 2427, 2488, 5298, 6315, 6848, 6885, 8200]

# --------------------------------------
# LENET
# --------------------------------------

wrongseeds_LeNet = [80, 132, 494, 500, 788, 1201, 1244, 1604, 2426, 2554, 2622, 2676, 3210, 4014, 4164, 4402, 4438,
                    4476, 4715, 5065, 5174, 5482, 5723, 5798, 5821, 5855, 6197, 6202, 6246, 6315, 6818, 6885, 7006,
                    7080, 7264, 7606, 7972, 7994, 8200, 8202, 8270, 8480, 8729, 8772, 8849, 9256, 9266]
LENET_0_5 = [
    [0, 1, 267, 107, 117, 830, 197, 53, 153, 146],
    [2, 0, 49, 37, 947, 35, 49, 21, 310, 310],
    [5, 27, 0, 470, 64, 49, 24, 231, 168, 11],
    [20, 31, 61, 0, 28, 41, 8, 84, 177, 126],
    [2, 43, 20, 111, 0, 132, 31, 324, 417, 194],
    [25, 1, 7, 337, 75, 0, 70, 20, 369, 150],
    [174, 58, 11, 69, 256, 577, 0, 0, 466, 13],
    [19, 80, 207, 669, 160, 77, 1, 0, 10, 707],
    [7, 27, 410, 735, 414, 5, 87, 195, 0, 9],
    [11, 5, 21, 318, 1136, 482, 10, 767, 718, 0]
]  # epsilon = 0.5

# --------------------------------------
# ALEXNET
# --------------------------------------
wrongseeds_AlexNet = [80, 132, 180, 262, 318, 390, 444, 494, 578, 588, 720, 754, 778, 854, 892, 995, 1032, 1047, 1058,
                      1088, 1143, 1244, 1258, 1330, 1404, 1498, 1512, 1604, 1674, 1784, 1854, 2052, 2078, 2148, 2184,
                      2338, 2426, 2490, 2576, 2652, 2676, 2901, 2902, 3094, 3268, 3308, 3484, 3692, 3730, 3756, 3798,
                      4005, 4292, 4438, 4476, 4729, 5110, 5216, 5332, 5338, 5430, 5466, 5554, 5632, 5659, 5706, 5790,
                      5936, 6102, 6197, 6251, 6269, 6347, 6606, 6658, 6714, 6755, 6771, 6798, 6810, 6879, 7080, 7347,
                      7528, 7628, 7758, 7920, 8202, 8384, 8428, 8470, 8480, 8560, 8670, 8688, 8772, 8853, 8883, 8898,
                      8918, 9040, 9104, 9216, 9220, 9396, 9504, 9528, 9608, 9687]
AlexNet_0_5 = [[0, 28, 14, 1, 2, 53, 14, 45, 147, 32],
               [0, 0, 1, 62, 15, 24, 4, 93, 302, 8],
               [28, 313, 0, 248, 2, 0, 5, 187, 77, 17],
               [37, 78, 59, 0, 0, 50, 0, 59, 22, 146],
               [2, 71, 1, 0, 0, 9, 10, 107, 13, 76],
               [8, 20, 1, 96, 1, 0, 4, 21, 36, 103],
               [86, 174, 9, 14, 55, 317, 0, 1, 146, 9],
               [0, 23, 30, 3, 0, 1, 0, 0, 56, 23],
               [26, 258, 242, 236, 64, 172, 8, 72, 0, 199],
               [9, 10, 5, 202, 36, 264, 0, 881, 227, 0]]  # epsilon = 0.5

# --------------------------------------
# VGG13
# --------------------------------------
wrongseeds_vgg13 = [132, 828, 1047, 1404, 1604, 2426, 3512, 3532, 3811, 4129, 4253, 4476, 5638, 5740, 5972, 6066, 6251,
                    6962, 7080, 7264, 8118, 8701, 9504, 9608]
vgg13_0_5 = [[0, 0, 597, 44, 1, 13, 20, 0, 403, 3], [0, 0, 1032, 46, 468, 20, 1, 173, 447, 32],
             [0, 29, 0, 94, 0, 0, 0, 111, 14, 2], [2, 25, 263, 0, 6, 74, 0, 71, 8, 40],
             [0, 1, 2, 0, 0, 17, 7, 19, 7, 171], [0, 0, 5, 226, 15, 0, 44, 1, 55, 118],
             [44, 3, 20, 51, 33, 751, 0, 0, 610, 3], [0, 0, 52, 388, 3, 3, 0, 0, 2, 45],
             [12, 6, 914, 854, 38, 336, 74, 15, 0, 167], [3, 0, 67, 523, 1213, 62, 0, 1160, 473, 0]]  # epsilon = 0.5

# --------------------------------------
# VGG16
# --------------------------------------
wrongseeds_vgg16 = [286, 386, 418, 440, 494, 524, 528, 605, 846, 854, 935, 958, 1004, 1021, 1030, 1047, 1077, 1088,
                    1120, 1134, 1160, 1259, 1376, 1404, 1501, 1604, 1634, 1682, 1784, 1871, 1872, 1888, 1930, 2014,
                    2044, 2202, 2220, 2320, 2410, 2426, 2488, 2652, 2676, 2718, 2720, 2765, 2845, 2901, 2931, 2958,
                    3030, 3268, 3290, 3370, 3426, 3456, 3532, 3730, 3756, 3772, 3986, 4028, 4030, 4046, 4153, 4164,
                    4165, 4210, 4292, 4334, 4434, 4438, 4460, 4476, 4560, 4634, 4640, 4646, 4692, 4759, 4799, 4822,
                    4863, 4893, 4907, 4945, 5002, 5052, 5084, 5126, 5194, 5216, 5260, 5278, 5283, 5298, 5338, 5462,
                    5474, 5513, 5554, 5632, 5738, 5780, 5798, 5821, 5896, 5912, 5997, 6050, 6066, 6102, 6129, 6220,
                    6246, 6259, 6315, 6347, 6428, 6466, 6486, 6498, 6578, 6582, 6658, 6704, 6706, 6746, 6792, 6839,
                    6885, 6920, 7009, 7074, 7080, 7146, 7225, 7264, 7283, 7293, 7347, 7452, 7544, 7584, 7588, 7606,
                    7760, 7768, 7784, 7898, 7920, 7972, 7984, 7994, 7995, 8116, 8200, 8202, 8226, 8268, 8284, 8480,
                    8485, 8596, 8617, 8645, 8661, 8670, 8759, 8761, 8772, 8797, 8799, 8867, 8898, 8974, 9098, 9158,
                    9302, 9392, 9396, 9433, 9436, 9455, 9549, 9551, 9571, 9657, 9665, 9669, 9681, 9687, 9711, 9717,
                    9782, 9932, 9952]
vgg16_0_5 = []  # epsilon = 0.5
