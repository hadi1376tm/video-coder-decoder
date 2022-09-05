
# Video Codec

import cv2
import numpy as np

cap = cv2.VideoCapture('sample 30 frame 1 min.mp4')
if (cap.isOpened()== False):
    print("can't open video")

gray_frames = []

while(cap.isOpened()):

    ret, frame = cap.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frames.append(gray)
    else:
        break

cap.release()

print("old frame size:")
print(gray_frames[0].shape)

# Padding
# Since we're going to work with 8 * 8 blocks in each frame, we need to have frames with dimensions multiples of 8

for i in range(len(gray_frames)):
    current_gray_frame = gray_frames[i]
    gray_frames[i] = cv2.copyMakeBorder(current_gray_frame, 2, 2, 0, 0, cv2.BORDER_REPLICATE)

print("new frame size:")
print(gray_frames[0].shape)


# motion estimation and motion compensation
# patern: ippp ippp ...
print("starting motion estimation")
motion_compensated_frames = [None] * len(gray_frames)
for i in range(len(gray_frames)):
    if i % 4 == 0:
        motion_compensated_frames[i] = gray_frames[i]
    else:
        motion_compensated_frames[i] = gray_frames[i] - gray_frames[i-1]

print("motion estimation finished")

#Discrete Cosine Transform (DCT)

print("starting DCT")

transformed_frames = []
block_width = 8
block_height = 8
frames_number = len(gray_frames)



for f in range(frames_number):
    current_frame = gray_frames[f]
    current_frame_transformed = np.empty_like(current_frame, dtype=np.float32)
    frame_height, frame_width = current_frame.shape
    for i in range(0, frame_height, block_height):
        for j in range(0, frame_width, block_width):
            current_block = np.array(current_frame[i: i + block_height, j: j + block_width], dtype=np.float32)
            transformed_current_block = cv2.dct(current_block)
            current_frame_transformed[i: i + block_height, j: j + block_width] = transformed_current_block
    transformed_frames.append(current_frame_transformed)

print("DCT finished")


# Quantization
print("starting Quantization")
is_negative = [transformed_frames[i] < 0.0 for i in range(len(transformed_frames))]
abs_transformed_frames = [np.abs(transformed_frames[i]).astype(np.uint32) for i in range(len(transformed_frames))]
# for quantization, we shift the values by 4 bits to reduce and remove the least significant bits in frame.
shift_transformed_frames = [abs_transformed_frames[i] >> 4 for i in range(len(transformed_frames))]
# convert values to 'signed int' and with help of 'is_negative' list, we'll specify the sign of the values in each frame.
shift_quantized_frames = [shift_transformed_frames[i].astype(np.int32) for i in range(len(transformed_frames))]
shift_quantized_frames = [np.where(is_negative[i] == True, -shift_quantized_frames[i], shift_quantized_frames[i]) for i in range(len(transformed_frames))]

print("Quantization finished")
# Quantization finished.


# Zig-Zag Scan

def zigzag(input):
    h = 0
    v = 0
    i = 0
    vmin = 0
    hmin = 0
    vmax = input.shape[0]
    hmax = input.shape[1]
    output = np.zeros(( vmax * hmax), dtype=int)

    while ((v < vmax) and (h < hmax)):

        if ((h + v) % 2) == 0:                 # going up

            if (v == vmin):
                output[i] = input[v, h]        # if we got to the first line

                if (h == hmax):
                    v = v + 1
                else:
                    h = h + 1

                i = i + 1

            elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
                output[i] = input[v, h]
                v = v + 1
                i = i + 1

            elif ((v > vmin) and (h < hmax -1 )):    # all other cases
                output[i] = input[v, h]
                v = v - 1
                h = h + 1
                i = i + 1


        else:                                    # going down

            if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
                output[i] = input[v, h]
                h = h + 1
                i = i + 1

            elif (h == hmin):                  # if we got to the first column
                output[i] = input[v, h]

                if (v == vmax -1):
                    h = h + 1
                else:
                    v = v + 1

                i = i + 1

            elif ((v < vmax -1) and (h > hmin)):     # all other cases
                output[i] = input[v, h]
                v = v + 1
                h = h - 1
                i = i + 1




        if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
            output[i] = input[v, h]
            break

    return output



def zigzag_encode_frames(shift_quantized_frames):
    zigzag_frames = []
    for f in range(len(shift_quantized_frames)):
        current_frame = shift_quantized_frames[f]
        zigzag_output = zigzag(current_frame)
        zigzag_frames.append(zigzag_output)
    return zigzag_frames

print("starting zigzag")
zigzag_frames = zigzag_encode_frames(shift_quantized_frames)
print("zigzag finished")

# run lenth scan
def rle_encode(in_list):

    out_list = [(in_list[0], 1)]

    for item in in_list[1:]:

        if item == out_list[-1][0]:
            out_list[-1] = (item, out_list[-1][1] + 1)
        else:
            out_list.append((item, 1))

    return out_list


def rle_encode_frames(zigzag_frames):
    rle_frames = []
    for f in range(len(zigzag_frames)):
        current_frame = zigzag_frames[f]
        output = rle_encode(current_frame)
        rle_frames.append(output)
    return rle_frames

print("starting RLE")
rle_frames = rle_encode_frames(zigzag_frames)


freq_dict = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, ' ': 0, '-': 0, '*': 0, 'f': 0,}
def calc_char_freq(rle_frame):
    for run_length in rle_frame:
        run, length = run_length
        run_str = str(run)
        for char in run_str:
            freq_dict[char] += 1
        freq_dict[' '] += 1
        length_str = str(length)
        for char in length_str:
            freq_dict[char] += 1
        freq_dict['*'] += 1
    freq_dict['f'] += 1

for f in range(len(rle_frames)):
    current_rle_frame = rle_frames[f]
    calc_char_freq(current_rle_frame)

# huffman
print("starting huffman:")
class node:
    def __init__(self, freq, symbol, left=None, right=None):

        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right
        self.huff = ''

def printNodes(node, val=''):

    newVal = val + str(node.huff)

    if(node.left):
        printNodes(node.left, newVal)
    if(node.right):
        printNodes(node.right, newVal)

    if(not node.left and not node.right):
        print(f"{node.symbol} -> {newVal}")

nodes = []
chars = list(freq_dict.keys())
freq = list(freq_dict.values())

for x in range(len(chars)):
    nodes.append(node(freq[x], chars[x]))

while len(nodes) > 1:
    nodes = sorted(nodes, key=lambda x: x.freq)
    left = nodes[0]
    right = nodes[1]

    left.huff = 0
    right.huff = 1
    newNode = node(left.freq+right.freq, left.symbol+right.symbol, left, right)
    nodes.remove(left)
    nodes.remove(right)
    nodes.append(newNode)

printNodes(nodes[0])

print("huffman finished")




# Now we store coded frames in a file for next functions.

motion_compensated_frames = open("motionCompensated_frames.txt", "a")
for f in range(len(rle_frames)):
    current_frame = rle_frames[f]
    for run_length in current_frame:
        run, length = run_length
        motion_compensated_frames.write(str(run))
        motion_compensated_frames.write(' ')
        motion_compensated_frames.write(str(length))
        motion_compensated_frames.write('*')
    motion_compensated_frames.write('f')
motion_compensated_frames.close()
coded_frames = open("coded_frames.txt", "a")

for f in range(len(rle_frames)):
    current_frame = rle_frames[f]
    for run_length in current_frame:
        run, length = run_length
        coded_frames.write(str(run))
        coded_frames.write(' ')
        coded_frames.write(str(length))
        coded_frames.write('*')
    coded_frames.write('f')
coded_frames.close()



# Decoder
coded_frames_file = open("coded_frames.txt", "r")
content = coded_frames_file.readlines()
coded_frames_file.close()
raw_frames = content[0].split('f')
raw_frames = raw_frames[: len(raw_frames) - 1]

def parse(raw_frame):
    rle_frame = []
    run_length_list = raw_frame.split('*')
    for i in range(len(run_length_list) - 1):
        run_length = run_length_list[i]
        run_length_pair = run_length.split(' ')
        run = run_length_pair[0]
        length =  run_length_pair[1]
        rle_frame.append((int(run), int(length)))
    return rle_frame


def parse_raw_frames(raw_frames):
    rle_frames = []
    for f in range(len(raw_frames)):
        current_raw_frame = raw_frames[f]
        rle_frame = parse(current_raw_frame)
        # rle_frame = np.array(rle_frame)
        rle_frames.append(rle_frame)
    return rle_frames

rle_frames = parse_raw_frames(raw_frames)



# Inverse Run-length Scan

def rle_decode(in_list):
    out_list = []
    for i in range(len(in_list)):
        value, length = in_list[i]
        for j in range(length):
            out_list.append(value)

    return out_list


inverse_zigzag_frames = []
for f in range(len(rle_frames)):
    current_frame = rle_frames[f]
    zigzag = rle_decode(current_frame)
    inverse_zigzag_frames.append(zigzag)



# Inverse Zig-Zag Scan

def inverse_zigzag(input, vmax, hmax):

    i = 0
    h = 0
    v = 0
    vmin = 0
    hmin = 0
    output = np.zeros((vmax, hmax), dtype=int)
    while ((v < vmax) and (h < hmax)):
        if ((h + v) % 2) == 0:                 # going up
            if (v == vmin):
                output[v, h] = input[i]        # if we got to the first line

                if (h == hmax):
                    v = v + 1
                else:
                    h = h + 1
                i = i + 1

            elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
                output[v, h] = input[i]
                v = v + 1
                i = i + 1
            elif ((v > vmin) and (h < hmax -1 )):    # all other cases
                output[v, h] = input[i]
                v = v - 1
                h = h + 1
                i = i + 1

        else:                    # going down

            if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
                output[v, h] = input[i]
                h = h + 1
                i = i + 1

            elif (h == hmin):                  # if we got to the first column
                output[v, h] = input[i]
                if (v == vmax -1):
                    h = h + 1
                else:
                    v = v + 1
                i = i + 1

            elif((v < vmax -1) and (h > hmin)):     # all other cases
                output[v, h] = input[i]
                v = v + 1
                h = h - 1
                i = i + 1

        if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
            output[v, h] = input[i]
            break

    return output


inverse_quantized_frames = []
vmax = gray_frames[0].shape[0]
hmax = gray_frames[0].shape[1]
for f in range(len(inverse_zigzag_frames)):
    zigzag = inverse_zigzag_frames[f]
    inverse_quantized_frame = inverse_zigzag(zigzag, vmax, hmax)
    inverse_quantized_frames.append(inverse_quantized_frame)


# Inverse Quantization
abs_quantized_frames = [np.abs(inverse_quantized_frames[i]) for i in range(len(inverse_quantized_frames))]
abs_quantized_frames = [abs_quantized_frames[i].astype(np.uint32) for i in range(len(transformed_frames))]
shift_transformed_frames = [abs_quantized_frames[i] << 4 for i in range(len(transformed_frames))]
transformed_frames = [shift_transformed_frames[i].astype(np.float32) for i in range(len(transformed_frames))]
transformed_frames = [np.where(is_negative[i] == True, -transformed_frames[i], transformed_frames[i]) for i in range(len(transformed_frames))]


# IDCT
inverse_transformed_frames = []
for f in range(frames_number):
    current_frame = transformed_frames[f]
    current_frame_inverse_transformed = np.empty_like(current_frame, dtype=np.float32)
    frame_height, frame_width = current_frame.shape

    for i in range(0, frame_height, block_height):
        for j in range(0, frame_width, block_width):
            current_block = np.array(current_frame[i: i + block_height, j: j + block_width], dtype=np.float32)
            inverse_transformed_current_block = cv2.idct(current_block)
            current_frame_inverse_transformed[i: i + block_height, j: j + block_width] = inverse_transformed_current_block
    inverse_transformed_frames.append(current_frame_inverse_transformed)

inverse_transformed_frames = [inverse_transformed_frames[i].astype(np.uint8) for i in range(len(inverse_transformed_frames))]


 # Decoding finished
print("Decoding finished")



# Saving decoded frames as video
color_frames = []
for f in range(frames_number):
    frame = inverse_transformed_frames[f]
    color = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    color_frames.append(color)


out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, (960, 544))
for i in range(len(inverse_transformed_frames)):
    out.write(color_frames[i])
out.release()
print("DONE")



def vid2img():
    cam = cv2.VideoCapture("sample 30 frame 1 min.mp4")

    currentframe = 0
    counter = 0

    while (True):
        ret, frame = cam.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = frame.astype(np.float32)
            if(counter%4 != 0 ):
                frame = np.subtract(prev_frame,frame)
            prev_frame = frame

            name = './frames/frame' + str(currentframe) + '.jpg'

            currentframe += 1
            counter += 1
            cv2.imwrite(name, frame)

        else:
            break

    cam.release()
    cv2.destroyAllWindows()
