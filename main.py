from keys import PressKey, ReleaseKey, W, A, D, SHIFT, ESC, ENTER, DOWN
from PIL import Image, ImageGrab
import PIL
import numpy as np
import time as t
import cv2
import pyocr

# setup tesseract
tools = pyocr.get_available_tools()
tool = tools[0]
builder = pyocr.builders.DigitBuilder()
# https://github.com/tesseract-ocr/tesseract/wiki/Command-Line-Usage
# 8 Treat the image as a single word.
builder.tesseract_flags = ['-psm', '8']

# score detected by tesseract ocr
highscore = 0
iteration = 0
states = 20
active_state = 0
actions_array = [SHIFT, A, D]
Q = np.zeros([states, len(actions_array)])

# learning parameters
lr = 0.25
y = 0.3

# settings
keypress_pause = 0.1


def remove_comma(detected_score):
    digits = []
    digits.append(detected_score[:, 45:55])
    digits.append(detected_score[:, 36:45])
    digits.append(detected_score[:, 28:36])
    digits.append(detected_score[:, 15:24])
    digits.append(detected_score[:, 6:15])
    detected_score = np.concatenate(
        (digits[4], digits[3], digits[2], digits[1], digits[0]), axis=1)
    return detected_score


def control_car(chosen_action):
    for action in actions_array:
        ReleaseKey(action)

    # Simplified state-action with
    # turning on constant accelerating
    # TODO remove after experimenting
    PressKey(W)

    # Perform action
    PressKey(actions_array[chosen_action])


def process_frame(img):
    img = remove_comma(img)
    # TODO apply more filters for easier detection
    return img


def run_q_algorithm():
    global active_state
    chosen_action = np.argmax(Q[active_state, :])

    control_car(chosen_action)
    reward = update_reward()

    Q[active_state, chosen_action] = (1 - lr) * Q[active_state, chosen_action] + lr * (reward + y * np.max(Q[(active_state + 1), :]))
    # print(Q)

    # if it is not final state, update active state, otherwise reset race
    if active_state + 2 != states:
        active_state += 1
    else:
        reset_race()
        active_state = 0
        highscore = 0


def update_reward():

    # grab piece of screen for processing
    screenshot = np.array(ImageGrab.grab(bbox=(650, 300, 710, 330)))

    # preprocess frame for easier tesseract detection
    img = process_frame(screenshot)

    # detect number on processed image using tesseract ocr
    detected_number = tool.image_to_string(PIL.Image.fromarray(img), lang="eng", builder=builder)

    reward = 0

    # successfully detected a number, add to highscore and update state reward
    if detected_number.isnumeric():
        detected_number = int(detected_number)
        print(detected_number)
        highscore += detected_number
        reward = detected_number

    # Show detection frame for debugging
    cv2.imshow("frame", img)

    # Return reward of this frame
    return reward


def reset_race():
    # Run keys pattern to reset race
    PressKey(ESC)
    t.sleep(keypress_pause)
    ReleaseKey(ESC)
    PressKey(ENTER)
    t.sleep(keypress_pause)
    ReleaseKey(ENTER)
    PressKey(DOWN)
    t.sleep(keypress_pause)
    ReleaseKey(DOWN)
    PressKey(ENTER)
    t.sleep(keypress_pause)
    ReleaseKey(ENTER)
    t.sleep(3 * keypress_pause)


while(True):
    run_q_algorithm()
    # print("Iteration" + str(iteration))
    iteration += 1

    key = cv2.waitKey(1)
    if key == 27:
        for action in actions:
            ReleaseKey(action)
        break

cv2.destroyAllWindows()
