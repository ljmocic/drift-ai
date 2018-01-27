from keys import PressKey, ReleaseKey, W, A, D, SHIFT, ESC, ENTER, DOWN
from PIL import Image, ImageGrab
import PIL
import numpy as np
import time as t
import cv2
import pyocr
import threading
import time


# setup tesseract
tools = pyocr.get_available_tools()
tool = tools[0]
builder = pyocr.builders.DigitBuilder()
# https://github.com/tesseract-ocr/tesseract/wiki/Command-Line-Usage
# 8 Treat the image as a single word.
builder.tesseract_flags = ['-psm', '8']

highscore = 0
iteration = 0
episode = 0
states = 17
active_state = 0
actions_array = [A, SHIFT, D]
actions_history = []
# Q = np.zeros([states, len(actions_array)])
Q = 1000 * np.random.rand(states, len(actions_array))

# learning parameters
lr = 0.5
y = 0.5

# settings
keypress_pause = 0.3


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
    #img = remove_comma(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = 255 - img
    #ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    return img


def print_action(chosen_aciton):
    print("Action: ")
    if chosen_aciton == 0:
        print("<-")
    elif chosen_aciton == 1:
        print("^")
    elif chosen_aciton == 2:
        print("->")


def run_q_algorithm():
    global active_state, highscore, episode, actions_history
    chosen_action = np.argmax(Q[active_state, :])
    actions_history.append(chosen_action)
    print("Value: ")
    print(Q[active_state, chosen_action])
    control_car(chosen_action)
    print_action(chosen_action)
    reward = update_reward()
    Q[active_state, chosen_action] = (1 - lr) * Q[active_state, chosen_action] + lr * (highscore + y * np.max(Q[(active_state + 1), :]))
    print(Q)

    # if it is not final state, update active state, otherwise reset race
    if active_state + 2 != states:
        active_state += 1
    else:
        reset_race()
        episode += 1
        print("Episode " + str(episode))
        np.savetxt("qmatrix" + str(highscore) + ".out", Q, delimiter=',')
        active_state = 0
        actions_history = []
        highscore = 0


def update_reward():
    global highscore

    # grab piece of screen for processing
    img = np.array(ImageGrab.grab(bbox=(1000, 200, 1100, 222)))

    # preprocess frame for easier tesseract detection
    img = process_frame(img)

    # detect number on processed image using tesseract ocr
    detected_number = tool.image_to_string(PIL.Image.fromarray(img), lang="eng", builder=builder)
    detected_number = "".join(i for i in detected_number if i in "0123456789")

    # successfully detected a number, add to highscore and update state reward
    if detected_number.isnumeric():
        if int(detected_number) > highscore:
            highscore = int(detected_number)

            # reward the path of taken actions
            for i in range(len(actions_history)):
                Q[i][actions_history[i]] += highscore
                
            print("=== Highscore: " + str(highscore) + " ====")

    # Show detection frame for debugging
    cv2.imshow("frame", img)


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

# Q = np.loadtxt('qmatrix226157.out', delimiter=',')
# print(Q)

while(True):
    run_q_algorithm()
    print("Iteration" + str(iteration))
    iteration += 1
    
    key = cv2.waitKey(1)
    if key == 27:
        for action in actions_array:
            ReleaseKey(action)
        break

cv2.destroyAllWindows()

