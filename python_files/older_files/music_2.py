import librosa
import numpy as np
import torch
import time
from music_NN import Model
import pygame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# The neural network has 91% accuracy on the training data

# song_name = "katawaredoki"  # .mp3
# song_name = "twinkle_twinkle_little_star"
song_name = "katawaredoki"
song_path = "other_files/"
song_format = ".mp3"
note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
colors_rgb = [(68, 102, 255), (255, 153, 0), (17, 255, 255), (255, 85, 143),
              (0, 255, 0), (204, 51, 255), (255, 255, 0), (0, 190, 255),
              (255, 51, 68), (0, 255, 210), (255, 102, 238), (183, 255, 0)]

# colors = [
#     (255, 0, 0), (255, 127, 0), (255, 255, 0),
#     (127, 255, 0), (0, 255, 0), (0, 255, 127),
#     (0, 255, 255), (0, 127, 255), (0, 0, 255),
#     (127, 0, 255), (255, 0, 255), (255, 0, 127)
# ]
# new_colors = []
# i = 0
# for _ in range(12):
#     new_colors.append(colors[i % 12])
#     i += 7
# colors_rgb = new_colors

print("Loading " + song_name + "...")

samples, sr = librosa.load(song_path + song_name + song_format)

print("Starting animation...")

# Load the neural network from music_NN.py
model = Model()
model.load_state_dict(torch.load("NN_parameters/music_NN_6.pth", map_location="cpu"))
model.eval()

# Define preprocessing function
len_window = 8192  # ~ 0.371s
jump_size = 2048  # ~ 0.092s

def process_notes(idx):
    """Outputs the (12, ) np.array with the probabilities of
    each note in the section x[idx: idx + len_window]"""
    assert idx + len_window < len(samples) - 2, "Song ended."

    with torch.no_grad():
        chunk = samples[idx: idx + len_window].reshape((1, -1))
        chunk /= np.linalg.norm(chunk) + 10e-8
        y_pred = model(torch.from_numpy(chunk))
    return y_pred[0].numpy()

def find_top_notes(notes_prob, threshold=0.22):
    """Given a (12, ) np.array, returns:
    a) top notes with probability above the threshold (sorted)
    b) list (12, ) with the scaled values of rgb color by probability
    c) a message"""
    top_notes = [i for i, p in enumerate(notes_prob) if p > threshold]
    top_notes = sorted(top_notes, key=lambda i: notes_prob[i], reverse=True)

    message = ""
    for note_id in top_notes:
        message += f"{note_names[note_id]}: {round(float(notes_prob[note_id]), 2)}     "

    scaled_colors = []
    for note_id in top_notes:
        note_p = notes_prob[note_id]
        alpha = min(1, (note_p - threshold)/(0.75 - threshold + 10e-8))  # factor
        triple = colors_rgb[note_id]
        scaled_colors.append((alpha*triple[0], alpha*triple[1], alpha*triple[2]))
    return top_notes, scaled_colors, message


# Initialize Pygame
pygame.init()
pygame.mixer.init()
sq_size = 250  # size of a note in the screen
width, height = 4*sq_size, 3*sq_size

window = pygame.display.set_mode((width, height))
pygame.display.set_caption(song_name + " animation")

pygame.mixer.music.load(song_path + song_name + song_format)

# Main loop
running = True
data_processed = False  # turns True in each iteration
idx = jump_size  # processing index

# Start playing the song
pygame.mixer.music.play()
start_time = time.time()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Extract note probabilities if not done already
    if not data_processed:
        try:
            notes_prob = process_notes(idx)  # call the model
            top_notes, scaled_colors, message = find_top_notes(notes_prob)
        except AssertionError:  # end of the song
            running = False
        data_processed = True

    # Update the animation if it's the right time
    current_time = time.time()
    time_left = idx/sr - (current_time - start_time)
    if time_left <= 0:
        # Update the screen with the processed image
        window.fill((0, 0, 0))
        for note_id, note_color in zip(top_notes, scaled_colors):
            x, y = sq_size*(note_id % 4), sq_size*(note_id//4)  # get position
            pygame.draw.rect(window, note_color, (x, y, sq_size, sq_size))
        pygame.display.flip()

        print(message)

        # Update which notes to process
        idx += jump_size
        data_processed = False
    else:
        time.sleep(time_left)

# Quit Pygame
pygame.mixer.quit()
pygame.quit()
