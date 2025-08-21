import librosa
import numpy as np
import matplotlib.pyplot as plt

notes_frequencies = [('C2', 65.41), ('C#2', 69.3), ('D2', 73.42), ('D#2', 77.78), ('E2', 82.41), ('F2', 87.31), ('F#2', 92.5),
                     ('G2', 98.0), ('G#2', 103.83), ('A2', 110.0), ('A#2', 116.54), ('B2', 123.47), ('C3', 130.81), ('C#3', 138.59), 
                     ('D3', 146.83), ('D#3', 155.56), ('E3', 164.81), ('F3', 174.61), ('F#3', 185.0), ('G3', 196.0), ('G#3', 207.65),
                     ('A3', 220.0), ('A#3', 233.08), ('B3', 246.94), ('C4', 261.63), ('C#4', 277.18), ('D4', 293.66), ('D#4', 311.13),
                     ('E4', 329.63), ('F4', 349.23), ('F#4', 369.99), ('G4', 392.0), ('G#4', 415.3), ('A4', 440.0), ('A#4', 466.16),
                     ('B4', 493.88), ('C5', 523.25), ('C#5', 554.37), ('D5', 587.33), ('D#5', 622.25), ('E5', 659.25), ('F5', 698.46),
                     ('F#5', 739.99), ('G5', 783.99), ('G#5', 830.61), ('A5', 880.0), ('A#5', 932.33), ('B5', 987.77), ('C6', 1046.5)]

def binary_search(L, x):
    """returns the smallest index i such that L[i] >= x, and -1 otherwise
    assumes L is in increasing order"""
    if L[-1] < x:
        return -1
    if L[0] >= x:
        return 0
    low, high = 0, len(L)-1
    while low < high-1:
        mid = (low+high)//2
        if L[mid] < x:
            low = mid
        else:
            high = mid
    return high

def load_katawaredoki(t0, tf):
    """returns an np.array with the first t <= 19.5 seconds of the song
    sampling frequency is 22050"""
    array = []
    with open("other_files/katawaredoki.txt", "r") as file:
        i = 0
        for num in file:
            if i < t0*22050:
                i += 1
                continue
            elif i > tf*22050:
                break
            array.append(np.float64(num))
            i += 1
    return np.array(array)

# katawaredoki = load_katawaredoki(0, 12)


def plot_sequence_fft(x, padding = False, plot_notes = False):
    """x is a 1-d np.ndarray, plots x, the fourier transform, and its logarithm
    if padding, adds zeros at the end of the array x to make the transforms smoother
    if plot_notes, adds a line near each note in the log-frequency domain"""
    N = len(x)
    time_signal = N/22050
    if padding:
        time_signal *= 2
        x = list(x)+[0 for _ in range(N)]
        x = np.array(x)
        N *= 2
    T = np.linspace(0, time_signal-1/22050, N)      # time axis
    F = np.arange(1+N//20)/time_signal              # frequency axis
    X = np.abs(np.fft.fft(x)[:1+N//20])             # fourier transform
    fig, ax = plt.subplots(ncols=1, nrows=3)
    ax[0].plot(T, x)
    ax[0].set_title("Time domain")
    ax[1].scatter(F, X)
    ax[1].set_title("Frequency domain")
    def f(y):
        return (np.log(y)-np.log(65.41))/np.log(2)
    F = f(F)                                        # log frequency axis
    n = len(F)
    F = F[n//20:]                                   # truncate values
    X = X[n//20:]
    ax[2].scatter(F, X)
    ax[2].set_title("Frequency domain with logarithmic x axis")
    if plot_notes:
        for i in range(len(notes_frequencies)):
            freq = notes_frequencies[i][1]
            log_freq = f(freq)
            indx_low = binary_search(F, log_freq-0.015)
            indx_high = binary_search(F, log_freq+0.015)
            maxval = max(X[indx_low:indx_high])
            ax[2].plot([log_freq-0.015, log_freq+0.015], [maxval, maxval], c="C1")
    plt.show()

# plot_sequence_fft(katawaredoki, True, True)

def importance_frequence(y, i, n = 10, plot = False):
    """plots the value of the fft at the highest point near the frequency at n different times"""
    def f(y):
        return (np.log(y)-np.log(65.41))/np.log(2)
    note, freq = notes_frequencies[i]
    log_freq = f(freq)
    values = [0]
    for j in range(n):
        x = y[j*len(y)//n:(j+1)*len(y)//n]
        N = len(x)
        time_signal = N/11025
        x = list(x)+[0 for _ in range(N)]
        x = np.array(x)
        N *= 2
        F = np.arange(1+N//20)/time_signal
        X = np.abs(np.fft.fft(x)[:1+N//20])
        F = f(F)
        m = len(F)
        F = F[m//20:]
        X = X[m//20:]

        indx_low = binary_search(F, log_freq-0.015)-1
        indx_high = binary_search(F, log_freq+0.015)
        maxval = max(X[indx_low:indx_high])             # local maximum near the note
        values.append(maxval)
    time_axis = np.linspace(0, len(y)/22050, n+1)
    if plot:
        plt.plot(time_axis, values)
        plt.title(f"Importance of the freq {note} from 0 to t in the x-axis")
        plt.show()
    return (time_axis, values)

def importance_frequencies(y, L, n = 10):
    if len(L) >= 4:
        m = int(np.sqrt(len(L)))
        fig, ax = plt.subplots(nrows=m, ncols=len(L)//m)
        for j in range(len(L)):
            i = L[j]
            time_axis, values = importance_frequence(y, i, n, False)
            ax[j % m, j//m].plot(time_axis, values)
            ax[j % m, j//m].set_title(f"Frequency {notes_frequencies[i][0]}")
    else:
        fig, ax = plt.subplots(nrows=len(L), ncols=1)
        for j in range(len(L)):
            i = L[j]
            time_axis, values = importance_frequence(y, i, n, False)
            ax[j].plot(time_axis, values)
            ax[j].set_title(f"Frequency {notes_frequencies[i][0]}")
    plt.show()

# importance_frequencies(katawaredoki, [36, 43, 24, 19], 40)

def value_frequence(x, i):
    "The idea is that x is short, I will use it of 0.5 seconds"
    freq = notes_frequencies[i][1]
    def f(y):
        return (np.log(y)-np.log(65.41))/np.log(2)
    log_freq = f(freq)
    N = len(x)
    time_signal = N/11025
    x = list(x)+[0 for _ in range(N)]
    x = np.array(x)
    N *= 2
    F = np.arange(1+N//20)/time_signal
    X = np.abs(np.fft.fft(x)[:1+N//20])
    F = f(F)
    m = len(F)
    F = F[m//20:]
    X = X[m//20:]
    indx_low = binary_search(F, log_freq-0.015)-1
    indx_high = binary_search(F, log_freq+0.015)
    return max(X[indx_low:indx_high])


def frequencies_in(y):
    """Assumes a note is significant if in 0.5 seconds has a value of at least 50"""
    return sorted([(notes_frequencies[i][0], i, value_frequence(y, i)) for i in range(len(notes_frequencies))], key=lambda x: -x[-1])

# freqs = frequencies_in(katawaredoki)

def plot_all_freq(y, L, n = 10):
    j = 0
    for i in L:
        x_vals, y_vals = importance_frequence(y, i, n, False)
        plt.plot(x_vals, y_vals, label=notes_frequencies[i][0], c="C"+str(j % 4))
        j += 1
    plt.legend(loc="best")
    plt.show()

def plot_all_notes(y, L, n = 10):
    """adds the contributions of frequencies that represent the same note"""
    j = 0
    fig, ax = plt.subplots()
    for i in L:
        i = i % 12
        time_axis, values1 = importance_frequence(y, i, n, False)
        values1 = np.array(values1)
        values2 = np.array(importance_frequence(y, i+12, n, False)[1])
        values3 = np.array(importance_frequence(y, i+24, n, False)[1])
        values4 = np.array(importance_frequence(y, i+36, n, False)[1])
        values = values1+values2+values3+values4
        ax.plot(time_axis, values, label=notes_frequencies[i][0][:-1], c="C"+str(j))
        j += 1
    plt.legend(loc="best")
    plt.show()

# plot_all_freq(katawaredoki, [9, 12, 17, 19], 40)
# plot_all_notes(katawaredoki, [9, 0, 5, 7], 40)


# y, sr = librosa.load("other_files/katawaredoki.mp3")
# print(y.shape)

# seconds = 19.5
# Spect = np.abs(librosa.stft(y[:int(seconds*sr)]))            # is (1025, 517)
# height, width = Spect.shape

# plt.imshow(Spect, cmap='vidris')
# plt.show()

# for i in range(12):               # more prevalent notes in the first 12 seconds
#     y_vals = np.zeros(width)
#     for j in [i, i+12, i+24, i+36]:
#         note, freq = notes_frequencies[j]
#         start, end = int(freq*(2048/sr)/2**(1/48)), int(freq*(2048/sr)*2**(1/48))
#         y_vals += np.max(Spect[start:end+1], axis=0)
#     plt.plot(np.linspace(0, seconds, width), y_vals, label=f"Strength of the note {note[:-1]}")

# for i in [9, 24, 29, 19]:
#     note, freq = notes_frequencies[i]
#     start, end = int(freq*(2048/sr)/2**(1/48)), int(freq*(2048/sr)*2**(1/48))
#     y_vals = np.max(Spect[start:end+1], axis=0)
#     plt.plot(np.linspace(0, seconds, width), y_vals, label=f"Strength of the frequency {note}")


# plt.plot(np.arange(1025)*sr/2048, Spect[:,517*4//12])


# Spect = librosa.amplitude_to_db(Spect, ref=np.max)
# fig, ax = plt.subplots()
# img = librosa.display.specshow(Spect, x_axis="time", y_axis="log", ax=ax)
# ax.set_title("Katawaredoki, first 12 seconds")
# fig.colorbar(img, ax=ax, format=f"%0.2f")
# plt.legend(loc="best")
# plt.show()











# new stuff --------------------------------------------------------------------

# recall that:
# if n is the len of the data
# there's n/sr seconds, in which there are n*fr/sr revolutions
# these finish at indices sr/fr and their multiples
# so they're modeled by exp(i*2*pi * x*fr/sr) from x=0 to x=n-1

# In the np implementation, A_k represents the function exp(2*pi*i * x*k/n)
# So we have k = n*fr/sr


# Animation of DTFT


# import pygame
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
# import time
import librosa

samples, sr = librosa.load("other_files/katawaredoki.mp3")

# # Initialize Pygame
# pygame.init()

# # Set up display
# width, height = 1200, 800
# window = pygame.display.set_mode((width, height))
# pygame.display.set_caption("Dynamic Matplotlib Plot in Pygame")

# Parameters
idx = 0
len_window = 8192  # ~ 0.371s
jump_size = 2048  # ~ 0.092s

# Create a Matplotlib figure and axis
fig, ax = plt.subplots()
# fig.set_size_inches(12, 8)
# canvas = FigureCanvas(fig)
x = np.linspace(0, len_window - 1, len_window)
# y = np.zeros(len_window)

idx = 8000
portion = samples[idx:idx+len_window]

note_idx = 1
fr = 32.7031998896
while note_idx < 7:
    x_axis = []
    y_axis = []
    factor = 25
    for i in range(-factor, factor):
        fr_i = fr*2**(i/(12*factor))
        value_i = np.abs(np.dot(portion, np.exp(-2j*np.pi * x*fr_i/sr)))
        y_axis.append(value_i)
        x_axis.append(fr_i)
    fr *= 2
    note_idx += 1

# 110, 220, 440
# E 329
# C# 554
# G 784
# B 987

# x_axis = np.linspace(22, 400-1, 400-22)*sr/len_window
# y_axis = np.abs(np.fft.rfft(samples[idx: idx + len_window]))[22:400]
    plt.plot(x_axis, y_axis)
    print(f"C{note_idx} at {fr}Hz")
    plt.show()

# line, = ax.plot(np.linspace(0, 1023, 1024), y)

# fr = 196
# G3_vec = np.exp(-2j*np.pi * x*fr/sr)

# # Function to convert Matplotlib figure to Pygame surface
# def draw_matplotlib_to_pygame(fig, canvas):
#     canvas.draw()
#     renderer = canvas.get_renderer()
#     raw_data = renderer.tostring_rgb()
#     size = canvas.get_width_height()
#     return pygame.image.fromstring(raw_data, size, "RGB")

# # Main loop
# running = True
# processed = False
# start_time = time.time()

# while running:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False
    
#     if not processed:
#         chunk = samples[idx: idx + len_window]
#         freq_dom = np.abs(np.fft.rfft(chunk))[:1024]
#         y = np.abs(np.dot(chunk, G3_vec))
#         processed = True

#     current_time = time.time()
#     time_left = idx/sr - (current_time - start_time)
#     if time_left <= 0:
#         # Update the plot
#         line.set_ydata(freq_dom)
#         ax.relim()
#         ax.autoscale_view()

#         # Convert Matplotlib plot to Pygame surface
#         plot_surface = draw_matplotlib_to_pygame(fig, canvas)

#         # Clear the screen and blit the plot surface
#         window.fill((255, 255, 255))
#         window.blit(plot_surface, (0, 0))

#         # Update the display
#         pygame.display.flip()
#         print(y)

#         # Control the frame rate
#         idx += jump_size
#         processed = False
#     else:
#         time.sleep(time_left)

# pygame.quit()



# Attempt with just convolution

