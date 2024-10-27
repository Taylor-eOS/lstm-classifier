import os
import tkinter as tk
from tkinter import simpledialog, messagebox
from pydub import AudioSegment
from io import BytesIO
import pygame

class TransitionRefineTool:
    def __init__(self, master, audio_file, timestamps):
        self.master = master
        self.audio_file = audio_file
        self.timestamps = timestamps.copy()
        self.original_timestamps = timestamps.copy()
        self.corrected_timestamps = []
        self.current_timestamp = None
        self.selected_time = 0
        self.colors = ["lightblue", "lightgreen"]
        self.color_index = 0
        pygame.mixer.init()
        self.load_audio()
        self.create_ui()
        self.next_timestamp()

    def load_audio(self):
        try:
            self.audio = AudioSegment.from_file(self.audio_file)
            self.audio_length = len(self.audio) / 1000
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load audio file: {e}")
            self.master.destroy()

    def create_ui(self):
        self.master.title("Refine Transition Timestamps")
        window_width = 1200
        window_height = 150
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        self.master.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.seek_bar = tk.Canvas(self.master, width=1200, height=50, bg=self.colors[self.color_index])
        self.seek_bar.pack(pady=20)
        self.seek_bar.bind("<Button-1>", self.on_seek_click)
        self.save_button = tk.Button(self.master, text="Save Timestamp", command=self.save_timestamp)
        self.save_button.pack(pady=10)
        self.master.bind('<Return>', lambda e: self.save_timestamp())
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

    def play_audio_segment(self, start_ms):
        pygame.mixer.music.stop()
        segment = self.audio[start_ms:start_ms + 20000]
        wav_io = BytesIO()
        segment.export(wav_io, format="wav")
        wav_io.seek(0)
        pygame.mixer.music.load(wav_io)
        pygame.mixer.music.play()

    def on_seek_click(self, event):
        click_ratio = event.x / self.seek_bar.winfo_width()
        window_start = max(0, self.current_timestamp - 10)
        window_end = min(self.audio_length, self.current_timestamp + 10)
        window_duration = window_end - window_start
        clicked_time = window_start + click_ratio * window_duration
        self.selected_time = clicked_time
        self.play_audio_segment(int(clicked_time * 1000))
        self.update_seek_bar(clicked_time, window_start, window_end)

    def update_seek_bar(self, clicked_time, window_start, window_end):
        self.seek_bar.delete("indicator")
        click_ratio = (clicked_time - window_start) / (window_end - window_start)
        x_pos = click_ratio * self.seek_bar.winfo_width()
        self.seek_bar.create_line(x_pos, 0, x_pos, 50, fill="red", width=2, tags="indicator")

    def save_timestamp(self):
        self.corrected_timestamps.append(round(self.selected_time))
        self.color_index = (self.color_index + 1) % len(self.colors)
        self.seek_bar.config(bg=self.colors[self.color_index])
        self.next_timestamp()

    def next_timestamp(self):
        if not self.timestamps:
            self.finish()
            return
        self.current_timestamp = self.timestamps.pop(0)
        window_start = max(0, self.current_timestamp - 10)
        self.play_audio_segment(int(window_start * 1000))
        self.seek_bar.delete("indicator")
        self.seek_bar.config(bg=self.colors[self.color_index])

    def finish(self):
        pygame.mixer.quit()
        print("Corrected Timestamps (in seconds):")
        print(" ".join(map(str, self.corrected_timestamps)))
        differences = [int(c) - int(o) for c, o in zip(self.corrected_timestamps, self.original_timestamps)]
        print("Differences (seconds):")
        print(" ".join(map(str, differences)))
        messagebox.showinfo("Done", "Corrected timestamps printed to console")
        self.master.destroy()

    def on_close(self):
        pygame.mixer.quit()
        self.master.destroy()

def main():
    root = tk.Tk()
    root.withdraw()
    file_path = simpledialog.askstring("Audio File", "Enter audio file path (MP3 or WAV):")
    if not file_path or not os.path.exists(file_path):
        messagebox.showerror("Error", "File not found")
        return
    raw_timestamps = simpledialog.askstring("Timestamps", "Enter timestamps in seconds (space-separated):")
    if not raw_timestamps:
        messagebox.showerror("Error", "No timestamps provided")
        return
    try:
        timestamps = [float(t) for t in raw_timestamps.strip().split()]
    except ValueError:
        messagebox.showerror("Error", "Invalid timestamp format")
        return
    root.deiconify()
    app = TransitionRefineTool(root, file_path, timestamps)
    root.mainloop()

if __name__ == "__main__":
    main()

