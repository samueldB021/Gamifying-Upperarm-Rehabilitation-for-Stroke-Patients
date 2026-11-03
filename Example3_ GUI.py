import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
import numpy as np
import threading
import time
import serial.tools.list_ports
from loopdevice import Device  # New Device API
import os
import datetime
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class EMGApp:
    def __init__(self, master):
        self.master = master
        master.title("Loop I - EMG Visualizer")

        # Device & state
        self.device = None
        self.port = None
        self.record = False
        self.folder_path = ""
        self.running = False
        self.sample_rate = 1000
        self.buffer_size = 40
        self.n_channels = 12
        self.time_window = 1.0  # Default to 1 second
        self.record_start_time = None
        self.record_counter = 1
        self.record_status_job = None  # Track the scheduled job
        self.offset_channels = tk.BooleanVar(value=False)  # Plot mode toggle

        # GUI layout
        self.setup_gui()

        # Plotting
        self.fig, (self.ax, self.freq_ax) = plt.subplots(2, 1, figsize=(10, 6))
        self.lines = self.ax.plot(np.zeros((self.buffer_size * int(self.time_window * 1000 / self.buffer_size), self.n_channels)))
        self.update_plot_ylim()
        self.ax.set_xlim(0, self.buffer_size * int(self.time_window * 1000 / self.buffer_size))
        self.ax.set_title("EMG Signal")

        self.freq_line = self.freq_ax.plot(np.zeros((100, self.n_channels)))
        self.freq_ax.set_xlim(0, self.sample_rate // 2)
        self.freq_ax.set_ylim(0, 100000)
        self.freq_ax.set_title("Frequency Response")
        self.freq_ax.set_ylabel("Amplitude")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.emg_buffer = np.zeros((self.buffer_size * int(self.time_window * 1000 / self.buffer_size), self.n_channels))

        self.update_plot_loop()
        self.update_record_status()

    def setup_gui(self):
        top = tk.Frame(self.master)
        top.pack(pady=10)
        self.plot_frame = tk.Frame(self.master)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)
        bottom = tk.Frame(self.master)
        bottom.pack(pady=10)

        tk.Label(top, text="Serial Port:").pack(side=tk.LEFT)
        self.port_combo = ttk.Combobox(top, values=self.get_ports(), width=60)
        self.port_combo.pack(side=tk.LEFT)
        ttk.Button(top, text="Refresh", command=self.refresh_ports).pack(side=tk.LEFT)

        self.connect_button = ttk.Button(top, text="Connect", command=self.toggle_connect)
        self.connect_button.pack(side=tk.LEFT)

        self.acq_button = ttk.Button(bottom, text="Start Acquisition", command=self.toggle_acquisition)
        self.acq_button.pack(side=tk.LEFT, padx=5)

        self.folder_button = ttk.Button(bottom, text="Select Folder", command=self.select_folder)
        self.folder_button.pack(side=tk.LEFT, padx=5)

        self.record_button = ttk.Button(bottom, text="Start Recording", command=self.toggle_record)
        self.record_button.pack(side=tk.LEFT, padx=5)

        tk.Label(bottom, text="Time Window (s):").pack(side=tk.LEFT)
        self.slider = tk.Scale(bottom, from_=0.2, to=10, resolution=0.2, orient=tk.HORIZONTAL, command=self.update_time_window)
        self.slider.set(1.0)
        self.slider.pack(side=tk.LEFT)

        self.freq_checkbox_var = tk.IntVar()
        self.freq_checkbox = ttk.Checkbutton(bottom, text="Show Frequency", variable=self.freq_checkbox_var)
        self.freq_checkbox.pack(side=tk.LEFT)

        self.offset_checkbox = ttk.Checkbutton(bottom, text="Offset Channels", variable=self.offset_channels, command=self.update_plot_ylim)
        self.offset_checkbox.pack(side=tk.LEFT)

        terminal_frame = tk.Frame(self.master)
        terminal_frame.pack(fill=tk.BOTH, expand=False)
        self.terminal = tk.Text(terminal_frame, height=6, bg="black", fg="lime", font=("Courier", 10))
        self.terminal.pack(fill=tk.BOTH, expand=True)
        self.terminal.config(state=tk.DISABLED)

    def log(self, text):
        self.terminal.config(state=tk.NORMAL)
        self.terminal.insert(tk.END, text + "\n")
        self.terminal.see(tk.END)
        self.terminal.config(state=tk.DISABLED)

    def get_ports(self):
        ports = []
        for port in serial.tools.list_ports.comports():
            desc = port.description
            name = port.device
            ports.append(f"{name} - {desc}")
        return ports

    def refresh_ports(self):
        self.port_combo['values'] = self.get_ports()

    def toggle_connect(self):
        if self.device is None:
            def connect_worker():
                self.log("Connecting...")
                port = self.port_combo.get().split(" - ")[0]
                self.device = Device(port)
                for _ in range(5):
                    self.log(".")
                    time.sleep(1)
                try:
                    info = self.device.connect()
                    self.port = port
                    self.connect_button.config(text="Disconnect")
                    self.log(f"Connected to {port}. Channels: {info.n_ch}")
                except Exception as e:
                    self.device = None
                    self.log(f"Connection failed: {e}")

            threading.Thread(target=connect_worker, daemon=True).start()
        else:
            self.device.disconnect()
            self.device = None
            self.connect_button.config(text="Connect")
            self.log("Device disconnected.")

    def toggle_acquisition(self):
        if not self.device:
            messagebox.showerror("Error", "Device not connected")
            return

        if not self.running:
            self.running = True
            if self.record and self.folder_path:
                filename = self.get_next_recording_filename()
                self.device.start_recording(raw_emg_csv=filename)
                self.record_start_time = time.time()
                self.log(f"Recording started. Saving to {filename}")
            self.device.start_acquisition(on_block=self.on_block)
            self.acq_button.config(text="Stop Acquisition")
            self.log("Acquisition started.")
        else:
            self.running = False
            self.device.stop_acquisition()
            self.device.stop_recording()
            self.acq_button.config(text="Start Acquisition")
            self.log("Acquisition stopped.")

    def toggle_record(self):
        if not self.record:
            if not self.folder_path:
                messagebox.showwarning("No Folder", "Please select a folder to save the recording.")
                return
            self.record = True
            self.record_start_time = time.time()
            self.record_button.config(text="Stop Recording")
            if self.running:
                filename = self.get_next_recording_filename()
                self.device.start_recording(raw_emg_csv=filename)
                self.log(f"Recording started. Saving to {filename}")
        else:
            self.record = False
            self.device.stop_recording()
            self.record_button.config(text="Start Recording")
            elapsed = time.time() - self.record_start_time if self.record_start_time else 0
            self.log(f"Recording stopped after {elapsed:.1f} seconds.")
            self.record_start_time = None

    def get_next_recording_filename(self):
        date_str = time.strftime("%d%m%y")
        while True:
            filename = os.path.join(self.folder_path, f"recording_{date_str}_{self.record_counter}.csv")
            if not os.path.exists(filename):
                break
            self.record_counter += 1
        return filename

    def update_record_status(self):
        if self.record and self.record_start_time:
            elapsed = time.time() - self.record_start_time
            self.log(f"Recording... {elapsed:.1f}s")
        self.record_status_job = self.master.after(1000, self.update_record_status)

    def select_folder(self):
        self.folder_path = filedialog.askdirectory()
        self.log(f"Folder selected: {self.folder_path}")

    def update_time_window(self, value):
        self.time_window = float(value)
        new_len = self.buffer_size * int(self.time_window * 1000 / self.buffer_size)
        self.emg_buffer = np.zeros((new_len, self.n_channels))
        self.ax.set_xlim(0, new_len)
        self.ax.cla()
        self.lines = self.ax.plot(np.zeros((new_len, self.n_channels)))
        self.update_plot_ylim()
        self.ax.set_xlim(0, new_len)
        self.ax.set_title("EMG Signal")

        self.freq_ax.cla()
        self.freq_line = self.freq_ax.plot(np.zeros((100, self.n_channels)))
        self.freq_ax.set_xlim(0, self.sample_rate // 2)
        self.freq_ax.set_ylim(0, 100000)
        self.freq_ax.set_title("Frequency Response")
        self.freq_ax.set_ylabel("Amplitude")

        self.canvas.draw()
        self.log(f"Time window updated: {self.time_window} seconds")

    def update_plot_ylim(self):
        if self.offset_channels.get():
            self.ax.set_ylim(0, 34000)
        else:
            self.ax.set_ylim(-2000, 2000)

    def on_block(self, block):
        if block.emg_raw is not None:
            new_data = block.emg_raw
            if new_data.shape[1] != self.n_channels:
                self.n_channels = new_data.shape[1]
                self.update_time_window(self.time_window)
            self.emg_buffer = np.vstack((self.emg_buffer, new_data))[-self.emg_buffer.shape[0]:]

    def update_plot_loop(self):
        if self.running:
            for i, line in enumerate(self.lines):
                if i < self.emg_buffer.shape[1]:
                    if self.offset_channels.get():
                        offset = 2500 * (self.n_channels - i)
                        line.set_ydata(self.emg_buffer[:, i] + offset)
                    else:
                        line.set_ydata(self.emg_buffer[:, i])
            self.update_plot_ylim()

            if self.freq_checkbox_var.get():
                freq_data = np.fft.rfft(self.emg_buffer, axis=0)
                freq_magnitude = np.abs(freq_data)
                freq_bins = np.fft.rfftfreq(self.emg_buffer.shape[0], d=1.0/self.sample_rate)
                self.freq_ax.cla()
                for i in range(min(self.n_channels, len(self.freq_line))):
                    self.freq_ax.plot(freq_bins, freq_magnitude[:, i], label=f"Ch {i+1}")
                self.freq_ax.set_xlim(0, self.sample_rate // 2)
                self.freq_ax.set_ylim(0, 100000)
                self.freq_ax.set_title("Frequency Response")
                self.freq_ax.set_ylabel("Amplitude")
                self.freq_ax.legend(loc="upper center", ncol=self.n_channels, bbox_to_anchor=(0.5, 1.15))

            self.canvas.draw()
        self.master.after(40, self.update_plot_loop)

        

if __name__ == "__main__":
    root = tk.Tk()
    app = EMGApp(root)

    def on_closing():
        if app.record_status_job is not None:
            root.after_cancel(app.record_status_job)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
