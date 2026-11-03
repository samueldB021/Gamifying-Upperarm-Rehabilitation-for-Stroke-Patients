from __future__ import annotations
import threading
import time
from queue import Queue, Empty
from typing import Callable, Optional
import serial
import numpy as np
import sys
import platform
import subprocess

from .types import DeviceInfo, SampleBlock
from .exceptions import ConnectionError, HandshakeError

OnBlockCallback = Callable[[SampleBlock], None]

class Device:
    """
    Thin, thread-safe Python API to your Loop I.
    """

    DEFAULT_BAUDRATE = 115200
    HAND_SEQUENCE = (1, 2, 3)  # replace with your real sequence

    def __init__(self, port: str, *, baudrate: int = DEFAULT_BAUDRATE, timeout: float = 1.0):
        self._port = port
        self._baudrate = baudrate
        self._timeout = timeout

        self._ser: Optional[serial.Serial] = None
        self._is_acquiring = False
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

        self._device_info: Optional[DeviceInfo] = None
        self._batch_idx = 0

        self._queue: Queue[SampleBlock] = Queue(maxsize=20000)
        self._callback: Optional[OnBlockCallback] = None

        # file handles for recording (optional)
        self._bin_emg = None
        self._bin_rms = None
        self._bin_kin = None
        self._csv_emg = None
        self._csv_rms = None
        self._csv_kin = None

    # ------------- Public API -------------

    def connect(self) -> DeviceInfo:
        """Open the serial port and query device info."""
        try:
            self._ser = serial.Serial(self._port, self._baudrate, timeout=self._timeout)
        except Exception as e:
            raise ConnectionError(f"Could not open serial port {self._port}: {e}") from e

        if not self._ser.is_open:
            raise ConnectionError(f"Serial port {self._port} is not open")

        self._device_info = self._get_device_info()
        return self._device_info

    def disconnect(self) -> None:
        """Close the serial port."""
        self.stop_acquisition()
        if self._ser and self._ser.is_open:
            self._ser.setDTR(False)
            self._ser.setRTS(False)
            self._ser.flush()
            self._ser.close()
        if sys.platform == "darwin":
            self.restart_bluetoothd()

    def start_acquisition(self, *, on_block: Optional[OnBlockCallback] = None) -> None:
        """Start background acquisition. Optionally pass a per-block callback."""
        if not self._device_info:
            raise ConnectionError("Device not connected. Call connect() first.")

        if self._is_acquiring:
            return

        self._callback = on_block
        self._is_acquiring = True
        self._stop_evt.clear()
        self._batch_idx = 0

        # Tell device to start
        self._write(b'p')  # pause/clear?
        self._ser.flush()
        self._write(b's')  # start

        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def stop_acquisition(self) -> None:
        """Stop acquisition and wait for the thread to end."""
        if not self._is_acquiring:
            return

        self._write(b'p')  # stop
        self._ser.flush()

        self._stop_evt.set()
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join()

        self._is_acquiring = False

    def read(self, timeout: Optional[float] = None) -> SampleBlock:
        """
        Blocking read of the next SampleBlock from an internal queue.
        Useful if you don't want to register a callback.
        """
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            raise TimeoutError("Timed out waiting for next sample block")

    def get_device_info(self) -> DeviceInfo:
        if not self._device_info:
            raise ConnectionError("Device not connected yet")
        return self._device_info

    def start_recording(
        self,
        *,
        raw_emg_bin: Optional[str] = None,
        rms_bin: Optional[str] = None,
        kin_bin: Optional[str] = None,
        raw_emg_csv: Optional[str] = None,
        rms_csv: Optional[str] = None,
        kin_csv: Optional[str] = None,
    ) -> None:
        """Open files for recording selected streams."""
        if raw_emg_bin:
            self._bin_emg = open(raw_emg_bin, "wb")
        if rms_bin:
            self._bin_rms = open(rms_bin, "wb")
        if kin_bin:
            self._bin_kin = open(kin_bin, "wb")
        if raw_emg_csv:
            self._csv_emg = open(raw_emg_csv, "w", buffering=1)
        if rms_csv:
            self._csv_rms = open(rms_csv, "w", buffering=1)
        if kin_csv:
            self._csv_kin = open(kin_csv, "w", buffering=1)

    def stop_recording(self) -> None:
        """Close any open recording files."""
        for f in (self._bin_emg, self._bin_rms, self._bin_kin,
                  self._csv_emg, self._csv_rms, self._csv_kin):
            if f:
                f.close()
        self._bin_emg = self._bin_rms = self._bin_kin = None
        self._csv_emg = self._csv_rms = self._csv_kin = None

    def __enter__(self) -> "Device":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.stop_recording()
        finally:
            self.disconnect()

    # ------------- Internals -------------

    def _write(self, data: bytes) -> None:
        assert self._ser is not None
        self._ser.write(data)

    def _read_loop(self) -> None:
        """Continuously read blocks until stopped."""
        assert self._device_info is not None
        info = self._device_info

        while not self._stop_evt.is_set():
            # Optional handshake
            if info.is_handshake:
                if not self._handshake():
                    # Could raise, or just continue
                    print("[Device] Wrong package (handshake failed)")
                    continue  # skip this cycle, do not push a block

            block = self._read_block(info)
            self._batch_idx += 1
            block = SampleBlock(
                *block[:-1],
                battery_level=block[-1],
                batch_index=self._batch_idx,
            )

            # Recording (optional)
            self._record_block(block)

            # Deliver to user
            if self._callback:
                try:
                    self._callback(block)
                except Exception as e:
                    # Avoid killing the read thread due to user callback errors
                    print(f"[Device] on_block callback error: {e}")

            try:
                self._queue.put_nowait(block)
            except:
                # Queue full; drop oldest
                _ = self._queue.get_nowait()
                self._queue.put_nowait(block)

    def _read_block(self, info: DeviceInfo):
        """Read and parse one complete block from the device."""
        assert self._ser is not None
        rd = self._ser.read

        def safe_read(n):
            data = rd(n)
            if len(data) != n:
                raise IOError(f"Expected {n} bytes, got {len(data)}")
            return data

        # EMG raw data
        emg = None
        if info.n_ch > 0:
            emg = np.zeros((info.acq_buffer_size, info.n_ch), dtype=np.int16)
            for ch in range(info.n_ch):
                for sample in range(info.acq_buffer_size):
                    emg[sample][ch] = int.from_bytes(safe_read(2), "little", signed=True)

        # Aux data
        aux = None
        if info.aux_ch > 0:
            aux = np.zeros((info.acq_buffer_size, info.aux_ch), dtype=np.int16)
            for ch in range(info.aux_ch):
                for sample in range(info.acq_buffer_size):
                    aux[sample][ch] = int.from_bytes(safe_read(2), "little", signed=True)

        # Regs
        regs = None
        if info.n_dofs > 0:
            regs = np.zeros(info.n_dofs, dtype=np.int16)
            for i in range(info.n_dofs):
                regs[i] = int.from_bytes(safe_read(2), "little", signed=True)

        # Norm
        norm = None
        if info.norm_chs > 0:
            norm = np.zeros(info.norm_chs, dtype=np.float32)
            for i in range(info.norm_chs):
                norm[i] = int.from_bytes(safe_read(2), "little", signed=True) / 10.0

        # RMS
        rms = None
        if info.rms_chs > 0:
            rms = np.zeros(info.rms_chs, dtype=np.int16)
            for i in range(info.rms_chs):
                rms[i] = int.from_bytes(safe_read(2), "little", signed=True)

        # Haptic (Stimulation)
        haptic = None
        if info.sti_ch > 0:
            haptic = np.zeros(info.sti_ch, dtype=np.int16)
            for i in range(info.sti_ch):
                haptic[i] = int.from_bytes(safe_read(2), "little", signed=True)

        # Kinematics
        kin = None
        if info.kin_ch > 0:
            kin = np.zeros(info.kin_ch, dtype=np.int16)
            for i in range(info.kin_ch):
                kin[i] = int.from_bytes(safe_read(2), "little", signed=True)

        # Battery
        battery = None
        if info.is_battery:
            battery = int.from_bytes(safe_read(2), "little", signed=True)

        return (emg, aux, regs, norm, rms, haptic, kin, battery)

    def _handshake(self) -> bool:
        """Check the 3-value handshake sequence."""
        assert self._ser is not None
        seq = np.frombuffer(self._ser.read(3 * 2), dtype="<i2")  # reading int16
        if tuple(int(v) for v in seq) == self.HAND_SEQUENCE:
            return True

        # Attempt to re-sync – naive version
        # (you can copy your original while-loop logic if needed)
        return False

    def _get_device_info(self) -> DeviceInfo:
        """Stop transmission, clear buffer, issue config command, read info."""
        assert self._ser is not None

        self._write(b'p')
        time.sleep(1)
        self._write(b'c0')
        # drain?
        time.sleep(3)
        _ = self._ser.in_waiting

        def read_u16():
            return int.from_bytes(self._ser.read(2), "little")

        is_master         = read_u16()
        is_slave_available= read_u16()
        sample_rate       = read_u16()
        acq_buffer_size   = read_u16()
        n_ch              = read_u16()
        aux_ch            = read_u16()
        n_dofs            = read_u16()
        norm_chs          = read_u16()
        rms_chs           = read_u16()
        sti_ch            = read_u16()
        kin_ch            = read_u16()
        is_battery        = read_u16()
        is_handshake      = read_u16()
        is_checksum       = read_u16()

        check_sum_bytes = 0
        if is_handshake:
            check_sum_bytes += 3 * 2

        callback_bytes = (
            n_ch * 2 * acq_buffer_size
            + aux_ch * 2 * acq_buffer_size
            + n_dofs * 2
            + norm_chs * 2
            + rms_chs * 2
            + sti_ch * 2
            + kin_ch * 2
            + is_battery * 2
            + check_sum_bytes
        )

        return DeviceInfo(
            is_master=is_master,
            is_slave_available=is_slave_available,
            sample_rate=sample_rate,
            acq_buffer_size=acq_buffer_size,
            n_ch=n_ch,
            aux_ch=aux_ch,
            n_dofs=n_dofs,
            norm_chs=norm_chs,
            rms_chs=rms_chs,
            sti_ch=sti_ch,
            kin_ch=kin_ch,
            is_battery=is_battery,
            is_handshake=is_handshake,
            is_checksum=is_checksum,
            callback_byte_count=callback_bytes,
        )

    def _record_block(self, block: SampleBlock) -> None:
        """Dump to files if recording is enabled."""
        # Binary
        if self._bin_emg and block.emg_raw is not None:
            self._bin_emg.write(block.emg_raw.tobytes())
        if self._bin_rms and block.rms is not None:
            self._bin_rms.write(block.rms.tobytes())
        if self._bin_kin and block.kinematics is not None:
            self._bin_kin.write(block.kinematics.tobytes())

        # CSV
        if self._csv_emg and block.emg_raw is not None:
            np.savetxt(self._csv_emg, block.emg_raw, fmt="%d", delimiter=",")
        if self._csv_rms and block.rms is not None:
            np.savetxt(self._csv_rms, block.rms.reshape(1, -1), fmt="%d", delimiter=",")
        if self._csv_kin and block.kinematics is not None:
            np.savetxt(self._csv_kin, block.kinematics.reshape(1, -1), fmt="%d", delimiter=",")

    def restart_bluetoothd(self):
        if platform.system() == "Darwin":  # macOS
            try:
                subprocess.run(
                    ["sudo", "pkill", "-f", "bluetoothd"],
                    check=True
                )
                print("macOS bluetoothd restarted")
            except subprocess.CalledProcessError as e:
                print("Failed to restart bluetoothd:", e)