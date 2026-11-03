from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass(frozen=True)
class DeviceInfo:
    is_master: int
    is_slave_available: int
    sample_rate: int
    acq_buffer_size: int
    n_ch: int
    aux_ch: int
    n_dofs: int
    norm_chs: int
    rms_chs: int
    sti_ch: int
    kin_ch: int
    is_battery: int
    is_handshake: int
    is_checksum: int
    callback_byte_count: int

@dataclass(frozen=True)
class SampleBlock:
    """
    One block of data read from the device.
    Shapes follow what the firmware sends (e.g. EMG: [acq_buffer_size, n_ch]).
    """
    emg_raw: Optional[np.ndarray]          # int16, shape (buffer, n_ch)
    aux: Optional[np.ndarray]              # int16, shape (buffer, aux_ch)
    regs: Optional[np.ndarray]             # int16, shape (n_dofs,)
    norm: Optional[np.ndarray]             # float32, shape (norm_chs,)
    rms: Optional[np.ndarray]              # int16, shape (rms_chs,)
    haptic: Optional[np.ndarray]           # int16, shape (sti_ch,)
    kinematics: Optional[np.ndarray]       # int16, shape (kin_ch,)
    battery_level: Optional[int]           # int
    batch_index: int                       # monotonically increasing block counter
