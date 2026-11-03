# Install dependencies with:
# pip install pyserial numpy scipy pygame matplotlib pillow

from __future__ import annotations

import csv
import math
import os
import random
import threading
import time
import numpy as np
from scipy.signal import butter, lfilter, iirnotch

import tkinter as tk
from tkinter import ttk, messagebox
import serial.tools.list_ports

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import pygame
from PIL import Image, ImageTk
from loopdevice import Device

# --- Hardware & Calibration Settings ---

# Channel mapping (0-based indexing)
EXT_CHANNELS = [1, 2, 3]
EXT_LABELS = ["Ext channel 1", "Ext channel 2", "Ext channel 3"]
EXT_COLORS = ["#2ecc71", "#3498db", "#e67e22"]
EXT_WEIGHTS = np.array([0.33, 0.33, 0.33])

# Flexion mapping: electrodes 6, 7, 8 map to indices 5, 6, 7
FLEX_CHANNELS = [6, 7, 8]
FLEX_LABELS = ["Flex channel 1", "Flex channel 2", "Flex channel 3"]
FLEX_COLORS = ["#9b59b6", "#e74c3c", "#1abc9c"]
FLEX_WEIGHTS = np.array([0.33, 0.33, 0.33])

# Timing screen defaults (in seconds)
DEFAULT_DURATION_L1 = 120
DEFAULT_DURATION_L2 = 120
DEFAULT_DURATION_L3 = 120

# Output calibration files
MVC_CSV_EXT = "mvc_wrist_extension.csv"
MVC_PEAKS_NPY = "mvc_peaks_extension.npy"

MVC_CSV_FLEX = "mvc_wrist_flexion.csv"
MVC_PEAKS_NPY_FLEX = "mvc_peaks_flexion.npy"

# Real-time DSP settings
SAMPLE_RATE = 1000
BANDPASS_LO = 20.0
BANDPASS_HI = 450.0
NOTCH_FREQ = 50.0
NOTCH_Q = 30.0
RMS_WINDOW = 10

# Game canvas dimensions
GAME_WIDTH = 920
GAME_HEIGHT = 620

# Image asset paths
ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
IMG_NEUTRAL = os.path.join(ASSET_DIR, "neutral.jpeg")
IMG_MVC_EXTENSION = os.path.join(ASSET_DIR, "MVC_extension_sample.jpeg")
IMG_WRIST_EXTENSION = os.path.join(ASSET_DIR, "wrist_extension_final.jpeg")
IMG_WRIST_FLEXION = os.path.join(ASSET_DIR, "wrist_flexion_final.jpeg")


def load_tk_image(path, max_size):
    """Load an image and scale it to fit without cropping."""
    image = Image.open(path)
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(image)

# Shared UI Palette
COLORS = {
    "background": (14, 22, 40),
    "grid": (22, 33, 54),
    "conveyor": (38, 52, 78),
    "conveyor_highlight": (55, 75, 110),
    "piston_body": (130, 148, 175),
    "success_green": (46, 204, 113),
    "fail_red": (231, 76, 60),
    "item_base": (200, 170, 90),
    "item_done": (46, 204, 113),
    "item_squash": (231, 76, 60),
    "activation": (46, 204, 113),
    "over_zone": (231, 76, 60),
    "text_primary": (220, 232, 255),
    "text_secondary": (100, 130, 180),
    "white": (255, 255, 255),
    "panel": (20, 30, 52),
    "accent_blue": (46, 204, 113),
    "tunnel_wall": (55, 80, 120),
    "tunnel_in": (20, 30, 52),
    "player_craft": (100, 200, 255),
    "obstacle": (231, 76, 60),
}

# Threshold required to trigger game actions
ACTIVATION_MVC = 0.20


# --- Pygame Drawing Helpers ---

def create_alpha_surface(color_rgb, alpha, width, height):
    surface = pygame.Surface((width, height), pygame.SRCALPHA)
    surface.fill((*color_rgb, alpha))
    return surface

def draw_alpha_rect(surface, color, alpha, rect):
    alpha_surf = create_alpha_surface(color, alpha, rect[2], rect[3])
    surface.blit(alpha_surf, (rect[0], rect[1]))

def draw_alpha_circle(surface, color, alpha, center_x, center_y, radius):
    circ_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    pygame.draw.circle(circ_surf, (*color, alpha), (radius, radius), radius)
    surface.blit(circ_surf, (center_x - radius, center_y - radius))

def draw_text_centered(surface, text, font, color, center_pos):
    rendered_text = font.render(text, True, color)
    text_rect = rendered_text.get_rect(center=center_pos)
    surface.blit(rendered_text, text_rect)

def lerp(a, b, t):
    return a + (b - a) * t


# --- Signal Processing ---

class EMGPreprocessor:
    """Filters one EMG block and calculates RMS values for selected channels."""
    
    def __init__(self, channel_count=3, sample_rate=SAMPLE_RATE):
        self.channel_count = channel_count
        self.sample_rate = sample_rate if sample_rate >= 1000 else SAMPLE_RATE
        self.reset_filters()

    def reset_filters(self):
        # Bandpass filter setup
        self._bandpass_b, self._bandpass_a = butter(
            4, [BANDPASS_LO, BANDPASS_HI], btype="bandpass", fs=self.sample_rate
        )
        
        # Notch filter setup
        self._notch_b, self._notch_a = iirnotch(NOTCH_FREQ, NOTCH_Q, fs=self.sample_rate)
        
        # Initial states
        bp_state = np.zeros(max(len(self._bandpass_a), len(self._bandpass_b)) - 1)
        notch_state = np.zeros(max(len(self._notch_a), len(self._notch_b)) - 1)
        
        self._bandpass_zi = [bp_state.copy() for _ in range(self.channel_count)]
        self._notch_zi = [notch_state.copy() for _ in range(self.channel_count)]

    def process(self, samples):
        rms_values = np.zeros(self.channel_count)
        
        for ch_idx in range(self.channel_count):
            signal = samples[:, ch_idx].astype(np.float32)
            
            # Apply filters
            signal, self._bandpass_zi[ch_idx] = lfilter(
                self._bandpass_b, self._bandpass_a, signal,
                zi=self._bandpass_zi[ch_idx]
            )
            signal, self._notch_zi[ch_idx] = lfilter(
                self._notch_b, self._notch_a, signal,
                zi=self._notch_zi[ch_idx]
            )
            
            # Calculate root mean square
            rms_values[ch_idx] = np.sqrt(np.mean(signal ** 2))
            
        return rms_values


# --- Visual Effects ---

class Particle:
    def __init__(self, x, y, color):
        angle = random.uniform(0, math.pi * 2)
        speed = random.uniform(70, 190)
        
        self.x = float(x)
        self.y = float(y)
        self.vel_x = math.cos(angle) * speed
        self.vel_y = math.sin(angle) * speed
        
        self.life = random.uniform(0.35, 0.65)
        self.max_life = self.life
        self.radius = random.randint(3, 7)
        self.color = color

    def update(self, dt):
        self.x += self.vel_x * dt
        self.y += self.vel_y * dt
        self.vel_y += 90 * dt  # Apply gravity
        self.life -= dt
        return self.life > 0

    def draw(self, surface):
        alpha = int(210 * self.life / self.max_life)
        draw_alpha_circle(surface, self.color, alpha, int(self.x), int(self.y), self.radius)


# --- Level 1 & 2: Piston Stamp ---

CONVEYOR_Y = GAME_HEIGHT - 115
PISTON_CENTER_X = GAME_WIDTH // 2
PISTON_REST_Y = 90
PISTON_MAX_Y = CONVEYOR_Y - 40
ITEM_STOP_X = PISTON_CENTER_X
ITEM_SPAWN_X = GAME_WIDTH + 80
ITEM_SPACING = 180
BELT_SPEED = 75

class ConveyorItem:
    WIDTH = 78
    HEIGHT = 42
    LABELS = ["⚙", "★", "♦", "▲", "●", "■"]

    def __init__(self):
        self.x = float(ITEM_SPAWN_X)
        self.y = float(CONVEYOR_Y + 10)
        self.is_done = False
        self.is_squashed = False
        self.done_timer = 0.0
        self.is_active = True
        self.label = random.choice(self.LABELS)

    def update(self, dt):
        if self.is_done or self.is_squashed:
            self.done_timer += dt
            if self.done_timer > 0.7:
                self.is_active = False
            return
            
        if self.x < -self.WIDTH * 2:
            self.is_active = False

    @property
    def rect(self):
        return (int(self.x - self.WIDTH // 2), int(self.y), self.WIDTH, self.HEIGHT)

    def draw(self, surface, font):
        if not self.is_active:
            return
            
        if self.is_done:
            color = COLORS["item_done"]
        elif self.is_squashed:
            color = COLORS["item_squash"]
        else:
            color = COLORS["item_base"]
            
        pygame.draw.rect(surface, color, self.rect, border_radius=5)
        
        # Draw activation threshold bar on the item
        bar_width = self.WIDTH - 14
        bar_x = int(self.x - bar_width // 2)
        bar_y = int(self.y) + 5
        
        pygame.draw.rect(surface, COLORS["grid"], (bar_x, bar_y, bar_width, 5), border_radius=2)
        marker_x = bar_x + int(bar_width * ACTIVATION_MVC)
        pygame.draw.line(surface, COLORS["activation"], (marker_x, bar_y), (marker_x, bar_y + 5), 2)
        
        label_surf = font.render(self.label, True, COLORS["background"])
        surface.blit(label_surf, label_surf.get_rect(center=(int(self.x), int(self.y) + self.HEIGHT // 2 + 4)))


class PistonGame:
    def __init__(self, screen, device, info, channels, weights, mvc_peaks,
                 baseline, duration, label, flipped=False):
        self.screen = screen
        self.device = device
        self.channels = channels
        self.weights = weights
        self.mvc_peaks = mvc_peaks.astype(np.float32)
        self.baseline = baseline.astype(np.float32)
        self.duration = float(duration)
        self.label = label
        self.flipped = flipped

        sample_rate = getattr(info, 'sample_rate', SAMPLE_RATE)
        self.preprocessor = EMGPreprocessor(channel_count=3, sample_rate=sample_rate)

        self._lock = threading.Lock()
        self._raw_force = 0.0
        self._display_force = 0.0

        self.items = []
        self.particles = []
        self.score = 0
        self.hits = 0
        self.attempts = 0
        self.elapsed = 0.0
        self.scroll_offset = 0.0
        self.time_left = self.duration
        self.game_over = False

        self.piston_y = float(PISTON_MAX_Y if flipped else PISTON_REST_Y)
        self.belt_moving = True
        self.current_item = None
        self.has_stamped = False
        self._above_activation = False
        self._awaiting_release = False

        self.font_main = pygame.font.SysFont("segoeui", 22, bold=True)
        self.font_small = pygame.font.SysFont("segoeui", 17)
        self.font_tiny = pygame.font.SysFont("segoeui", 13)

        self._spawn_item()

    def _on_block(self, block):
        if block.emg_raw is None:
            return
            
        chunk = block.emg_raw[:, self.channels].copy()
        rms = self.preprocessor.process(chunk)
        
        # Normalize against baseline and peaks
        norm = np.maximum(0.0, rms - self.baseline) / np.maximum(1.0, self.mvc_peaks - self.baseline)
        force = float(np.clip((norm * self.weights).sum(), 0.0, 1.0))
        
        with self._lock:
            self._raw_force = force

    def _spawn_item(self):
        self.items.append(ConveyorItem())

    def update(self, dt):
        if self.game_over:
            return
            
        self.elapsed += dt
        self.time_left = max(0.0, self.duration - self.elapsed)
        
        if self.time_left <= 0.0:
            self.game_over = True
            return

        self.scroll_offset = (self.scroll_offset + BELT_SPEED * 0.6 * dt) % 55

        with self._lock:
            force_fraction = self._raw_force

        # Smooth the visual force reading
        self._display_force += (force_fraction - self._display_force) * min(1.0, dt * 12)

        if not self.flipped:
            target_y = PISTON_REST_Y + force_fraction * (PISTON_MAX_Y - PISTON_REST_Y)
        else:
            target_y = PISTON_MAX_Y - force_fraction * (PISTON_MAX_Y - PISTON_REST_Y)
            
        self.piston_y += (target_y - self.piston_y) * min(1.0, dt * 14)

        above_activation = force_fraction > ACTIVATION_MVC
        crossed_activation = above_activation and not self._above_activation
        self._above_activation = above_activation

        if self.belt_moving:
            if crossed_activation:
                self.attempts += 1
                
            for item in self.items:
                if item.is_active:
                    item.x -= BELT_SPEED * dt
                    
                    if not item.is_done and not item.is_squashed and item.x <= ITEM_STOP_X:
                        item.x = float(ITEM_STOP_X)
                        self.belt_moving = False
                        self.current_item = item
                        self.has_stamped = False
                        break
        else:
            item_valid = self.current_item and not self.current_item.is_done and not self.current_item.is_squashed
            
            if item_valid:
                if force_fraction > ACTIVATION_MVC and not self.has_stamped:
                    self.has_stamped = True
                    self.current_item.is_done = True
                    self.score += 100
                    self.hits += 1
                    self.attempts += 1
                    self._create_burst(self.current_item.x, self.current_item.y + 10, COLORS["success_green"])
                    self._awaiting_release = True
                    
            elif self.current_item and self.current_item.is_done and self._awaiting_release:
                if not above_activation:
                    self.belt_moving = True
                    self.current_item = None
                    self._awaiting_release = False
                    
            elif self.current_item and not self.has_stamped:
                # Proceed if the user relaxes entirely
                if force_fraction < 0.05:
                    self.attempts += 1
                    self.belt_moving = True
                    self.current_item = None

        # Clean up off-screen/dead items
        for item in self.items:
            item.update(dt)
            
        active_items = [i for i in self.items if i.is_active]
        if not active_items or active_items[-1].x < GAME_WIDTH - ITEM_SPACING:
            self._spawn_item()
            
        self.items = [i for i in self.items if i.is_active]
        self.particles = [p for p in self.particles if p.update(dt)]

    def _create_burst(self, x, y, color):
        for _ in range(18):
            self.particles.append(Particle(x, y, color))

    def draw(self):
        surface = self.screen
        surface.fill(COLORS["background"])
        
        # Grid background
        for x in range(-55 + int(self.scroll_offset), GAME_WIDTH, 55):
            pygame.draw.line(surface, COLORS["grid"], (x, 0), (x, GAME_HEIGHT))
        for y in range(0, GAME_HEIGHT, 55):
            pygame.draw.line(surface, COLORS["grid"], (0, y), (GAME_WIDTH, y))
            
        self._draw_conveyor(surface)
        self._draw_piston(surface)
        
        for item in self.items:
            item.draw(surface, self.font_main)
            
        for p in self.particles:
            p.draw(surface)
            
        self._draw_hud(surface)
        
        if self.game_over:
            self._draw_results_overlay(surface)

    def _draw_conveyor(self, surface):
        pygame.draw.rect(surface, COLORS["conveyor"], (0, CONVEYOR_Y, GAME_WIDTH, GAME_HEIGHT - CONVEYOR_Y))
        for rx in range(30, GAME_WIDTH, 55):
            pygame.draw.circle(surface, COLORS["conveyor_highlight"], (rx, CONVEYOR_Y + 5), 6)
        pygame.draw.line(surface, COLORS["text_secondary"], (0, CONVEYOR_Y), (GAME_WIDTH, CONVEYOR_Y), 2)

    def _draw_piston(self, surface):
        piston_y = int(self.piston_y)
        cx = PISTON_CENTER_X

        if not self.flipped:
            activation_y = int(PISTON_REST_Y + ACTIVATION_MVC * (PISTON_MAX_Y - PISTON_REST_Y))
            
            # Draw piston arm and block
            pygame.draw.rect(surface, COLORS["piston_body"], (cx - 7, PISTON_REST_Y - 30, 14, piston_y - PISTON_REST_Y + 30))
            pygame.draw.rect(surface, COLORS["piston_body"], (cx - 32, PISTON_REST_Y - 58, 64, 32), border_radius=4)
            draw_text_centered(surface, "▼", self.font_main, COLORS["text_primary"], (cx, PISTON_REST_Y - 42))
            
            # Tip
            pygame.draw.rect(surface, COLORS["success_green"], (cx - 32, piston_y, 64, 22), border_radius=5)
            pygame.draw.rect(surface, COLORS["white"], (cx - 32, piston_y, 64, 22), 1, border_radius=5)
            
            # Guide line
            nx = cx + 58
            pygame.draw.line(surface, COLORS["text_secondary"], (nx, PISTON_REST_Y), (nx, PISTON_MAX_Y), 1)
            pygame.draw.circle(surface, COLORS["text_primary"], (nx, piston_y + 11), 5)
            pygame.draw.line(surface, COLORS["activation"], (nx - 8, activation_y), (nx + 8, activation_y), 2)
        else:
            activation_y = int(PISTON_MAX_Y - ACTIVATION_MVC * (PISTON_MAX_Y - PISTON_REST_Y))
            
            # Draw piston arm and block
            pygame.draw.rect(surface, COLORS["piston_body"], (cx - 7, piston_y, 14, PISTON_MAX_Y - piston_y + 32))
            pygame.draw.rect(surface, COLORS["piston_body"], (cx - 32, PISTON_MAX_Y + 10, 64, 32), border_radius=4)
            draw_text_centered(surface, "▲", self.font_main, COLORS["text_primary"], (cx, PISTON_MAX_Y + 26))
            
            # Tip
            pygame.draw.rect(surface, COLORS["success_green"], (cx - 32, piston_y - 22, 64, 22), border_radius=5)
            pygame.draw.rect(surface, COLORS["white"], (cx - 32, piston_y - 22, 64, 22), 1, border_radius=5)
            
            # Guide line
            nx = cx + 58
            pygame.draw.line(surface, COLORS["text_secondary"], (nx, PISTON_REST_Y), (nx, PISTON_MAX_Y), 1)
            pygame.draw.circle(surface, COLORS["text_primary"], (nx, piston_y - 11), 5)
            pygame.draw.line(surface, COLORS["activation"], (nx - 8, activation_y), (nx + 8, activation_y), 2)

    def _draw_hud(self, surface):
        box_x, box_y, box_w, box_h = 28, GAME_HEIGHT - 90, 220, 28
        
        # Power bar
        pygame.draw.rect(surface, COLORS["panel"], (box_x, box_y, box_w, box_h), border_radius=6)
        fill_width = int((box_w - 8) * self._display_force)
        if fill_width > 0:
            pygame.draw.rect(surface, COLORS["accent_blue"], (box_x + 4, box_y + 4, fill_width, box_h - 8), border_radius=4)
        pygame.draw.rect(surface, COLORS["text_secondary"], (box_x, box_y, box_w, box_h), 1, border_radius=6)
        
        motion = "Flexion" if self.flipped else "Extension"
        power_text = self.font_small.render(f"{motion}  {int(self._display_force * 100):3d}% MVC", True, COLORS["text_primary"])
        surface.blit(power_text, (box_x, box_y - 22))
        
        # Score & Accuracy
        score_text = self.font_main.render(f"Score  {self.score}", True, COLORS["text_primary"])
        surface.blit(score_text, (28, 18))
        
        accuracy = f"{int(self.hits / self.attempts * 100)}%" if self.attempts else "—"
        stats_text = self.font_small.render(f"Hits  {self.hits}   Accuracy  {accuracy}", True, COLORS["text_secondary"])
        surface.blit(stats_text, (28, 50))
        
        # Timer
        minutes, seconds = divmod(int(self.time_left), 60)
        timer_col = COLORS["fail_red"] if self.time_left <= 30 else COLORS["text_primary"]
        timer_font = pygame.font.SysFont("segoeui", 28, bold=True)
        draw_text_centered(surface, f"{minutes:02d}:{seconds:02d}", timer_font, timer_col, (GAME_WIDTH // 2, 22))
        
        draw_text_centered(surface, "ESC – end level early", self.font_tiny, COLORS["text_secondary"], (GAME_WIDTH // 2, 46))
        draw_text_centered(surface, self.label, self.font_small, COLORS["accent_blue"], (GAME_WIDTH // 2, GAME_HEIGHT - 22))

    def _draw_results_overlay(self, surface):
        overlay = pygame.Surface((GAME_WIDTH, GAME_HEIGHT), pygame.SRCALPHA)
        overlay.fill((10, 15, 30, 210))
        surface.blit(overlay, (0, 0))
        
        font_xl = pygame.font.SysFont("segoeui", 44, bold=True)
        font_large = pygame.font.SysFont("segoeui", 28, bold=True)
        font_medium = pygame.font.SysFont("segoeui", 21)
        font_small = pygame.font.SysFont("segoeui", 16)
        
        center_x, center_y = GAME_WIDTH // 2, GAME_HEIGHT // 2
        card_w, card_h = 560, 360
        
        card = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
        card.fill((20, 32, 58, 230))
        surface.blit(card, (center_x - card_w // 2, center_y - card_h // 2))
        pygame.draw.rect(surface, COLORS["accent_blue"], (center_x - card_w // 2, center_y - card_h // 2, card_w, card_h), 2, border_radius=10)
        
        y_pos = center_y - card_h // 2 + 26
        draw_text_centered(surface, f"{self.label}  Complete!", font_xl, COLORS["accent_blue"], (center_x, y_pos))
        y_pos += 54
        
        pygame.draw.line(surface, COLORS["panel"], (center_x - card_w // 2 + 30, y_pos), (center_x + card_w // 2 - 30, y_pos), 1)
        y_pos += 16
        
        minutes, seconds = divmod(int(self.elapsed), 60)
        accuracy = f"{int(self.hits / self.attempts * 100)}%" if self.attempts else "—"
        
        stats = [
            ("Time played", f"{minutes:02d}:{seconds:02d}"),
            ("Items presented", str(self.attempts)),
            ("Successful stamps", str(self.hits)),
            ("Accuracy", accuracy),
            ("Score", str(self.score))
        ]
        
        for label, val in stats:
            surface.blit(font_medium.render(label, True, COLORS["text_secondary"]), (center_x - card_w // 2 + 36, y_pos))
            val_surface = font_large.render(val, True, COLORS["text_primary"])
            surface.blit(val_surface, (center_x + card_w // 2 - 36 - val_surface.get_width(), y_pos - 2))
            y_pos += 36
            
        y_pos += 6
        pygame.draw.line(surface, COLORS["panel"], (center_x - card_w // 2 + 30, y_pos), (center_x + card_w // 2 - 30, y_pos), 1)
        y_pos += 14
        
        draw_text_centered(surface, "Press  ENTER  to continue to next level", font_small, COLORS["text_secondary"], (center_x, y_pos))


# --- Level 3: Tunnel Mini-Game ---

TUNNEL_CENTER_Y = GAME_HEIGHT // 2
TUNNEL_HEIGHT = 105
AVATAR_X = 200
SEGMENT_WIDTH = 120
N_SEGMENTS = GAME_WIDTH // SEGMENT_WIDTH + 3
TUNNEL_SPEED = 80

class TunnelSegment:
    def __init__(self, x, center_y_offset):
        self.x = float(x)
        self.center_y_offset = center_y_offset

    def update(self, dt):
        self.x -= TUNNEL_SPEED * dt

    @property
    def cx(self):
        return int(self.x + SEGMENT_WIDTH // 2)

    @property
    def cy(self):
        return int(TUNNEL_CENTER_Y + self.center_y_offset)


class Obstacle:
    WIDTH, HEIGHT = 28, 28

    def __init__(self, segment_idx, segments):
        segment = segments[segment_idx]
        self.x = float(segment.x + SEGMENT_WIDTH // 2)
        cy = int(TUNNEL_CENTER_Y + segment.center_y_offset)
        side = random.choice([-1, 1])
        self.y = float(cy + side * random.randint(TUNNEL_HEIGHT // 4, TUNNEL_HEIGHT * 3 // 4 - self.HEIGHT // 2))
        self.is_active = True

    def update(self, dt):
        self.x -= TUNNEL_SPEED * dt
        if self.x < -self.WIDTH:
            self.is_active = False

    @property
    def rect(self):
        return pygame.Rect(self.x - self.WIDTH // 2, self.y - self.HEIGHT // 2, self.WIDTH, self.HEIGHT)


class TunnelGame:
    LIFT_ACCEL = 400
    DAMPEN = 0.88

    def __init__(self, screen, device, info, ext_channels, ext_weights, ext_peaks,
                 ext_baseline, flex_channels, flex_weights, flex_peaks, flex_baseline, duration):
        self.screen = screen
        self.device = device
        self.duration = float(duration)

        sample_rate = getattr(info, 'sample_rate', SAMPLE_RATE)
        self.ext_pre = EMGPreprocessor(3, sample_rate)
        self.flex_pre = EMGPreprocessor(3, sample_rate)
        
        self.ext_ch = ext_channels
        self.ext_w = ext_weights
        self.ext_pk = ext_peaks.astype(np.float32)
        self.ext_base = ext_baseline.astype(np.float32)
        
        self.flex_ch = flex_channels
        self.flex_w = flex_weights
        self.flex_pk = flex_peaks.astype(np.float32)
        self.flex_base = flex_baseline.astype(np.float32)

        self._lock = threading.Lock()
        self._ext_force = 0.0
        self._flex_force = 0.0

        self.avatar_y = float(TUNNEL_CENTER_Y)
        self.vel_y = 0.0
        self.pulse = 0.0
        self.elapsed = 0.0
        self.time_left = self.duration
        self.game_over = False
        self.scroll_offset = 0.0

        self.score = 0
        self.collected = 0
        self.total_coins = 0
        self.particles = []

        self.segments = []
        self.obstacles = []
        self._obs_timer = 0.0
        
        for i in range(N_SEGMENTS + 2):
            self._add_segment(i * SEGMENT_WIDTH)

        self.font_main = pygame.font.SysFont("segoeui", 22, bold=True)
        self.font_small = pygame.font.SysFont("segoeui", 17)
        self.font_tiny = pygame.font.SysFont("segoeui", 13)

    def _on_block(self, block):
        if block.emg_raw is None:
            return
            
        ext_rms = self.ext_pre.process(block.emg_raw[:, self.ext_ch].copy())
        flex_rms = self.flex_pre.process(block.emg_raw[:, self.flex_ch].copy())
        
        ext_norm = np.maximum(0.0, ext_rms - self.ext_base) / np.maximum(1.0, self.ext_pk - self.ext_base)
        flex_norm = np.maximum(0.0, flex_rms - self.flex_base) / np.maximum(1.0, self.flex_pk - self.flex_base)
        
        ext_force = float(np.clip((ext_norm * self.ext_w).sum(), 0.0, 1.0))
        flex_force = float(np.clip((flex_norm * self.flex_w).sum(), 0.0, 1.0))
        
        with self._lock:
            self._ext_force = ext_force
            self._flex_force = flex_force

    def _add_segment(self, x):
        self.segments.append(TunnelSegment(float(x), 0.0))

    def _avatar_tunnel_cy(self):
        for i, segment in enumerate(self.segments):
            if segment.x <= AVATAR_X < segment.x + SEGMENT_WIDTH:
                t = (AVATAR_X - segment.x) / SEGMENT_WIDTH
                next_offset = self.segments[i + 1].center_y_offset if i + 1 < len(self.segments) else segment.center_y_offset
                return int(TUNNEL_CENTER_Y + lerp(segment.center_y_offset, next_offset, t))
        return TUNNEL_CENTER_Y

    def update(self, dt):
        if self.game_over:
            return
            
        self.elapsed += dt
        self.time_left = max(0.0, self.duration - self.elapsed)
        
        if self.time_left <= 0.0:
            self.game_over = True
            return

        self.pulse += dt * 3.5
        self.scroll_offset = (self.scroll_offset + TUNNEL_SPEED * 0.3 * dt) % 55

        with self._lock:
            extension_force = self._ext_force
            flexion_force = self._flex_force

        # Movement physics driven by muscle inputs
        extension_drive = max(0.0, (extension_force - ACTIVATION_MVC) / (1.0 - ACTIVATION_MVC))
        flexion_drive = max(0.0, (flexion_force - ACTIVATION_MVC) / (1.0 - ACTIVATION_MVC))
        
        vertical_input = extension_drive - flexion_drive
        self.vel_y -= vertical_input * self.LIFT_ACCEL * dt
        self.vel_y *= (1.0 - (1.0 - self.DAMPEN) * dt * 60)
        self.avatar_y += self.vel_y * dt

        # Tunnel segment pooling
        for seg in self.segments:
            seg.update(dt)
            
        while self.segments and self.segments[0].x + SEGMENT_WIDTH < 0:
            self.segments.pop(0)
            
        while len(self.segments) < N_SEGMENTS + 2:
            self._add_segment(self.segments[-1].x + SEGMENT_WIDTH)

        # Wall collisions
        cy = self._avatar_tunnel_cy()
        wall_top = cy - TUNNEL_HEIGHT
        wall_bot = cy + TUNNEL_HEIGHT
        hit_wall = False
        
        if self.avatar_y < wall_top:
            self.avatar_y = float(wall_top)
            self.vel_y = abs(self.vel_y) * 0.15
            hit_wall = True
            
        if self.avatar_y > wall_bot:
            self.avatar_y = float(wall_bot)
            self.vel_y = -abs(self.vel_y) * 0.15
            hit_wall = True
            
        if hit_wall:
            for _ in range(6):
                self.particles.append(Particle(AVATAR_X, int(self.avatar_y), COLORS["fail_red"]))

        # Obstacle logic
        self._obs_timer += dt
        if self._obs_timer >= 3.0 and len(self.segments) > 5:
            self._obs_timer = 0.0
            mid_idx = len(self.segments) // 2
            self.obstacles.append(Obstacle(mid_idx, self.segments))
            self.total_coins += 1

        # Match the visible pulsing collection ring surrounding the spacecraft.
        collection_radius = 26 + 3 + int(math.sin(self.pulse) * 2)
        
        for obs in self.obstacles:
            obs.update(dt)
            distance_to_coin = math.hypot(obs.x - AVATAR_X, obs.y - self.avatar_y)
            if obs.is_active and distance_to_coin <= collection_radius + obs.WIDTH / 2:
                obs.is_active = False
                self.collected += 1
                self.score += 50
                for _ in range(12):
                    self.particles.append(Particle(int(obs.x), int(obs.y), COLORS["item_base"]))

        self.obstacles = [o for o in self.obstacles if o.is_active]
        self.particles = [p for p in self.particles if p.update(dt)]

    def draw(self):
        surface = self.screen
        surface.fill(COLORS["background"])
        
        for x in range(-55 + int(self.scroll_offset), GAME_WIDTH, 55):
            pygame.draw.line(surface, COLORS["grid"], (x, 0), (x, GAME_HEIGHT))
        for y in range(0, GAME_HEIGHT, 55):
            pygame.draw.line(surface, COLORS["grid"], (0, y), (GAME_WIDTH, y))
            
        self._draw_tunnel(surface)
        self._draw_obstacles(surface)
        self._draw_avatar(surface)
        
        for p in self.particles:
            p.draw(surface)
            
        self._draw_hud(surface)
        
        if self.game_over:
            self._draw_results_overlay(surface)

    def _draw_tunnel(self, surface):
        points_top = []
        points_bottom = []
        
        for seg in self.segments:
            points_top.append((seg.cx, seg.cy - TUNNEL_HEIGHT))
            points_bottom.append((seg.cx, seg.cy + TUNNEL_HEIGHT))
            
        if len(points_top) < 2:
            return
            
        # Draw solid walls and interior
        pygame.draw.polygon(surface, COLORS["tunnel_wall"], [(points_top[0][0], 0)] + points_top + [(points_top[-1][0], 0)])
        pygame.draw.polygon(surface, COLORS["tunnel_wall"], [(points_bottom[0][0], GAME_HEIGHT)] + points_bottom + [(points_bottom[-1][0], GAME_HEIGHT)])
        pygame.draw.polygon(surface, COLORS["tunnel_in"], list(points_top) + list(reversed(points_bottom)))
        
        # Edge lines
        pygame.draw.lines(surface, COLORS["accent_blue"], False, points_top, 2)
        pygame.draw.lines(surface, COLORS["fail_red"], False, points_bottom, 2)

    def _draw_obstacles(self, surface):
        for obs in self.obstacles:
            if not obs.is_active:
                continue
            
            center = (int(obs.x), int(obs.y))
            pygame.draw.circle(surface, COLORS["item_base"], center, obs.WIDTH // 2)
            pygame.draw.circle(surface, COLORS["white"], center, obs.WIDTH // 2, 2)
            pygame.draw.circle(surface, (245, 215, 120), center, obs.WIDTH // 4)
            
            coin = self.font_tiny.render("●", True, COLORS["white"])
            surface.blit(coin, coin.get_rect(center=center))

    def _draw_avatar(self, surface):
        ax = AVATAR_X
        ay = int(self.avatar_y)
        pulsing_radius = 3 + int(math.sin(self.pulse) * 2)
        
        draw_alpha_circle(surface, COLORS["player_craft"], 50, ax, ay, 26 + pulsing_radius)
        
        points = [(ax - 24, ay), (ax - 10, ay - 12), (ax + 20, ay - 8),
                  (ax + 28, ay), (ax + 20, ay + 8), (ax - 10, ay + 12)]
                  
        pygame.draw.polygon(surface, COLORS["player_craft"], points)
        pygame.draw.polygon(surface, COLORS["white"], points, 1)
        
        pygame.draw.circle(surface, COLORS["background"], (ax + 8, ay), 7)
        pygame.draw.circle(surface, (60, 160, 220), (ax + 8, ay), 5)

    def _draw_hud(self, surface):
        box_x, box_y, box_w, box_h = 28, GAME_HEIGHT - 60, 200, 22
        
        with self._lock:
            extension_force = self._ext_force
            flexion_force = self._flex_force
            
        # Extension UI
        pygame.draw.rect(surface, COLORS["panel"], (box_x, box_y, box_w, box_h), border_radius=5)
        extension_width = int((box_w - 6) * extension_force)
        if extension_width:
            pygame.draw.rect(surface, COLORS["accent_blue"], (box_x + 3, box_y + 3, extension_width, box_h - 6), border_radius=3)
        pygame.draw.rect(surface, COLORS["text_secondary"], (box_x, box_y, box_w, box_h), 1, border_radius=5)
        surface.blit(self.font_tiny.render(f"Extension {int(extension_force * 100)}%", True, COLORS["accent_blue"]), (box_x, box_y - 16))

        # Flexion UI
        box2_x = box_x + box_w + 14
        pygame.draw.rect(surface, COLORS["panel"], (box2_x, box_y, box_w, box_h), border_radius=5)
        flexion_width = int((box_w - 6) * flexion_force)
        if flexion_width:
            pygame.draw.rect(surface, COLORS["fail_red"], (box2_x + 3, box_y + 3, flexion_width, box_h - 6), border_radius=3)
        pygame.draw.rect(surface, COLORS["text_secondary"], (box2_x, box_y, box_w, box_h), 1, border_radius=5)
        surface.blit(self.font_tiny.render(f"Flexion {int(flexion_force * 100)}%", True, COLORS["fail_red"]), (box2_x, box_y - 16))

        # Stats
        surface.blit(self.font_main.render(f"Score  {self.score}", True, COLORS["text_primary"]), (28, 18))
        accuracy = f"{int(self.collected / self.total_coins * 100)}%" if self.total_coins else "—"
        surface.blit(self.font_small.render(f"Collected  {self.collected}  /  {self.total_coins}   Accuracy  {accuracy}", True, COLORS["text_secondary"]), (28, 50))
        
        minutes, seconds = divmod(int(self.time_left), 60)
        timer_col = COLORS["fail_red"] if self.time_left <= 30 else COLORS["text_primary"]
        draw_text_centered(surface, f"{minutes:02d}:{seconds:02d}", pygame.font.SysFont("segoeui", 28, bold=True), timer_col, (GAME_WIDTH // 2, 22))
        
        draw_text_centered(surface, "ESC – end level early", self.font_tiny, COLORS["text_secondary"], (GAME_WIDTH // 2, 46))
        draw_text_centered(surface, "LEVEL 3 – COIN COLLECTION TUNNEL  |  Ext ↑  /  Flex ↓", self.font_small, COLORS["fail_red"], (GAME_WIDTH // 2, GAME_HEIGHT - 22))

    def _draw_results_overlay(self, surface):
        overlay = pygame.Surface((GAME_WIDTH, GAME_HEIGHT), pygame.SRCALPHA)
        overlay.fill((10, 15, 30, 210))
        surface.blit(overlay, (0, 0))
        
        font_xl = pygame.font.SysFont("segoeui", 44, bold=True)
        font_large = pygame.font.SysFont("segoeui", 28, bold=True)
        font_medium = pygame.font.SysFont("segoeui", 21)
        font_small = pygame.font.SysFont("segoeui", 16)
        
        center_x, center_y = GAME_WIDTH // 2, GAME_HEIGHT // 2
        card_w, card_h = 560, 340
        card = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
        card.fill((20, 32, 58, 230))
        
        surface.blit(card, (center_x - card_w // 2, center_y - card_h // 2))
        pygame.draw.rect(surface, COLORS["fail_red"], (center_x - card_w // 2, center_y - card_h // 2, card_w, card_h), 2, border_radius=10)
        
        y_pos = center_y - card_h // 2 + 26
        draw_text_centered(surface, "Level 3 Complete!", font_xl, COLORS["fail_red"], (center_x, y_pos))
        y_pos += 54
        
        pygame.draw.line(surface, COLORS["panel"], (center_x - card_w // 2 + 30, y_pos), (center_x + card_w // 2 - 30, y_pos), 1)
        y_pos += 16
        
        minutes, seconds = divmod(int(self.elapsed), 60)
        accuracy = f"{int(self.collected / self.total_coins * 100)}%" if self.total_coins else "—"
        
        stats = [
            ("Time played", f"{minutes:02d}:{seconds:02d}"),
            ("Coins presented", str(self.total_coins)),
            ("Coins collected", str(self.collected)),
            ("Accuracy", accuracy),
            ("Score", str(self.score))
        ]
        
        for label, val in stats:
            surface.blit(font_medium.render(label, True, COLORS["text_secondary"]), (center_x - card_w // 2 + 36, y_pos))
            val_surface = font_large.render(val, True, COLORS["text_primary"])
            surface.blit(val_surface, (center_x + card_w // 2 - 36 - val_surface.get_width(), y_pos - 2))
            y_pos += 34
            
        y_pos += 6
        pygame.draw.line(surface, COLORS["panel"], (center_x - card_w // 2 + 30, y_pos), (center_x + card_w // 2 - 30, y_pos), 1)
        y_pos += 14
        
        draw_text_centered(surface, "Press  ENTER  to see full session results", font_small, COLORS["text_secondary"], (center_x, y_pos))


# --- Tkinter UI Screens ---

class ConnectScreen(tk.Frame):
    def __init__(self, master, on_connected):
        super().__init__(master, bg="#1a2540")
        self.on_connected = on_connected
        self._build_ui()

    def _build_ui(self):
        tk.Label(self, text="Wrist Rehabilitation Game Suite", font=("Segoe UI", 22, "bold"), bg="#1a2540", fg="#ecf0f1").pack(pady=(40, 4))
        tk.Label(self, text="University of Sussex  |  MSc Project (RAS)  |  Candidate 307631", font=("Segoe UI", 11), bg="#1a2540", fg="#7f8c8d").pack(pady=(0, 30))
        
        frame = tk.Frame(self, bg="#253154")
        frame.pack(padx=60, pady=10, fill=tk.X)
        
        tk.Label(frame, text="Serial Port:", font=("Segoe UI", 12), bg="#253154", fg="#bdc3c7").grid(row=0, column=0, padx=16, pady=16, sticky=tk.W)
        self.port_combo = ttk.Combobox(frame, values=self._get_ports(), width=52)
        self.port_combo.grid(row=0, column=1, padx=8, pady=16)
        
        if self.port_combo['values']:
            self.port_combo.current(0)
            
        ttk.Button(frame, text="Refresh", command=self._refresh_ports).grid(row=0, column=2, padx=8)
        
        self.status_label = tk.Label(self, text="", font=("Segoe UI", 11), bg="#1a2540", fg="#e74c3c")
        self.status_label.pack(pady=8)
        
        tk.Button(self, text="Connect to Device", font=("Segoe UI", 13, "bold"), bg="#2ecc71", fg="white", relief=tk.FLAT, padx=24, pady=10, cursor="hand2", command=self._connect).pack(pady=16)
        
    def _get_ports(self):
        ports = [f"{x.device} – {x.description}" for x in serial.tools.list_ports.comports()]
        return ports

    def _refresh_ports(self):
        self.port_combo['values'] = self._get_ports()

    def _connect(self):
        port_string = self.port_combo.get()
        if not port_string:
            self.status_label.config(text="Select a port.")
            return
            
        self.status_label.config(text="Connecting…", fg="#f39c12")
        self.update()
        threading.Thread(target=self._connection_worker, args=(port_string,), daemon=True).start()

    def _connection_worker(self, port_string):
        try:
            device = Device(port_string.split(" – ")[0].split(" - ")[0])
            info = device.connect()
            self.after(0, lambda: self.on_connected(device, info))
        except Exception as error:
            self.after(0, lambda: self.status_label.config(text=f"Failed: {error}", fg="#e74c3c"))


class MVCScreen(tk.Frame):
    def __init__(self, master, device, info, channels, labels, colors, title, motion, instr_body, csv_file, npy_file, on_complete):
        super().__init__(master, bg="#1a2540")
        self.device = device
        self.channels = channels
        self.labels = labels
        self.colors = colors
        self.title_str = title
        self.motion = motion
        self.csv_file = csv_file
        self.npy_file = npy_file
        self.on_complete = on_complete
        self.instr_body = instr_body
        self.action_image = IMG_MVC_EXTENSION if motion == "Wrist Extension" else IMG_WRIST_FLEXION

        self._lock = threading.Lock()
        self._raw_data = np.zeros(3)
        self._is_recording = False
        self._raw_frames = []
        self._all_peaks = []
        self._trial_count = 0
        self._current_peaks = np.zeros(3)
        
        self._build_instructions()

    def _build_instructions(self):
        self._build_live_chart()
        dialog = tk.Toplevel(self.master)
        dialog.title(f"MVC – {self.motion}")
        dialog.resizable(False, False)
        dialog.grab_set()
        dialog.configure(bg="#1a2540")
        
        tk.Label(dialog, text=f"MVC Calibration: {self.motion}", font=("Segoe UI", 16, "bold"), bg="#1a2540", fg="#ecf0f1").pack(padx=30, pady=(24, 8))
        
        img_frame = tk.Frame(dialog, bg="#1a2540")
        img_frame.pack(padx=30, pady=(4, 8))
        self._mvc_images = []
        
        for path, caption in [(IMG_NEUTRAL, "Starting position"), (self.action_image, "Contraction position")]:
            col_frame = tk.Frame(img_frame, bg="#253154", relief=tk.RIDGE, bd=2)
            col_frame.pack(side=tk.LEFT, padx=6)
            photo = load_tk_image(path, (210, 150))
            self._mvc_images.append(photo)
            
            tk.Label(col_frame, image=photo, bg="#253154").pack(padx=5, pady=(5, 2))
            tk.Label(col_frame, text=caption, font=("Segoe UI", 9, "bold"), bg="#253154", fg="#bdc3c7").pack(pady=(0, 5))
            
        tk.Label(dialog, text=self.instr_body, font=("Segoe UI", 11), bg="#1a2540", fg="#bdc3c7", justify=tk.LEFT).pack(padx=30, pady=8)
        tk.Button(dialog, text="OK – Start Calibration", font=("Segoe UI", 12, "bold"), bg="#2ecc71", fg="white", relief=tk.FLAT, padx=20, pady=8, cursor="hand2", command=lambda: self._start_calibration(dialog)).pack(pady=(8, 24))

    def _build_live_chart(self):
        tk.Label(self, text=self.title_str, font=("Segoe UI", 18, "bold"), bg="#1a2540", fg="#ecf0f1").pack(pady=(20, 4))
        tk.Label(self, text="  ·  ".join(self.labels), font=("Segoe UI", 10), bg="#1a2540", fg="#7f8c8d").pack()
        
        self.fig, self.ax = plt.subplots(figsize=(7, 3.2), facecolor="#1a2540")
        self.ax.set_facecolor("#253154")
        self.bars = self.ax.bar(self.labels, [0, 0, 0], color=self.colors, edgecolor="#1a2540", width=0.5)
        self.ax.set_ylim(0, 3000)
        self.ax.set_ylabel("Raw Amplitude (abs mean)", color="#bdc3c7")
        self.ax.tick_params(colors="#bdc3c7")
        
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#3d5170")
            
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(padx=20, pady=8, fill=tk.BOTH)
        
        peaks_frame = tk.Frame(self, bg="#1a2540")
        peaks_frame.pack()
        
        self._peak_labels = []
        for label, color in zip(self.labels, self.colors):
            row_frame = tk.Frame(peaks_frame, bg="#1a2540")
            row_frame.pack(side=tk.LEFT, padx=20)
            tk.Label(row_frame, text=label, font=("Segoe UI", 9), bg="#1a2540", fg=color).pack()
            
            var = tk.StringVar(value="Peak: 0")
            tk.Label(row_frame, textvariable=var, font=("Segoe UI", 11, "bold"), bg="#1a2540", fg=color).pack()
            self._peak_labels.append(var)
            
        self.status_var = tk.StringVar(value="Waiting…")
        tk.Label(self, textvariable=self.status_var, font=("Segoe UI", 11, "italic"), bg="#1a2540", fg="#f39c12").pack(pady=6)
        
        self.trial_var = tk.StringVar(value="Trial: –")
        tk.Label(self, textvariable=self.trial_var, font=("Segoe UI", 11), bg="#1a2540", fg="#bdc3c7").pack()
        
        btn_frame = tk.Frame(self, bg="#1a2540")
        btn_frame.pack(pady=12)
        
        self.record_btn = tk.Button(btn_frame, text="▶  Start Trial Recording", font=("Segoe UI", 12, "bold"), bg="#3498db", fg="white", relief=tk.FLAT, padx=18, pady=8, cursor="hand2", command=self._toggle_recording, state=tk.DISABLED)
        self.record_btn.pack(side=tk.LEFT, padx=8)
        
        self.save_btn = tk.Button(btn_frame, text="✔  Save MVC & Continue", font=("Segoe UI", 12, "bold"), bg="#2ecc71", fg="white", relief=tk.FLAT, padx=18, pady=8, cursor="hand2", command=self._save_data, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=8)

    def _start_calibration(self, dialog):
        dialog.destroy()
        self.device.start_acquisition(on_block=self._on_block)
        self.status_var.set("Ready – press Start Trial to record your MVC.")
        self.record_btn.config(state=tk.NORMAL)
        self._update_chart()

    def _on_block(self, block):
        if block.emg_raw is None:
            return
            
        chunk = block.emg_raw[:, self.channels].copy().astype(np.float32)
        raw_mean = np.mean(np.abs(chunk), axis=0)
        
        with self._lock:
            self._raw_data = raw_mean
            if self._is_recording:
                self._raw_frames.append(raw_mean.copy())

    def _update_chart(self):
        with self._lock:
            values = self._raw_data.copy()
            
        for bar, value in zip(self.bars, values):
            bar.set_height(value)
            bar.set_facecolor("#e74c3c" if value > 2200 else "#f39c12" if value > 1200 else "#2ecc71")
            
        self._current_peaks = np.maximum(self._current_peaks, values)
        
        for i, label in enumerate(self._peak_labels):
            label.set(f"Peak: {int(self._current_peaks[i])}")
            
        self.ax.set_ylim(0, max(3000, float(self._current_peaks.max()) * 1.2))
        self.canvas.draw_idle()
        
        self.after(50, self._update_chart)

    def _toggle_recording(self):
        if not self._is_recording:
            self._trial_count += 1
            self._raw_frames = []
            self._is_recording = True
            self._current_peaks = np.zeros(3)
            
            self.record_btn.config(text="■  Stop Trial", bg="#e74c3c")
            self.trial_var.set(f"Trial {self._trial_count} – Recording…")
            self.status_var.set(f"Perform {self.motion} as hard as possible!")
        else:
            self._is_recording = False
            self.record_btn.config(text="▶  Start Another Trial", bg="#3498db")
            self.status_var.set(f"Trial {self._trial_count} done. Record more or click Save.")
            
            if self._raw_frames:
                trial_array = np.vstack(self._raw_frames)
                self._all_peaks.append(trial_array.max(axis=0))
                
            if self._trial_count >= 1:
                self.save_btn.config(state=tk.NORMAL)

    def _save_data(self):
        if not self._all_peaks:
            messagebox.showwarning("No Data", "Record at least one trial first.")
            return
            
        final_peaks = np.maximum(1.0, np.vstack(self._all_peaks).max(axis=0))
        col_names = [label.replace(" ", "_") for label in self.labels]
        
        try:
            with open(self.csv_file, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(col_names)
                writer.writerows(np.vstack(self._all_peaks).tolist())
                
            np.save(self.npy_file, final_peaks)
            self.status_var.set(f"Saved. MVC peaks: {final_peaks.round(0)}")
        except Exception as error:
            messagebox.showerror("Save Error", str(error))
            return
            
        self.device.stop_acquisition()
        self.after(600, lambda: self.on_complete(final_peaks))


class TimingScreen(tk.Frame):
    def __init__(self, master, mvc_peaks_ext, mvc_peaks_flex, on_start):
        super().__init__(master, bg="#1a2540")
        self.mvc_peaks_ext = mvc_peaks_ext
        self.mvc_peaks_flex = mvc_peaks_flex
        self.on_start = on_start
        self._build_ui()

    def _build_ui(self):
        tk.Label(self, text="Session Timing Setup", font=("Segoe UI", 22, "bold"), bg="#1a2540", fg="#ecf0f1").pack(pady=(36, 6))
        tk.Label(self, text="Set the duration (in seconds) for each game level.", font=("Segoe UI", 12), bg="#1a2540", fg="#bdc3c7").pack(pady=(0, 6))
        
        ext_peak_str = "  ".join([f"Ch{ch+1}: {self.mvc_peaks_ext[i]:.0f}" for i, ch in enumerate(EXT_CHANNELS)])
        tk.Label(self, text=f"Extension MVC  –  {ext_peak_str}", font=("Segoe UI", 10), bg="#1a2540", fg="#2ecc71").pack(pady=(0, 2))
        
        flex_peak_str = "  ".join([f"Ch{ch+1}: {self.mvc_peaks_flex[i]:.0f}" for i, ch in enumerate(FLEX_CHANNELS)])
        tk.Label(self, text=f"Flexion MVC  –  {flex_peak_str}", font=("Segoe UI", 10), bg="#1a2540", fg="#9b59b6").pack(pady=(0, 24))

        levels = [
            ("Level 1 – Piston Stamp  (Wrist Extension)", DEFAULT_DURATION_L1),
            ("Level 2 – Piston Stamp  (Wrist Flexion)", DEFAULT_DURATION_L2),
            ("Level 3 – Coin Collection Tunnel  (Combined)", DEFAULT_DURATION_L3),
        ]
        
        self._time_vars = []
        for name, default_time in levels:
            row_frame = tk.Frame(self, bg="#253154")
            row_frame.pack(padx=60, pady=6, fill=tk.X)
            
            tk.Label(row_frame, text=name, font=("Segoe UI", 12), bg="#253154", fg="#bdc3c7", width=46, anchor=tk.W).pack(side=tk.LEFT, padx=14, pady=10)
            
            time_var = tk.IntVar(value=default_time)
            spinner = tk.Spinbox(row_frame, from_=30, to=600, increment=30, textvariable=time_var, width=6, font=("Segoe UI", 12))
            spinner.pack(side=tk.LEFT, padx=8)
            
            tk.Label(row_frame, text="seconds", font=("Segoe UI", 11), bg="#253154", fg="#7f8c8d").pack(side=tk.LEFT)
            self._time_vars.append(time_var)

        tk.Button(self, text="Start Session →", font=("Segoe UI", 14, "bold"), bg="#2ecc71", fg="white", relief=tk.FLAT, padx=28, pady=12, cursor="hand2", command=self._start_session).pack(pady=28)

    def _start_session(self):
        durations = [var.get() for var in self._time_vars]
        self.on_start(*durations)


class InstructionScreen(tk.Frame):
    def __init__(self, master, title, text, btn_label, on_ok, start_img_label=None, end_img_label=None, start_image=IMG_NEUTRAL, end_image=IMG_WRIST_EXTENSION, start_caption="Starting position", end_caption="Target / end position"):
        super().__init__(master, bg="#1a2540")
        tk.Label(self, text=title, font=("Segoe UI", 22, "bold"), bg="#1a2540", fg="#ecf0f1").pack(pady=(28, 10))
        tk.Label(self, text=text, font=("Segoe UI", 12), bg="#1a2540", fg="#bdc3c7", justify=tk.LEFT).pack(padx=60, pady=6)

        if start_img_label or end_img_label:
            img_row = tk.Frame(self, bg="#1a2540")
            img_row.pack(padx=60, pady=8, fill=tk.X)
            self._instruction_images = []
            
            image_data = [
                (start_image, start_img_label, start_caption),
                (end_image, end_img_label, end_caption),
            ]
            
            for path, label_text, caption in image_data:
                if label_text is None:
                    continue
                    
                col_frame = tk.Frame(img_row, bg="#1a2540")
                col_frame.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=8)
                
                tk.Label(col_frame, text=caption, font=("Segoe UI", 10, "bold"), bg="#1a2540", fg="#bdc3c7").pack(pady=(0, 4))
                
                box = tk.Frame(col_frame, bg="#253154", relief=tk.RIDGE, bd=2)
                box.pack()
                
                photo = load_tk_image(path, (280, 180))
                self._instruction_images.append(photo)
                
                tk.Label(box, image=photo, bg="#253154").pack(padx=5, pady=(5, 2))
                tk.Label(box, text=label_text, font=("Segoe UI", 8, "italic"), bg="#253154", fg="#7f8c8d", justify=tk.CENTER, wraplength=260).pack(padx=5, pady=(0, 5))

        tk.Button(self, text=btn_label, font=("Segoe UI", 14, "bold"), bg="#2ecc71", fg="white", relief=tk.FLAT, padx=28, pady=12, cursor="hand2", command=on_ok).pack(pady=18)


class SessionResultsScreen(tk.Frame):
    def __init__(self, master, results):
        super().__init__(master, bg="#1a2540")
        self._build_ui(results)

    def _build_ui(self, results):
        tk.Label(self, text="Session Complete – Full Results", font=("Segoe UI", 22, "bold"), bg="#1a2540", fg="#2ecc71").pack(pady=(32, 8))
        tk.Label(self, text="University of Sussex  |  MSc Project (RAS)  |  Candidate 307631", font=("Segoe UI", 10), bg="#1a2540", fg="#7f8c8d").pack(pady=(0, 20))

        headers = ["", "Time Played", "Attempts", "Hits / Collected", "Accuracy", "Score"]
        col_widths = [34, 12, 10, 14, 10, 10]

        table = tk.Frame(self, bg="#253154")
        table.pack(padx=40, pady=8, fill=tk.X)

        def create_cell(parent, text, row, col, bold=False, fg_color="#bdc3c7"):
            font_weight = "bold" if bold else "normal"
            tk.Label(parent, text=text, font=("Segoe UI", 11, font_weight), bg="#253154", fg=fg_color, width=col_widths[col], anchor=tk.W, padx=10, pady=8).grid(row=row, column=col, sticky=tk.W)

        for col_idx, header in enumerate(headers):
            create_cell(table, header, 0, col_idx, bold=True, fg_color="#ecf0f1")

        rows = [
            ("L1 – Extension Piston", results[0]),
            ("L2 – Flexion Piston", results[1]),
            ("L3 – Tunnel", results[2]),
        ]
        
        for row_idx, (name, (elapsed, attempts, hits, score)) in enumerate(rows, 1):
            minutes, seconds = divmod(int(elapsed), 60)
            accuracy = f"{int(hits / attempts * 100)}%" if attempts else "—"
            
            create_cell(table, name, row_idx, 0, fg_color="#ecf0f1")
            create_cell(table, f"{minutes:02d}:{seconds:02d}", row_idx, 1)
            create_cell(table, str(attempts), row_idx, 2)
            create_cell(table, str(hits), row_idx, 3)
            create_cell(table, accuracy, row_idx, 4)
            create_cell(table, str(score), row_idx, 5)

        total_score = sum(results[i][3] for i in range(3))
        total_attempts = sum(results[i][1] for i in range(3))
        total_hits = sum(results[i][2] for i in range(3))
        total_accuracy = f"{int(total_hits / total_attempts * 100)}%" if total_attempts else "—"
        
        summary_frame = tk.Frame(self, bg="#1a2540")
        summary_frame.pack(pady=20)
        
        tk.Label(summary_frame, text=f"Total Score:  {total_score}", font=("Segoe UI", 18, "bold"), bg="#1a2540", fg="#2ecc71").pack()
        tk.Label(summary_frame, text=f"Total Accuracy:  {total_accuracy}", font=("Segoe UI", 16, "bold"), bg="#1a2540", fg="#ecf0f1").pack(pady=(8, 0))
        tk.Label(summary_frame, text=f"Successful actions:  {total_hits} out of {total_attempts}", font=("Segoe UI", 12), bg="#1a2540", fg="#bdc3c7").pack(pady=(2, 0))
        
        tk.Button(self, text="Close", font=("Segoe UI", 12), bg="#e74c3c", fg="white", relief=tk.FLAT, padx=20, pady=8, command=self.master.destroy).pack()


# --- Main Game Host Screen ---

class BaselineScreen(tk.Frame):
    """Live per-channel baseline selection to threshold out resting noise."""
    
    def __init__(self, master, device, info, groups, title, on_done):
        super().__init__(master, bg="#1a2540")
        self.device = device
        self.on_done = on_done
        
        self.channels = [ch for group_channels, *_ in groups for ch in group_channels]
        self.labels = [lbl for _, group_labels, *_ in groups for lbl in group_labels]
        self.colors = [col for *_, group_colors, _ in groups for col in group_colors]
        self.peaks = np.concatenate([group_peaks for *_, group_peaks in groups]).astype(np.float32)
        
        sample_rate = getattr(info, 'sample_rate', SAMPLE_RATE)
        self.preprocessor = EMGPreprocessor(len(self.channels), sample_rate)
        
        self._lock = threading.Lock()
        self._live_data = np.zeros(len(self.channels))
        
        self._values = [tk.StringVar(value="0") for _ in self.channels]
        self._cuts = [tk.StringVar(value="0") for _ in self.channels]
        
        self._build_ui(title)
        
        self.device.start_acquisition(on_block=self._on_block)
        self.after(80, self._refresh_data)

    def _build_ui(self, title):
        tk.Label(self, text=title, font=("Segoe UI", 18, "bold"), bg="#1a2540", fg="#ecf0f1").pack(pady=(20, 6))
        tk.Label(self, text="Relax, review each live RMS reading, then set a noise cut-off per channel.", font=("Segoe UI", 10), bg="#1a2540", fg="#bdc3c7").pack()
        
        self.chart = tk.Canvas(self, width=780, height=190, bg="#111b30", highlightthickness=0)
        self.chart.pack(padx=60, pady=(14, 4))
        
        table = tk.Frame(self, bg="#253154")
        table.pack(padx=60, pady=18, fill=tk.X)
        
        for row, (label, color, live_val, cut_val) in enumerate(zip(self.labels, self.colors, self._values, self._cuts), start=1):
            tk.Label(table, text=label, width=20, anchor=tk.W, font=("Segoe UI", 11, "bold"), bg="#253154", fg=color).grid(row=row, column=0, padx=12, pady=7)
            tk.Label(table, textvariable=live_val, width=12, bg="#253154", fg="#ecf0f1").grid(row=row, column=1)
            tk.Entry(table, textvariable=cut_val, width=12).grid(row=row, column=2, padx=12)
            
        tk.Label(table, text="Live RMS", bg="#253154", fg="#7f8c8d").grid(row=0, column=1, sticky=tk.N)
        tk.Label(table, text="Cut-off", bg="#253154", fg="#7f8c8d").grid(row=0, column=2, sticky=tk.N)
        
        buttons_frame = tk.Frame(self, bg="#1a2540")
        buttons_frame.pack(pady=8)
        tk.Button(buttons_frame, text="Use Current Readings", command=self._autoset_cuts).pack(side=tk.LEFT, padx=8)
        tk.Button(buttons_frame, text="Continue", bg="#2ecc71", fg="white", command=self._finalize_cuts).pack(side=tk.LEFT, padx=8)

    def _on_block(self, block):
        if block.emg_raw is not None:
            with self._lock:
                self._live_data = self.preprocessor.process(block.emg_raw[:, self.channels].copy())

    def _refresh_data(self):
        with self._lock:
            values = self._live_data.copy()
            
        for var, value in zip(self._values, values):
            var.set(f"{value:.1f}")
            
        self._draw_chart(values)
        self.after(80, self._refresh_data)

    def _draw_chart(self, values):
        self.chart.delete("all")
        width, height = 780, 190
        bar_width = max(48, (width - 70) // len(values) - 24)
        scale = max(1.0, float(self.peaks.max()))
        
        for i, (label, color, value, peak, cut) in enumerate(zip(self.labels, self.colors, values, self.peaks, self._cuts)):
            x_pos = 38 + i * (bar_width + 24)
            bottom, top = 155, 30
            
            # Draw background bar
            self.chart.create_rectangle(x_pos, top, x_pos + bar_width, bottom, fill="#253154", outline="#3d5170")
            
            # Draw live fill
            live_top = bottom - int(min(1.0, value / scale) * (bottom - top))
            self.chart.create_rectangle(x_pos, live_top, x_pos + bar_width, bottom, fill=color, outline="")
            
            # Parse user cutoff for line
            try:
                cutoff = max(0.0, float(cut.get()))
            except ValueError:
                cutoff = 0.0
                
            cut_y = bottom - int(min(1.0, cutoff / scale) * (bottom - top))
            self.chart.create_line(x_pos - 5, cut_y, x_pos + bar_width + 5, cut_y, fill="#2ecc71", width=3)
            
            # Labels
            self.chart.create_text(x_pos + bar_width // 2, 170, text=label, fill="#bdc3c7", font=("Segoe UI", 8))
            self.chart.create_text(x_pos + bar_width // 2, 18, text=f"MVC {peak:.0f}", fill="#7f8c8d", font=("Segoe UI", 8))
            
        self.chart.create_text(8, 16, text="RMS", fill="#7f8c8d", anchor=tk.W, font=("Segoe UI", 8))

    def _autoset_cuts(self):
        with self._lock:
            values = self._live_data.copy()
            
        for var, value in zip(self._cuts, values):
            var.set(f"{value:.1f}")

    def _finalize_cuts(self):
        try:
            cuts = np.array([float(var.get()) for var in self._cuts], dtype=np.float32)
        except ValueError:
            messagebox.showerror("Invalid cut-off", "Enter a numeric cut-off for every channel.")
            return
            
        if np.any(cuts < 0) or np.any(cuts >= self.peaks):
            messagebox.showerror("Invalid cut-off", "Each cut-off must be non-negative and below its MVC peak.")
            return
            
        self.device.stop_acquisition()
        self.on_done(cuts)

    def stop(self):
        self.device.stop_acquisition()


class GameScreen(tk.Frame):
    def __init__(self, master, device, info, mvc_peaks_ext, mvc_peaks_flex, durations, on_session_end, on_pre_level, on_before_level):
        super().__init__(master)
        self.device = device
        self.info = info
        self.mvc_peaks_ext = mvc_peaks_ext
        self.mvc_peaks_flex = mvc_peaks_flex
        self.durations = durations
        self.on_session_end = on_session_end
        self.on_pre_level = on_pre_level
        self.on_before_level = on_before_level
        
        self.ext_base = np.zeros(3, dtype=np.float32)
        self.flex_base = np.zeros(3, dtype=np.float32)
        
        self._level = 0
        self._game = None
        self._is_running = False
        self._is_transitioning = False
        self._baseline_screen = None
        self._instruction_screen = None
        self._results = []

        self.canvas = tk.Canvas(self, width=GAME_WIDTH, height=GAME_HEIGHT, bg="black", highlightthickness=0)
        self.canvas.pack()
        
        self.status = tk.Label(self, text="", font=("Segoe UI", 10), fg="#7f8c8d", bg="#1a2540")
        self.status.pack(fill=tk.X)

        self.canvas.focus_set()
        self.canvas.bind("<Button-1>", lambda event: self.canvas.focus_set())
        self.bind_all("<Return>", self._handle_return_key)
        self.bind_all("<Escape>", self._handle_escape_key)

        self.after(100, self._initialize_pygame)

    def _handle_return_key(self, event):
        if self._game and self._game.game_over:
            self._advance_level()

    def _handle_escape_key(self, event):
        if not self._game:
            return
            
        if self._game.game_over:
            self._advance_level()
        else:
            self._game.game_over = True

    def _initialize_pygame(self):
        os.environ['SDL_WINDOWID'] = str(self.canvas.winfo_id())
        os.environ['SDL_VIDEODRIVER'] = 'windib' if os.name == 'nt' else 'x11'
        
        pygame.init()
        pygame.display.init()
        
        self.pg_surface = pygame.display.set_mode((GAME_WIDTH, GAME_HEIGHT))
        self.clock = pygame.time.Clock()
        self._request_baselines()

    def _request_baselines(self):
        self._is_running = False
        self.canvas.pack_forget()
        self.status.pack_forget()
        self.on_pre_level(self._level, self, self._on_baselines_completed)

    def show_level_instruction(self, title, text, button_label, start_img_label=None, end_img_label=None, start_image=IMG_NEUTRAL, end_image=IMG_WRIST_EXTENSION, start_caption="Starting position", end_caption="Target / end position"):
        self.canvas.pack_forget()
        self.status.pack_forget()
        
        self._instruction_screen = InstructionScreen(
            self, title, text, button_label, self._on_instruction_completed,
            start_img_label, end_img_label, start_image, end_image,
            start_caption, end_caption
        )
        self._instruction_screen.pack(fill=tk.BOTH, expand=True)

    def _on_instruction_completed(self):
        if self._instruction_screen:
            self._instruction_screen.destroy()
            self._instruction_screen = None
            
        self.canvas.pack()
        self.status.pack(fill=tk.X)
        self.after(150, self._request_baselines)

    def _on_baselines_completed(self, ext_base, flex_base):
        self.ext_base = ext_base
        self.flex_base = flex_base
        
        if self._baseline_screen:
            self._baseline_screen.destroy()
            self._baseline_screen = None
            
        self.canvas.pack()
        self.status.pack(fill=tk.X)
        self._is_transitioning = False
        self.after(150, self._launch_level)

    def _launch_level(self):
        dur_1, dur_2, dur_3 = self.durations
        mv_ext = self.mvc_peaks_ext
        mv_flex = self.mvc_peaks_flex

        if self._level == 0:
            self._game = PistonGame(self.pg_surface, self.device, self.info, EXT_CHANNELS, EXT_WEIGHTS, mv_ext, self.ext_base, dur_1, "LEVEL 1 – Wrist Extension", flipped=False)
            self.status.config(text="Level 1: Wrist Extension – exceed 20 % MVC to stamp the target")
            
        elif self._level == 1:
            self._game = PistonGame(self.pg_surface, self.device, self.info, FLEX_CHANNELS, FLEX_WEIGHTS, mv_flex, self.flex_base, dur_2, "LEVEL 2 – Wrist Flexion", flipped=True)
            self.status.config(text="Level 2: Wrist Flexion – exceed 20 % MVC to stamp the target")
            
        elif self._level == 2:
            self._game = TunnelGame(self.pg_surface, self.device, self.info, EXT_CHANNELS, EXT_WEIGHTS, mv_ext, self.ext_base, FLEX_CHANNELS, FLEX_WEIGHTS, mv_flex, self.flex_base, dur_3)
            self.status.config(text="Level 3: Collect coins – extend above 20 % MVC to go up, flex above 20 % MVC to go down")

        self.device.start_acquisition(on_block=self._game._on_block)
        self._is_running = True
        self._game_loop()

    def _game_loop(self):
        if not self._is_running:
            return
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._is_running = False
                return
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self._game.game_over:
                        self._advance_level()
                    else:
                        self._game.game_over = True
                elif event.key == pygame.K_RETURN and self._game.game_over:
                    self._advance_level()

        delta_time = min(self.clock.tick(60) / 1000.0, 0.05)
        self._game.update(delta_time)
        self._game.draw()
        
        pygame.display.flip()
        self.after(0, self._game_loop)

    def _advance_level(self):
        if self._is_transitioning or not self._game or not self._game.game_over:
            return
            
        self._is_transitioning = True
        self._is_running = False
        
        game = self._game
        if isinstance(game, TunnelGame):
            level_results = (game.elapsed, game.total_coins, game.collected, game.score)
        else:
            level_results = (game.elapsed, game.attempts, game.hits, game.score)
            
        self._results.append(level_results)
        self._level += 1
        
        if self._level < 3:
            self.device.stop_acquisition()
            self.after(150, lambda: self.on_before_level(self._level, self))
        else:
            self._is_running = False
            self.device.stop_acquisition()
            self.after(0, lambda: self.on_session_end(self._results))

    def stop(self):
        self._is_running = False
        try:
            self.unbind_all("<Return>")
            self.unbind_all("<Escape>")
            self.device.stop_acquisition()
        except Exception:
            pass


# --- Application Controller ---

class WristRehabApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Wrist Rehabilitation Suite")
        self.root.configure(bg="#1a2540")
        self.root.geometry("980x720")
        self.root.resizable(False, False)
        
        self._device = None
        self._info = None
        self._mvc_peaks_ext = None
        self._mvc_peaks_flex = None
        self._durations = (DEFAULT_DURATION_L1, DEFAULT_DURATION_L2, DEFAULT_DURATION_L3)
        self._current_screen = None
        
        self._show_connect_screen()

    def _clear_screen(self):
        if self._current_screen:
            if hasattr(self._current_screen, 'stop'):
                self._current_screen.stop()
            self._current_screen.destroy()
            self._current_screen = None

    def _display(self, widget):
        self._clear_screen()
        self._current_screen = widget
        self._current_screen.pack(fill=tk.BOTH, expand=True)

    def _show_connect_screen(self):
        self._display(ConnectScreen(self.root, self._on_device_connected))

    def _on_device_connected(self, device, info):
        self._device = device
        self._info = info
        
        self._display(MVCScreen(
            self.root, device, info,
            EXT_CHANNELS, EXT_LABELS, EXT_COLORS,
            "MVC Calibration – Wrist Extension",
            "Wrist Extension",
            ("Push your wrist upward (extension) as hard as you can and hold.\n\n"
             "This records your Maximum Voluntary Contraction (MVC), which is used "
             "to normalise your muscle activity in the games that follow."),
            MVC_CSV_EXT, MVC_PEAKS_NPY,
            self._on_extension_mvc_completed
        ))

    def _on_extension_mvc_completed(self, mvc_peaks):
        self._mvc_peaks_ext = mvc_peaks
        
        self._display(MVCScreen(
            self.root, self._device, self._info,
            FLEX_CHANNELS, FLEX_LABELS, FLEX_COLORS,
            "MVC Calibration – Wrist Flexion",
            "Wrist Flexion",
            ("Curl your wrist downward (flexion) as hard as you can and hold.\n\n"
             "This records your Maximum Voluntary Contraction (MVC), which is used "
             "to normalise your muscle activity in the games that follow."),
            MVC_CSV_FLEX, MVC_PEAKS_NPY_FLEX,
            self._on_flexion_mvc_completed
        ))

    def _on_flexion_mvc_completed(self, mvc_peaks):
        self._mvc_peaks_flex = mvc_peaks
        self._display(TimingScreen(self.root, self._mvc_peaks_ext, self._mvc_peaks_flex, self._on_timing_configured))

    def _on_timing_configured(self, dur_1, dur_2, dur_3):
        self._durations = (dur_1, dur_2, dur_3)
        self._display(InstructionScreen(
            self.root,
            "Level 1 – Piston Stamp (Wrist Extension)",
            ("Items slide along a conveyor belt under a piston.\n\n"
             "  •  When an item stops, EXTEND your wrist upward.\n"
             "  •  The piston drives down proportional to your extension force.\n"
             "  •  Reach above 20 % MVC to stamp the item – 100 pts!\n"
             "  •  Relax your wrist fully to advance the belt to the next item.\n\n"
             f"Hit requirement: above 20 % MVC  |  Duration: {dur_1} s"),
            "OK – Start Level 1",
            self._start_game_sequence,
            start_img_label="Forearm flat on table, palm down,\nwrist in neutral (starting) position",
            end_img_label="Wrist extended upward against\nresistance – target contraction position",
        ))

    def _start_game_sequence(self):
        self._display(GameScreen(
            self.root, self._device, self._info,
            self._mvc_peaks_ext, self._mvc_peaks_flex,
            self._durations, on_session_end=self._on_session_finished,
            on_pre_level=self._handle_pre_level, on_before_level=self._handle_before_level,
        ))

    def _handle_before_level(self, level, host):
        dur_1, dur_2, dur_3 = self._durations
        
        if level == 1:
            host.show_level_instruction(
                "Level 2 – Piston Stamp (Wrist Flexion)",
                ("Items slide along a conveyor belt under a piston.\n\n"
                 "  •  When an item stops, FLEX your wrist downward.\n"
                 "  •  The piston rises proportional to your flexion force.\n"
                 "  •  Reach above 20 % MVC to stamp the item – 100 pts!\n"
                 "  •  Relax your wrist fully to advance the belt to the next item.\n\n"
                 f"Hit requirement: above 20 % MVC  |  Duration: {dur_2} s"),
                "OK – Start Level 2",
                start_img_label="Forearm flat on table, palm down,\nwrist in neutral (starting) position",
                end_img_label="Curl wrist downward against\nresistance – target contraction position",
                end_image=IMG_WRIST_FLEXION,
            )
        elif level == 2:
            host.show_level_instruction(
                "Level 3 – Coin Collection Tunnel",
                ("Guide the craft through the tunnel to collect the gold coins.\n\n"
                 "  •  EXTEND your wrist to move the craft upward.\n"
                 "  •  FLEX your wrist to move the craft downward.\n"
                 "  •  Touch each gold coin to collect it – 50 pts!\n"
                 "  •  Stay within the tunnel walls while lining up with each coin.\n\n"
                 f"Collection goal: touch coins  |  Duration: {dur_3} s  |  Both muscle groups active"),
                "OK – Start Level 3",
                start_img_label="Wrist extension moves the craft upward",
                end_img_label="Wrist flexion moves the craft downward",
                start_image=IMG_WRIST_EXTENSION,
                end_image=IMG_WRIST_FLEXION,
                start_caption="Move up",
                end_caption="Move down",
            )

    def _handle_pre_level(self, level, host, resume_callback):
        if level == 0:
            title = "Level 1 - Extension Noise Baselines"
            groups = [(EXT_CHANNELS, EXT_LABELS, EXT_COLORS, self._mvc_peaks_ext)]
        elif level == 1:
            title = "Level 2 - Flexion Noise Baselines"
            groups = [(FLEX_CHANNELS, FLEX_LABELS, FLEX_COLORS, self._mvc_peaks_flex)]
        else:
            title = "Level 3 - Extension and Flexion Noise Baselines"
            groups = [(EXT_CHANNELS, EXT_LABELS, EXT_COLORS, self._mvc_peaks_ext),
                      (FLEX_CHANNELS, FLEX_LABELS, FLEX_COLORS, self._mvc_peaks_flex)]
                      
        def on_done(cuts):
            if level == 0:
                resume_callback(cuts, np.zeros(3, dtype=np.float32))
            elif level == 1:
                resume_callback(np.zeros(3, dtype=np.float32), cuts)
            else:
                resume_callback(cuts[:3], cuts[3:])
                
        host._baseline_screen = BaselineScreen(host, self._device, self._info, groups, title, on_done)
        host._baseline_screen.pack(fill=tk.BOTH, expand=True)

    def _on_session_finished(self, results):
        self._display(SessionResultsScreen(self.root, results))

    def _handle_close(self):
        try:
            if self._device:
                self._device.stop_acquisition()
                self._device.disconnect()
        except Exception:
            pass
            
        try:
            pygame.quit()
        except Exception:
            pass
            
        self.root.destroy()

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._handle_close)
        self.root.mainloop()


if __name__ == "__main__":
    app = WristRehabApp()
    app.run()
