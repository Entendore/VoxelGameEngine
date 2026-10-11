import pygame
import math
import random
import sys
import os
import threading
import queue
import numpy as np
from numba import njit
import time
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

try:
    import cupy as cp
    print("CuPy detected. Using GPU for Cellular Automata.")
    USE_GPU = True
except ImportError:
    print("CuPy not found. Falling back to NumPy for CA.")
    USE_GPU = False

# --- Configuration ---
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60

WORLD_SIZE_X = 128
WORLD_SIZE_Y = 128
WORLD_SIZE_Z = 40
CHUNK_SIZE = 16
SAVE_FILE = "world_save.npy"
NUM_TEXTURES = 42  # Expanded to fit all new blocks

# --- Block Types --- (name, transparent, light, is_emissive)
BLOCK_TYPES = {
    1:  ("Grass",        False, 0, False),
    2:  ("Dirt",         False, 0, False),
    3:  ("Stone",        False, 0, False),
    4:  ("Sand",         False, 0, False),
    5:  ("Water",        True,  0, False),
    6:  ("Wood",         False, 0, False),
    7:  ("Leaves",       False, 0, False),
    8:  ("Glass",        True,  0, False),
    9:  ("Snow",         False, 0, False),
    10: ("Coal Ore",     False, 0, False),
    11: ("Iron Ore",     False, 0, False),
    12: ("Glowstone",    False, 15, True),
    13: ("Diamond",      False, 0, False),
    14: ("Desert Sand",  False, 0, False),
    15: ("Cloud",        True,  0, False),
    16: ("Planks",       False, 0, False),
    17: ("Cobblestone",  False, 0, False),
    18: ("Bricks",       False, 0, False),
    19: ("Sandstone",    False, 0, False),
    20: ("Obsidian",     False, 0, False),
    21: ("Gold Ore",     False, 0, False),
    22: ("Copper Ore",   False, 0, False),
    23: ("Ice",          True,  0, False),
    24: ("Packed Ice",   False, 0, False),
    25: ("Cactus",       False, 0, False),
    26: ("Clay",         False, 0, False),
    27: ("Mossy Stone",  False, 0, False),
    28: ("Marble",       False, 0, False),
    29: ("Lava",         True,  15, True),
    30: ("Bedrock",      False, 0, False),
    31: ("Netherrack",   False, 0, False),
    32: ("End Stone",    False, 0, False),
    33: ("Pumpkin",      False, 0, False),
    34: ("Mud",          False, 0, False),
    35: ("TNT",          False, 0, False),
}

BLOCK_PARTICLE_COLORS = {
    1: (86, 125, 70), 2: (134, 96, 67), 3: (128, 128, 128), 4: (194, 178, 128),
    5: (50, 100, 200), 6: (80, 50, 20), 7: (50, 120, 40), 8: (200, 220, 255),
    9: (240, 240, 250), 10: (40, 40, 40), 11: (180, 140, 90), 12: (240, 220, 80),
    13: (120, 240, 240), 14: (210, 190, 140), 15: (240, 240, 245), 16: (160, 120, 60),
    17: (100, 100, 100), 18: (150, 50, 40), 19: (220, 200, 150), 20: (20, 10, 30),
    21: (240, 200, 50), 22: (200, 120, 80), 23: (150, 200, 255), 24: (100, 160, 220),
    25: (50, 130, 50), 26: (160, 170, 180), 27: (90, 110, 80), 28: (230, 230, 235),
    29: (230, 90, 20), 30: (50, 50, 50), 31: (100, 40, 40), 32: (220, 220, 160),
    33: (230, 140, 30), 34: (80, 60, 40), 35: (200, 40, 30)
}

# ==========================================
# PROCEDURAL TEXTURE ATLAS
# ==========================================
def get_texture_id(block_id, face_id):
    # face: 0=top (+Z), 1=bottom (-Z), 2,3,4,5=sides
    if block_id == 1: return 0 if face_id == 0 else (2 if face_id == 1 else 1)
    if block_id == 2: return 2
    if block_id == 3: return 3
    if block_id == 4: return 4
    if block_id == 5: return 5
    if block_id == 6: return 6 if face_id <= 1 else 7
    if block_id == 7: return 8
    if block_id == 8: return 9
    if block_id == 9: return 10
    if block_id == 10: return 11
    if block_id == 11: return 12
    if block_id == 12: return 13
    if block_id == 13: return 14
    if block_id == 14: return 15
    if block_id == 15: return 16
    if block_id == 16: return 17
    if block_id == 17: return 18
    if block_id == 18: return 19
    if block_id == 19: return 20 if face_id == 0 else 21
    if block_id == 20: return 22
    if block_id == 21: return 23
    if block_id == 22: return 24
    if block_id == 23: return 25
    if block_id == 24: return 26
    if block_id == 25: return 27 if face_id == 0 else 28
    if block_id == 26: return 29
    if block_id == 27: return 30
    if block_id == 28: return 31
    if block_id == 29: return 32
    if block_id == 30: return 33
    if block_id == 31: return 34
    if block_id == 32: return 35
    if block_id == 33: return 36 if face_id == 0 else 37
    if block_id == 34: return 38
    if block_id == 35: return 39 if face_id == 0 else 40
    return 0

def create_texture_atlas():
    size = 16
    atlas = pygame.Surface((size * NUM_TEXTURES, size), pygame.SRCALPHA)
    
    def draw_noise(rect, color, variance, alpha=255):
        for x in range(rect.w):
            for y in range(rect.h):
                n = random.randint(-variance, variance)
                c = (max(0, min(255, color[0]+n)), max(0, min(255, color[1]+n)), max(0, min(255, color[2]+n)), alpha)
                atlas.set_at((rect.x + x, rect.y + y), c)

    def draw_ore(rect, base_color, ore_color):
        draw_noise(rect, base_color, 20)
        for _ in range(4):
            ox = random.randint(2, 13); oy = random.randint(2, 13)
            pygame.draw.circle(atlas, ore_color, (rect.x + ox, rect.y + oy), random.randint(1, 2))

    def draw_lines_h(rect, color, step=4):
        for y in range(0, rect.h, step):
            pygame.draw.line(atlas, color, (rect.x, rect.y + y), (rect.x + rect.w, rect.y + y))

    # 0: Grass Top
    draw_noise(pygame.Rect(0, 0, 16, 16), (86, 125, 70), 25)
    # 1: Grass Side
    draw_noise(pygame.Rect(16, 0, 16, 16), (134, 96, 67), 20)
    pygame.draw.rect(atlas, (86, 125, 70), pygame.Rect(16, 0, 16, 4))
    for x in range(16):
        for y in range(4, 6):
            if random.random() > 0.5: atlas.set_at((16+x, y), (86+random.randint(-25,25), 125+random.randint(-25,25), 70+random.randint(-25,25), 255))
    # 2: Dirt
    draw_noise(pygame.Rect(32, 0, 16, 16), (134, 96, 67), 20)
    # 3: Stone
    draw_noise(pygame.Rect(48, 0, 16, 16), (128, 128, 128), 20)
    # 4: Sand
    draw_noise(pygame.Rect(64, 0, 16, 16), (194, 178, 128), 15)
    # 5: Water
    draw_noise(pygame.Rect(80, 0, 16, 16), (50, 100, 200), 20, alpha=180)
    # 6: Wood Top
    draw_noise(pygame.Rect(96, 0, 16, 16), (80, 50, 20), 15)
    pygame.draw.circle(atlas, (60, 40, 15), (104, 8), 3)
    # 7: Wood Side
    draw_noise(pygame.Rect(112, 0, 16, 16), (80, 50, 20), 10)
    draw_lines_h(pygame.Rect(112, 0, 16, 16), (60, 40, 15), 4)
    # 8: Leaves (Alpha Cutout)
    draw_noise(pygame.Rect(128, 0, 16, 16), (50, 120, 40), 30)
    for _ in range(20): atlas.set_at((128+random.randint(0,15), random.randint(0,15)), (0,0,0,0))
    # 9: Glass
    atlas.fill((200, 220, 255, 100), pygame.Rect(144, 0, 16, 16))
    pygame.draw.rect(atlas, (255, 255, 255, 200), pygame.Rect(144, 0, 16, 16), 1)
    # 10: Snow
    draw_noise(pygame.Rect(160, 0, 16, 16), (240, 240, 250), 10)
    # 11: Coal Ore
    draw_ore(pygame.Rect(176, 0, 16, 16), (128, 128, 128), (40, 40, 40))
    # 12: Iron Ore
    draw_ore(pygame.Rect(192, 0, 16, 16), (128, 128, 128), (180, 140, 90))
    # 13: Glowstone
    draw_noise(pygame.Rect(208, 0, 16, 16), (240, 220, 80), 30)
    for _ in range(5): pygame.draw.circle(atlas, (255, 255, 200), (208 + random.randint(2,13), random.randint(2,13)), 1)
    # 14: Diamond
    draw_ore(pygame.Rect(224, 0, 16, 16), (128, 128, 128), (120, 240, 240))
    # 15: Desert Sand
    draw_noise(pygame.Rect(240, 0, 16, 16), (210, 190, 140), 15)
    # 16: Cloud
    draw_noise(pygame.Rect(256, 0, 16, 16), (240, 240, 245), 10, alpha=180)
    # 17: Planks
    draw_noise(pygame.Rect(272, 0, 16, 16), (160, 120, 60), 10)
    draw_lines_h(pygame.Rect(272, 0, 16, 16), (120, 80, 30), 4)
    # 18: Cobblestone
    draw_noise(pygame.Rect(288, 0, 16, 16), (100, 100, 100), 25)
    for _ in range(3): pygame.draw.line(atlas, (60,60,60), (288+random.randint(0,15), random.randint(0,15)), (288+random.randint(0,15), random.randint(0,15)), 1)
    # 19: Bricks
    draw_noise(pygame.Rect(304, 0, 16, 16), (150, 50, 40), 10)
    pygame.draw.rect(atlas, (200, 200, 200), pygame.Rect(304, 0, 16, 4))
    pygame.draw.rect(atlas, (200, 200, 200), pygame.Rect(304, 8, 16, 4))
    pygame.draw.rect(atlas, (200, 200, 200), pygame.Rect(304+8, 0, 1, 4))
    pygame.draw.rect(atlas, (200, 200, 200), pygame.Rect(304+4, 4, 1, 4))
    pygame.draw.rect(atlas, (200, 200, 200), pygame.Rect(304+12, 4, 1, 4))
    pygame.draw.rect(atlas, (200, 200, 200), pygame.Rect(304+0, 8, 1, 4))
    pygame.draw.rect(atlas, (200, 200, 200), pygame.Rect(304+8, 8, 1, 4))
    pygame.draw.rect(atlas, (200, 200, 200), pygame.Rect(304+4, 12, 1, 4))
    pygame.draw.rect(atlas, (200, 200, 200), pygame.Rect(304+12, 12, 1, 4))
    # 20: Sandstone Top
    draw_noise(pygame.Rect(320, 0, 16, 16), (220, 200, 150), 5)
    # 21: Sandstone Side
    draw_noise(pygame.Rect(336, 0, 16, 16), (220, 200, 150), 10)
    pygame.draw.rect(atlas, (200, 180, 130), pygame.Rect(336, 0, 16, 2))
    pygame.draw.rect(atlas, (200, 180, 130), pygame.Rect(336, 14, 16, 2))
    # 22: Obsidian
    draw_noise(pygame.Rect(352, 0, 16, 16), (20, 10, 30), 10)
    for _ in range(3): pygame.draw.circle(atlas, (80, 50, 100), (352+random.randint(2,13), random.randint(2,13)), 1)
    # 23: Gold Ore
    draw_ore(pygame.Rect(368, 0, 16, 16), (128, 128, 128), (240, 200, 50))
    # 24: Copper Ore
    draw_ore(pygame.Rect(384, 0, 16, 16), (128, 128, 128), (200, 120, 80))
    # 25: Ice
    draw_noise(pygame.Rect(400, 0, 16, 16), (150, 200, 255), 15, alpha=180)
    for _ in range(4): pygame.draw.line(atlas, (255,255,255), (400+random.randint(0,15), random.randint(0,15)), (400+random.randint(0,15), random.randint(0,15)), 1)
    # 26: Packed Ice
    draw_noise(pygame.Rect(416, 0, 16, 16), (100, 160, 220), 15, alpha=255)
    # 27: Cactus Top
    draw_noise(pygame.Rect(432, 0, 16, 16), (50, 130, 50), 15)
    pygame.draw.rect(atlas, (30, 100, 30), pygame.Rect(432, 0, 16, 16), 2)
    # 28: Cactus Side
    draw_noise(pygame.Rect(448, 0, 16, 16), (50, 130, 50), 15)
    pygame.draw.rect(atlas, (30, 100, 30), pygame.Rect(448, 0, 2, 16))
    pygame.draw.rect(atlas, (30, 100, 30), pygame.Rect(448+14, 0, 2, 16))
    # 29: Clay
    draw_noise(pygame.Rect(464, 0, 16, 16), (160, 170, 180), 10)
    # 30: Mossy Stone
    draw_noise(pygame.Rect(480, 0, 16, 16), (90, 110, 80), 20)
    # 31: Marble
    draw_noise(pygame.Rect(496, 0, 16, 16), (230, 230, 235), 5)
    pygame.draw.line(atlas, (180,180,180), (496, 4), (512, 12), 1)
    pygame.draw.line(atlas, (180,180,180), (496, 10), (512, 6), 1)
    # 32: Lava
    draw_noise(pygame.Rect(512, 0, 16, 16), (230, 90, 20), 20)
    for _ in range(6): pygame.draw.circle(atlas, (255, 200, 50), (512+random.randint(2,13), random.randint(2,13)), 1)
    # 33: Bedrock
    draw_noise(pygame.Rect(528, 0, 16, 16), (50, 50, 50), 15)
    # 34: Netherrack
    draw_noise(pygame.Rect(544, 0, 16, 16), (100, 40, 40), 20)
    for _ in range(2): pygame.draw.line(atlas, (60,20,20), (544+random.randint(0,15), random.randint(0,15)), (544+random.randint(0,15), random.randint(0,15)), 1)
    # 35: End Stone
    draw_noise(pygame.Rect(560, 0, 16, 16), (220, 220, 160), 10)
    # 36: Pumpkin Top
    draw_noise(pygame.Rect(576, 0, 16, 16), (230, 140, 30), 15)
    pygame.draw.rect(atlas, (100, 60, 20), pygame.Rect(576+6, 6, 4, 4))
    # 37: Pumpkin Side
    draw_noise(pygame.Rect(592, 0, 16, 16), (230, 140, 30), 15)
    for x in range(0, 16, 2): pygame.draw.line(atlas, (180, 100, 20), (592+x, 0), (592+x, 16))
    # 38: Mud
    draw_noise(pygame.Rect(608, 0, 16, 16), (80, 60, 40), 15)
    # 39: TNT Top
    draw_noise(pygame.Rect(624, 0, 16, 16), (200, 40, 30), 10)
    pygame.draw.circle(atlas, (50,50,50), (624+8, 8), 3)
    # 40: TNT Side
    draw_noise(pygame.Rect(640, 0, 16, 16), (200, 40, 30), 10)
    for y in range(0, 16, 2): pygame.draw.line(atlas, (255,255,255), (640, y), (640+16, y), 1)

    return atlas

# ==========================================
# GPU: CELLULAR AUTOMATA
# ==========================================
def generate_ca_grid_gpu(width, height, seed, fill_prob=0.3, iters=5):
    xp = cp if USE_GPU else np
    xp.random.seed(seed)
    grid = (xp.random.rand(width, height) < fill_prob).astype(xp.int32)
    for _ in range(iters):
        padded = xp.pad(grid, 1, mode='constant', constant_values=0)
        neighbors = (padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
                     padded[1:-1, :-2] +                      padded[1:-1, 2:] +
                     padded[2:, :-2]  + padded[2:, 1:-1]  + padded[2:, 2:])
        grid = (neighbors >= 5).astype(xp.int32)
    return cp.asnumpy(grid) if USE_GPU else grid

# ==========================================
# CPU: NUMBA PROCEDURAL GEN
# ==========================================
@njit(fastmath=True)
def value_noise_2d(x, y, seed):
    n = (int(x) * 374761393 + int(y) * 668265263 + seed * 362437) & 0xFFFFFFFF
    n = (n ^ (n >> 13)) * 1274126177 & 0xFFFFFFFF
    return ((n ^ (n >> 16)) & 0xFFFFFFFF) / 0xFFFFFFFF

@njit(fastmath=True)
def fade(t): return t * t * t * (t * (t * 6 - 15) + 10)

@njit(fastmath=True)
def lerp(a, b, t): return a + t * (b - a)

@njit(fastmath=True)
def fbm_2d(x, y, seed, octaves=4, persistence=0.5, lacunarity=2.0):
    total = 0.0; amplitude = 1.0; frequency = 1.0; max_value = 0.0
    for _ in range(octaves):
        xi, yi = int(x * frequency), int(y * frequency)
        xf, yf = x * frequency - xi, y * frequency - yi
        v00 = value_noise_2d(xi, yi, seed); v01 = value_noise_2d(xi, yi+1, seed)
        v10 = value_noise_2d(xi+1, yi, seed); v11 = value_noise_2d(xi+1, yi+1, seed)
        u, v = fade(xf), fade(yf)
        total += lerp(lerp(v00, v10, u), lerp(v01, v11, u), v) * amplitude
        max_value += amplitude; amplitude *= persistence; frequency *= lacunarity
    return total / max_value

@njit
def generate_terrain_numba(world, ca_grid, seed, water_level):
    sx, sy, sz = world.shape
    for x in range(sx):
        for y in range(sy):
            warp_x = (fbm_2d(x*0.05, y*0.05, seed, 4, 0.5, 2.0) - 0.5) * 20
            warp_y = (fbm_2d(x*0.05+5.0, y*0.05+5.0, seed, 4, 0.5, 2.0) - 0.5) * 20
            h_noise = fbm_2d((x + warp_x)*0.03, (y + warp_y)*0.03, seed, 4, 0.5, 2.0)
            height = int(h_noise * 20 + 4)
            moisture = fbm_2d(x*0.08, y*0.08, seed+1, 2, 0.5, 2.0)
            is_desert = moisture < 0.4
            is_snowy = moisture > 0.7 and height > 12
            
            for z in range(sz):
                if z < height:
                    n1 = fbm_2d(x*0.1, z*0.1, seed+2, 2, 0.5, 2.0)
                    n2 = fbm_2d(y*0.1, z*0.1, seed+3, 2, 0.5, 2.0)
                    if n1 > 0.75 and n2 > 0.75: world[x, y, z] = 0
                    else:
                        if z == 0: world[x, y, z] = 30 # Bedrock
                        elif z == height - 1 and height > water_level:
                            if is_snowy: world[x, y, z] = 9
                            elif is_desert: world[x, y, z] = 14
                            else: world[x, y, z] = 1
                        elif z >= height - 3:
                            if is_desert and height >= water_level: world[x, y, z] = 14
                            elif is_snowy: world[x, y, z] = 9
                            else: world[x, y, z] = 2
                        else:
                            world[x, y, z] = 3
                            if z < 12:
                                ore_n = fbm_2d(x*0.2, (y+z)*0.2, seed+4, 2, 0.5, 2.0)
                                if ore_n > 0.85: world[x, y, z] = 10
                                elif ore_n < 0.15 and z < 6: world[x, y, z] = 13
                                elif ore_n > 0.8 and z < 10: world[x, y, z] = 11
                                elif ore_n > 0.82 and z < 8: world[x, y, z] = 21
                                elif ore_n < 0.12 and z < 8: world[x, y, z] = 22
                                elif n1 < 0.3 and z < 5: world[x, y, z] = 29 # Lava pools
                                elif n1 < 0.4 and n2 < 0.4: world[x, y, z] = 28 # Marble
                                elif n1 > 0.6 and n2 > 0.6: world[x, y, z] = 27 # Mossy Stone
                elif z < water_level:
                    if is_snowy and z == water_level - 1: world[x, y, z] = 23 # Ice on top of water
                    else: world[x, y, z] = 5 # Water
                else:
                    world[x, y, z] = 0

            if height > water_level and ca_grid[x, y] == 1:
                for bx in range(-1, 2):
                    for by in range(-1, 2):
                        for bz in range(0, 2):
                            if (bx*bx + by*by + bz*bz) < 2.25:
                                nx, ny, nz = x+bx, y+by, height+bz
                                if 0 <= nx < sx and 0 <= ny < sy and 0 <= nz < sz: world[nx, ny, nz] = 17 # Cobblestone boulders

            tree_n = fbm_2d(x*0.2, y*0.2, seed+5, 1, 0.5, 2.0)
            if height > water_level and not is_desert and not is_snowy and tree_n > 0.7 and random.random() < 0.4:
                trunk_h = random.randint(2, 4)
                for tz in range(height, height + trunk_h):
                    if 0 <= tz < sz: world[x, y, tz] = 6
                for lx in range(-2, 3):
                    for ly in range(-2, 3):
                        for lz in range(height + trunk_h - 1, height + trunk_h + 2):
                            if (lx*lx + ly*ly) < 6.25:
                                if lx == 0 and ly == 0 and lz < height + trunk_h: continue
                                nx, ny, nz = x + lx, y + ly, lz
                                if 0 <= nx < sx and 0 <= ny < sy and 0 <= nz < sz: world[nx, ny, nz] = 7
            
            # Cactus in deserts
            if height > water_level and is_desert and tree_n > 0.8 and random.random() < 0.3:
                for tz in range(height, height + random.randint(2, 4)):
                    if 0 <= tz < sz: world[x, y, tz] = 25

            # Pumpkins occasionally on grass
            if height > water_level and not is_desert and not is_snowy and tree_n < 0.2 and random.random() < 0.005:
                if 0 <= height < sz: world[x, y, height] = 33

@njit
def generate_clouds_numba(world, seed, cloud_z=34, threshold=0.65):
    sx, sy, sz = world.shape
    for x in range(sx):
        for y in range(sy):
            c = fbm_2d(x*0.06, y*0.06, seed+99, 3, 0.5, 2.0)
            if c > threshold:
                if world[x, y, cloud_z] == 0: world[x, y, cloud_z] = 15

# ==========================================
# SHADERS
# ==========================================
VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aUV;
layout(location = 2) in vec3 aNormal;
out vec2 vUV;
out vec3 vNormal;
out float vDepth;
uniform mat4 uMVP;
uniform mat4 uView;
void main() {
    vec4 viewPos = uView * vec4(aPos, 1.0);
    vDepth = -viewPos.z;
    gl_Position = uMVP * vec4(aPos, 1.0);
    vUV = aUV; vNormal = aNormal;
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec2 vUV;
in vec3 vNormal;
in float vDepth;
out vec4 FragColor;
uniform sampler2D uAtlas;
uniform vec3 uSunDir;
uniform vec3 uSunColor;
uniform float uAmbient;
uniform vec3 uFogColor;
uniform float uFogDensity;

float get_face_ao(vec3 normal) {
    if (normal.z > 0.5) return 1.0;      // Top
    if (normal.z < -0.5) return 0.6;     // Bottom
    return 0.8;                          // Sides
}

void main() {
    vec4 texColor = texture(uAtlas, vUV);
    if (texColor.a < 0.1) discard;
    
    float ao = get_face_ao(vNormal);
    float diff = max(dot(normalize(vNormal), normalize(uSunDir)), 0.0);
    vec3 light = (uAmbient * ao) + (diff * uSunColor * ao);
    
    vec3 lit = texColor.rgb * light;
    
    // Exponential squared fog
    float fogFactor = 1.0 - exp(-pow(vDepth * uFogDensity, 2.0));
    vec3 finalColor = mix(lit, uFogColor, fogFactor);
    
    FragColor = vec4(finalColor, texColor.a);
}
"""

POST_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexCoord;
out vec2 vTexCoord;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    vTexCoord = aTexCoord;
}
"""

POST_FRAGMENT_SHADER = """
#version 330 core
in vec2 vTexCoord;
out vec4 FragColor;
uniform sampler2D uScene;
uniform float uExposure;

vec3 ACESFilm(vec3 x) {
    float a = 2.51; float b = 0.03; float c = 2.43; float d = 0.59; float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 color = texture(uScene, vTexCoord).rgb;
    color *= uExposure;
    color = ACESFilm(color);
    color = pow(color, vec3(1.0 / 2.2)); // Gamma correction
    FragColor = vec4(color, 1.0);
}
"""

LINE_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 uMVP;
void main() { gl_Position = uMVP * vec4(aPos, 1.0); }
"""

LINE_FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;
uniform vec3 uColor;
void main() { FragColor = vec4(uColor, 1.0); }
"""

UI_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexCoord;
out vec2 vTexCoord;
void main() { gl_Position = vec4(aPos, 0.0, 1.0); vTexCoord = aTexCoord; }
"""

UI_FRAGMENT_SHADER = """
#version 330 core
in vec2 vTexCoord;
out vec4 FragColor;
uniform sampler2D uTexture;
void main() { FragColor = texture(uTexture, vTexCoord); }
"""

# ==========================================
# MATH HELPERS & FRUSTUM
# ==========================================
def mat_identity(): return np.identity(4, dtype=np.float32)

def mat_ortho(l, r, b, t, n, f):
    return np.array([
        [2/(r-l), 0, 0, -(r+l)/(r-l)], [0, 2/(t-b), 0, -(t+b)/(t-b)],
        [0, 0, -2/(f-n), -(f+n)/(f-n)], [0, 0, 0, 1]
    ], dtype=np.float32)

def mat_translate(x, y, z):
    return np.array([[1,0,0,x],[0,1,0,y],[0,0,1,z],[0,0,0,1]], dtype=np.float32)

def mat_rotate_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]], dtype=np.float32)

def mat_rotate_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]], dtype=np.float32)

def unproject(mx, my, proj, view):
    ndc_x = (2.0 * mx) / SCREEN_WIDTH - 1.0
    ndc_y = 1.0 - (2.0 * my) / SCREEN_HEIGHT
    inv_vp = np.linalg.inv(proj @ view)
    near_point = inv_vp @ np.array([ndc_x, ndc_y, -1.0, 1.0], dtype=np.float32)
    far_point  = inv_vp @ np.array([ndc_x, ndc_y,  1.0, 1.0], dtype=np.float32)
    near_point /= near_point[3]; far_point  /= far_point[3]
    return near_point[:3], far_point[:3]

def get_frustum_planes(mvp):
    m = mvp.T
    planes = np.array([m[3] + m[0], m[3] - m[0], m[3] + m[1], m[3] - m[1], m[3] + m[2], m[3] - m[2]])
    for i in range(6):
        p = planes[i]; l = np.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
        if l > 0: planes[i] = p / l
    return planes

def is_chunk_in_frustum(planes, cx, cy):
    center = np.array([cx*16+8, cy*16+8, 20]); radius = 24
    for p in planes:
        if p[0]*center[0] + p[1]*center[1] + p[2]*center[2] + p[3] < -radius: return False
    return True

def raycast_voxels(world, origin, direction):
    ix, iy, iz = int(math.floor(origin[0])), int(math.floor(origin[1])), int(math.floor(origin[2]))
    step_x = 1 if direction[0] > 0 else -1; step_y = 1 if direction[1] > 0 else -1; step_z = 1 if direction[2] > 0 else -1
    t_delta_x = abs(1.0 / direction[0]) if abs(direction[0]) > 1e-6 else 1e12
    t_delta_y = abs(1.0 / direction[1]) if abs(direction[1]) > 1e-6 else 1e12
    t_delta_z = abs(1.0 / direction[2]) if abs(direction[2]) > 1e-6 else 1e12

    def calc_t_max(o, i, d, step):
        edge = (i + 1.0) if step > 0 else i * 1.0
        if abs(d) < 1e-6: return 1e12
        return (edge - o) / d

    t_max_x = calc_t_max(origin[0], ix, direction[0], step_x)
    t_max_y = calc_t_max(origin[1], iy, direction[1], step_y)
    t_max_z = calc_t_max(origin[2], iz, direction[2], step_z)
    max_dist = math.sqrt(WORLD_SIZE_X**2 + WORLD_SIZE_Y**2 + WORLD_SIZE_Z**2) * 2
    t = 0.0; normal = (0, 0, 0)

    if 0 <= ix < WORLD_SIZE_X and 0 <= iy < WORLD_SIZE_Y and 0 <= iz < WORLD_SIZE_Z:
        if world[ix, iy, iz] != 0: return (ix, iy, iz), (0, 0, 0)

    while t < max_dist:
        if t_max_x < t_max_y:
            if t_max_x < t_max_z: ix += step_x; t = t_max_x; t_max_x += t_delta_x; normal = (-step_x, 0, 0)
            else: iz += step_z; t = t_max_z; t_max_z += t_delta_z; normal = (0, 0, -step_z)
        else:
            if t_max_y < t_max_z: iy += step_y; t = t_max_y; t_max_y += t_delta_y; normal = (0, -step_y, 0)
            else: iz += step_z; t = t_max_z; t_max_z += t_delta_z; normal = (0, 0, -step_z)
        if 0 <= ix < WORLD_SIZE_X and 0 <= iy < WORLD_SIZE_Y and 0 <= iz < WORLD_SIZE_Z:
            if world[ix, iy, iz] != 0: return (ix, iy, iz), normal
    return None, None

# ==========================================
# DAY/NIGHT
# ==========================================
SKY_PHASES = [
    (0.00, (0.02, 0.03, 0.08), 0.30), (0.20, (0.05, 0.06, 0.12), 0.35),
    (0.25, (0.95, 0.55, 0.35), 0.55), (0.35, (0.55, 0.75, 0.95), 0.95),
    (0.50, (0.40, 0.65, 0.95), 1.00), (0.65, (0.55, 0.70, 0.95), 0.95),
    (0.75, (0.85, 0.45, 0.55), 0.55), (0.80, (0.15, 0.10, 0.20), 0.40),
    (1.00, (0.02, 0.03, 0.08), 0.30),
]

def get_sky_and_ambient(t):
    for i in range(len(SKY_PHASES) - 1):
        t0, c0, a0 = SKY_PHASES[i]; t1, c1, a1 = SKY_PHASES[i+1]
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0
            return tuple(c0[k] + (c1[k] - c0[k]) * f for k in range(3)), a0 + (a1 - a0) * f
    return (0.1, 0.1, 0.15), 0.5

# ==========================================
# POST-PROCESSING FBO
# ==========================================
class PostProcessor:
    def __init__(self, w, h):
        self.fbo = glGenFramebuffers(1)
        self.color_tex = glGenTextures(1)
        self.depth_rbo = glGenRenderbuffers(1)
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.shader = compileProgram(compileShader(POST_VERTEX_SHADER, GL_VERTEX_SHADER), compileShader(POST_FRAGMENT_SHADER, GL_FRAGMENT_SHADER))
        self.resize(w, h)
        self._setup_quad()

    def resize(self, w, h):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glBindTexture(GL_TEXTURE_2D, self.color_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0, GL_RGBA, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.color_tex, 0)
        
        glBindRenderbuffer(GL_RENDERBUFFER, self.depth_rbo)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, self.depth_rbo)

    def _setup_quad(self):
        glBindVertexArray(self.vao); glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        verts = np.array([-1,-1, 0,0,  1,-1, 1,0,  -1,1, 0,1,  1,1, 1,1], dtype=np.float32)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1); glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(2 * 4))
        glBindVertexArray(0)

    def bind(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)

    def render(self, exposure):
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glDisable(GL_DEPTH_TEST)
        glUseProgram(self.shader)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.color_tex)
        glUniform1i(glGetUniformLocation(self.shader, "uScene"), 0)
        glUniform1f(glGetUniformLocation(self.shader, "uExposure"), exposure)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)

# ==========================================
# ENGINE CORE
# ==========================================
class Particle:
    __slots__ = ('x','y','z','vx','vy','vz','life','color')
    def __init__(self, x, y, z, color):
        self.x = x + random.uniform(-0.2, 0.2); self.y = y + random.uniform(-0.2, 0.2); self.z = z + random.uniform(0, 0.5)
        self.vx = random.uniform(-2, 2); self.vy = random.uniform(-2, 2); self.vz = random.uniform(2, 5)
        self.life = 1.0; self.color = color

class Chunk:
    def __init__(self, cx, cy, world):
        self.cx = cx; self.cy = cy; self.world = world
        self.is_dirty = True; self.is_building = False
        
        self.solid_vao = glGenVertexArrays(1); self.solid_vbo = glGenBuffers(1); self.solid_ebo = glGenBuffers(1)
        self.solid_index_count = 0
        self.trans_vao = glGenVertexArrays(1); self.trans_vbo = glGenBuffers(1); self.trans_ebo = glGenBuffers(1)
        self.trans_index_count = 0
        
        self._solid_verts = None; self._solid_indices = None
        self._trans_verts = None; self._trans_indices = None

    def get_block(self, x, y, z):
        if 0 <= x < WORLD_SIZE_X and 0 <= y < WORLD_SIZE_Y and 0 <= z < WORLD_SIZE_Z: return self.world[x, y, z]
        return 0

    def _face_visible(self, neighbor_id, self_trans):
        if neighbor_id == 0: return True
        n_trans = BLOCK_TYPES.get(neighbor_id, ("",True,0,False))[1]
        if n_trans and not self_trans: return True
        return False

    def build_mesh_data(self):
        solid_verts = []; solid_indices = []
        trans_verts = []; trans_indices = []
        visited = {'top': set(), 'bottom': set(), 'right': set(), 'left': set(), 'front': set(), 'back': set()}
        sx, sy = self.cx * CHUNK_SIZE, self.cy * CHUNK_SIZE

        for lx in range(CHUNK_SIZE):
            for ly in range(CHUNK_SIZE):
                wx, wy = sx + lx, sy + ly
                for lz in range(WORLD_SIZE_Z):
                    x, y, z = wx, wy, lz
                    block_id = self.get_block(x, y, z)
                    if block_id == 0: continue
                    is_trans = BLOCK_TYPES[block_id][1]
                    target_verts = trans_verts if is_trans else solid_verts
                    target_idx = trans_indices if is_trans else solid_indices
                    target_vis = visited

                    # ---- TOP (+Z) ----
                    if (x, y, z) not in target_vis['top']:
                        above = self.get_block(x, y, z + 1)
                        if self._face_visible(above, is_trans):
                            w, h = 1, 1
                            while lx + w < CHUNK_SIZE and self.get_block(x + w, y, z) == block_id:
                                if not self._face_visible(self.get_block(x + w, y, z + 1), is_trans): break
                                target_vis['top'].add((x + w, y, z)); w += 1
                            valid = True
                            while valid and ly + h < CHUNK_SIZE:
                                for i in range(w):
                                    if self.get_block(x + i, y + h, z) != block_id: valid = False; break
                                    if not self._face_visible(self.get_block(x + i, y + h, z + 1), is_trans): valid = False; break
                                if valid:
                                    for i in range(w): target_vis['top'].add((x + i, y + h, z))
                                    h += 1
                            target_vis['top'].add((x, y, z))
                            tex_id = get_texture_id(block_id, 0)
                            u_offset = tex_id / NUM_TEXTURES
                            u0 = u_offset + 0.001; u1 = u_offset + w * (1.0 / NUM_TEXTURES) - 0.001
                            v0 = 0.001; v1 = h * 1.0 - 0.001
                            idx_base = len(target_verts) // 8
                            target_verts.extend([
                                x, y, z+1, u0, v0, 0, 0, 1,       x+w, y, z+1, u1, v0, 0, 0, 1,
                                x+w, y+h, z+1, u1, v1, 0, 0, 1,   x, y+h, z+1, u0, v1, 0, 0, 1
                            ])
                            target_idx.extend([idx_base, idx_base+1, idx_base+2, idx_base+2, idx_base+3, idx_base])

                    # ---- BOTTOM (-Z) ----
                    if (x, y, z) not in target_vis['bottom']:
                        below = self.get_block(x, y, z - 1)
                        if self._face_visible(below, is_trans):
                            w, h = 1, 1
                            while lx + w < CHUNK_SIZE and self.get_block(x + w, y, z) == block_id:
                                if not self._face_visible(self.get_block(x + w, y, z - 1), is_trans): break
                                target_vis['bottom'].add((x + w, y, z)); w += 1
                            valid = True
                            while valid and ly + h < CHUNK_SIZE:
                                for i in range(w):
                                    if self.get_block(x + i, y + h, z) != block_id: valid = False; break
                                    if not self._face_visible(self.get_block(x + i, y + h, z - 1), is_trans): valid = False; break
                                if valid:
                                    for i in range(w): target_vis['bottom'].add((x + i, y + h, z))
                                    h += 1
                            target_vis['bottom'].add((x, y, z))
                            tex_id = get_texture_id(block_id, 1)
                            u_offset = tex_id / NUM_TEXTURES
                            u0 = u_offset + 0.001; u1 = u_offset + w * (1.0 / NUM_TEXTURES) - 0.001
                            v0 = 0.001; v1 = h * 1.0 - 0.001
                            idx_base = len(target_verts) // 8
                            target_verts.extend([
                                x, y, z, u0, v0, 0, 0, -1,       x+w, y+h, z, u1, v1, 0, 0, -1,
                                x+w, y, z, u1, v0, 0, 0, -1,     x, y+h, z, u0, v1, 0, 0, -1
                            ])
                            target_idx.extend([idx_base, idx_base+1, idx_base+2, idx_base+2, idx_base+3, idx_base])

                    # ---- RIGHT (+X) ----
                    if (x, y, z) not in target_vis['right']:
                        neighbor = self.get_block(x + 1, y, z)
                        if self._face_visible(neighbor, is_trans):
                            w, h = 1, 1
                            while ly + w < CHUNK_SIZE and self.get_block(x, y + w, z) == block_id:
                                if not self._face_visible(self.get_block(x + 1, y + w, z), is_trans): break
                                target_vis['right'].add((x, y + w, z)); w += 1
                            valid = True
                            while valid and lz + h < WORLD_SIZE_Z:
                                for i in range(w):
                                    if self.get_block(x, y + i, z + h) != block_id: valid = False; break
                                    if not self._face_visible(self.get_block(x + 1, y + i, z + h), is_trans): valid = False; break
                                if valid:
                                    for i in range(w): target_vis['right'].add((x, y + i, z + h))
                                    h += 1
                            target_vis['right'].add((x, y, z))
                            tex_id = get_texture_id(block_id, 2)
                            u_offset = tex_id / NUM_TEXTURES
                            u0 = u_offset + 0.001; u1 = u_offset + w * (1.0 / NUM_TEXTURES) - 0.001
                            v0 = 0.001; v1 = h * 1.0 - 0.001
                            idx_base = len(target_verts) // 8
                            target_verts.extend([
                                x+1, y, z, u0, v0, 1, 0, 0,       x+1, y+w, z, u1, v0, 1, 0, 0,
                                x+1, y+w, z+h, u1, v1, 1, 0, 0,   x+1, y, z+h, u0, v1, 1, 0, 0
                            ])
                            target_idx.extend([idx_base, idx_base+1, idx_base+2, idx_base+2, idx_base+3, idx_base])

                    # ---- LEFT (-X) ----
                    if (x, y, z) not in target_vis['left']:
                        neighbor = self.get_block(x - 1, y, z)
                        if self._face_visible(neighbor, is_trans):
                            w, h = 1, 1
                            while ly + w < CHUNK_SIZE and self.get_block(x, y + w, z) == block_id:
                                if not self._face_visible(self.get_block(x - 1, y + w, z), is_trans): break
                                target_vis['left'].add((x, y + w, z)); w += 1
                            valid = True
                            while valid and lz + h < WORLD_SIZE_Z:
                                for i in range(w):
                                    if self.get_block(x, y + i, z + h) != block_id: valid = False; break
                                    if not self._face_visible(self.get_block(x - 1, y + i, z + h), is_trans): valid = False; break
                                if valid:
                                    for i in range(w): target_vis['left'].add((x, y + i, z + h))
                                    h += 1
                            target_vis['left'].add((x, y, z))
                            tex_id = get_texture_id(block_id, 3)
                            u_offset = tex_id / NUM_TEXTURES
                            u0 = u_offset + 0.001; u1 = u_offset + w * (1.0 / NUM_TEXTURES) - 0.001
                            v0 = 0.001; v1 = h * 1.0 - 0.001
                            idx_base = len(target_verts) // 8
                            target_verts.extend([
                                x, y, z, u0, v0, -1, 0, 0,       x, y+w, z+h, u1, v1, -1, 0, 0,
                                x, y+w, z, u1, v0, -1, 0, 0,     x, y, z+h, u0, v1, -1, 0, 0
                            ])
                            target_idx.extend([idx_base, idx_base+1, idx_base+2, idx_base+2, idx_base+3, idx_base])

                    # ---- FRONT (+Y) ----
                    if (x, y, z) not in target_vis['front']:
                        neighbor = self.get_block(x, y + 1, z)
                        if self._face_visible(neighbor, is_trans):
                            w, h = 1, 1
                            while lx + w < CHUNK_SIZE and self.get_block(x + w, y, z) == block_id:
                                if not self._face_visible(self.get_block(x + w, y + 1, z), is_trans): break
                                target_vis['front'].add((x + w, y, z)); w += 1
                            valid = True
                            while valid and lz + h < WORLD_SIZE_Z:
                                for i in range(w):
                                    if self.get_block(x + i, y, z + h) != block_id: valid = False; break
                                    if not self._face_visible(self.get_block(x + i, y + 1, z + h), is_trans): valid = False; break
                                if valid:
                                    for i in range(w): target_vis['front'].add((x + i, y, z + h))
                                    h += 1
                            target_vis['front'].add((x, y, z))
                            tex_id = get_texture_id(block_id, 4)
                            u_offset = tex_id / NUM_TEXTURES
                            u0 = u_offset + 0.001; u1 = u_offset + w * (1.0 / NUM_TEXTURES) - 0.001
                            v0 = 0.001; v1 = h * 1.0 - 0.001
                            idx_base = len(target_verts) // 8
                            target_verts.extend([
                                x, y+1, z, u0, v0, 0, 1, 0,       x+w, y+1, z, u1, v0, 0, 1, 0,
                                x+w, y+1, z+h, u1, v1, 0, 1, 0,   x, y+1, z+h, u0, v1, 0, 1, 0
                            ])
                            target_idx.extend([idx_base, idx_base+1, idx_base+2, idx_base+2, idx_base+3, idx_base])

                    # ---- BACK (-Y) ----
                    if (x, y, z) not in target_vis['back']:
                        neighbor = self.get_block(x, y - 1, z)
                        if self._face_visible(neighbor, is_trans):
                            w, h = 1, 1
                            while lx + w < CHUNK_SIZE and self.get_block(x + w, y, z) == block_id:
                                if not self._face_visible(self.get_block(x + w, y - 1, z), is_trans): break
                                target_vis['back'].add((x + w, y, z)); w += 1
                            valid = True
                            while valid and lz + h < WORLD_SIZE_Z:
                                for i in range(w):
                                    if self.get_block(x + i, y, z + h) != block_id: valid = False; break
                                    if not self._face_visible(self.get_block(x + i, y - 1, z + h), is_trans): valid = False; break
                                if valid:
                                    for i in range(w): target_vis['back'].add((x + i, y, z + h))
                                    h += 1
                            target_vis['back'].add((x, y, z))
                            tex_id = get_texture_id(block_id, 5)
                            u_offset = tex_id / NUM_TEXTURES
                            u0 = u_offset + 0.001; u1 = u_offset + w * (1.0 / NUM_TEXTURES) - 0.001
                            v0 = 0.001; v1 = h * 1.0 - 0.001
                            idx_base = len(target_verts) // 8
                            target_verts.extend([
                                x, y, z, u0, v0, 0, -1, 0,       x+w, y, z+h, u1, v1, 0, -1, 0,
                                x+w, y, z, u1, v0, 0, -1, 0,     x, y, z+h, u0, v1, 0, -1, 0
                            ])
                            target_idx.extend([idx_base, idx_base+1, idx_base+2, idx_base+2, idx_base+3, idx_base])

        self._solid_verts = np.array(solid_verts, dtype=np.float32)
        self._solid_indices = np.array(solid_indices, dtype=np.uint32)
        self._trans_verts = np.array(trans_verts, dtype=np.float32)
        self._trans_indices = np.array(trans_indices, dtype=np.uint32)

    def upload_to_gpu(self):
        if self._solid_verts is not None:
            self._upload_mesh(self.solid_vao, self.solid_vbo, self.solid_ebo, self._solid_verts, self._solid_indices)
            self.solid_index_count = len(self._solid_indices)
            self._solid_verts = None; self._solid_indices = None
        if self._trans_verts is not None:
            self._upload_mesh(self.trans_vao, self.trans_vbo, self.trans_ebo, self._trans_verts, self._trans_indices)
            self.trans_index_count = len(self._trans_indices)
            self._trans_verts = None; self._trans_indices = None
        self.is_building = False

    def _upload_mesh(self, vao, vbo, ebo, verts, indices):
        if len(verts) == 0: return
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1); glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(2); glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(5 * 4))
        glBindVertexArray(0)

class VoxelEngine:
    def __init__(self):
        self.world = np.zeros((WORLD_SIZE_X, WORLD_SIZE_Y, WORLD_SIZE_Z), dtype=np.uint8)
        self.chunks = {}
        self.time_of_day = 0.4
        self.cam_x = WORLD_SIZE_X // 2; self.cam_y = WORLD_SIZE_Y // 2; self.cam_z = 10.0
        self.zoom = 16.0; self.yaw = 45.0; self.pitch = 35.264

        self.target_block = None; self.target_normal = (0, 0, 0); self.place_block = None
        self.selected_block = 1; self.particles = []; self.seed = 1337
        self.wireframe_mode = False; self.show_debug = True; self.show_help = False; self.time_paused = False

        self._load_textures()
        self.post_processor = PostProcessor(SCREEN_WIDTH, SCREEN_HEIGHT)
        
        self.shader = compileProgram(compileShader(VERTEX_SHADER, GL_VERTEX_SHADER), compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER))
        self.line_shader = compileProgram(compileShader(LINE_VERTEX_SHADER, GL_VERTEX_SHADER), compileShader(LINE_FRAGMENT_SHADER, GL_FRAGMENT_SHADER))
        self.ui_shader = compileProgram(compileShader(UI_VERTEX_SHADER, GL_VERTEX_SHADER), compileShader(UI_FRAGMENT_SHADER, GL_FRAGMENT_SHADER))

        self.ui_vao = glGenVertexArrays(1); self.ui_vbo = glGenBuffers(1); self.ui_texture = glGenTextures(1)
        self._setup_ui_quad()
        self.line_vao = glGenVertexArrays(1); self.line_vbo = glGenBuffers(1); self._setup_line_buffer()
        self.particle_vao = glGenVertexArrays(1); self.particle_vbo = glGenBuffers(1); self._setup_particle_buffer()

        self.build_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.builder_thread = threading.Thread(target=self._chunk_builder_worker, daemon=True)
        self.builder_thread.start()

    def _chunk_builder_worker(self):
        while True:
            chunk = self.build_queue.get()
            if chunk is None: break
            chunk.build_mesh_data()
            self.result_queue.put(chunk)

    def update_chunks(self):
        uploaded = 0
        while not self.result_queue.empty() and uploaded < 3:
            chunk = self.result_queue.get()
            chunk.upload_to_gpu()
            uploaded += 1
            
        for chunk in self.chunks.values():
            if chunk.is_dirty and not chunk.is_building:
                chunk.is_dirty = False
                chunk.is_building = True
                self.build_queue.put(chunk)

    def _load_textures(self):
        atlas_surf = create_texture_atlas()
        self.atlas_width = atlas_surf.get_width(); self.atlas_height = atlas_surf.get_height()
        tex_data = pygame.image.tostring(atlas_surf, "RGBA", True)
        self.atlas_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.atlas_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.atlas_width, self.atlas_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glGenerateMipmap(GL_TEXTURE_2D)

    def _setup_ui_quad(self):
        glBindVertexArray(self.ui_vao); glBindBuffer(GL_ARRAY_BUFFER, self.ui_vbo)
        verts = np.array([-1,-1, 0,0,  1,-1, 1,0,  -1,1, 0,1,  1,1, 1,1], dtype=np.float32)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1); glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(2 * 4))
        glBindVertexArray(0); glBindTexture(GL_TEXTURE_2D, self.ui_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    def _setup_line_buffer(self):
        glBindVertexArray(self.line_vao); glBindBuffer(GL_ARRAY_BUFFER, self.line_vbo)
        glBufferData(GL_ARRAY_BUFFER, 24 * 3 * 4, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0)); glBindVertexArray(0)

    def _setup_particle_buffer(self):
        glBindVertexArray(self.particle_vao); glBindBuffer(GL_ARRAY_BUFFER, self.particle_vbo)
        glBufferData(GL_ARRAY_BUFFER, 1000 * 4 * 8 * 4, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1); glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(2); glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(5 * 4))
        glBindVertexArray(0)

    def set_voxel(self, x, y, z, block_id):
        if 0 <= x < WORLD_SIZE_X and 0 <= y < WORLD_SIZE_Y and 0 <= z < WORLD_SIZE_Z:
            old_block = self.world[x, y, z]; self.world[x, y, z] = block_id
            cx, cy = x // CHUNK_SIZE, y // CHUNK_SIZE
            if (cx, cy) not in self.chunks: self.chunks[(cx, cy)] = Chunk(cx, cy, self.world)
            self.chunks[(cx, cy)].is_dirty = True
            lx, ly = x % CHUNK_SIZE, y % CHUNK_SIZE
            if lx == 0 and (cx-1, cy) in self.chunks: self.chunks[(cx-1, cy)].is_dirty = True
            if lx == CHUNK_SIZE-1 and (cx+1, cy) in self.chunks: self.chunks[(cx+1, cy)].is_dirty = True
            if ly == 0 and (cx, cy-1) in self.chunks: self.chunks[(cx, cy-1)].is_dirty = True
            if ly == CHUNK_SIZE-1 and (cx, cy+1) in self.chunks: self.chunks[(cx, cy+1)].is_dirty = True
            if old_block != 0 and block_id == 0:
                color = BLOCK_PARTICLE_COLORS.get(old_block, (128, 128, 128))
                for _ in range(10): self.particles.append(Particle(x, y, z, color))

    def generate_terrain(self):
        print("Compiling Numba kernels and generating world...")
        start = time.time()
        ca_grid = generate_ca_grid_gpu(WORLD_SIZE_X, WORLD_SIZE_Y, seed=99, fill_prob=0.3, iters=5)
        self.world.fill(0)
        generate_terrain_numba(self.world, ca_grid, seed=self.seed, water_level=4)
        generate_clouds_numba(self.world, seed=self.seed, cloud_z=34, threshold=0.65)
        self.chunks.clear()
        for cx in range(WORLD_SIZE_X // CHUNK_SIZE):
            for cy in range(WORLD_SIZE_Y // CHUNK_SIZE):
                self.chunks[(cx, cy)] = Chunk(cx, cy, self.world)
        print(f"World gen complete in {time.time() - start:.2f}s. Voxels: {np.count_nonzero(self.world)}")

    def save_world(self):
        try: np.save(SAVE_FILE, self.world); print(f"Saved to {SAVE_FILE}"); return True
        except Exception as e: print(f"Save failed: {e}"); return False

    def load_world(self):
        if not os.path.exists(SAVE_FILE): return False
        try:
            self.world = np.load(SAVE_FILE); self.chunks.clear()
            for cx in range(WORLD_SIZE_X // CHUNK_SIZE):
                for cy in range(WORLD_SIZE_Y // CHUNK_SIZE):
                    self.chunks[(cx, cy)] = Chunk(cx, cy, self.world)
            print(f"Loaded from {SAVE_FILE}"); return True
        except Exception as e: print(f"Load failed: {e}"); return False

    def get_view_matrix(self):
        view = mat_identity()
        view = mat_translate(-self.cam_x, -self.cam_y, -self.cam_z) @ view
        view = mat_rotate_x(math.radians(self.pitch)) @ view
        view = mat_rotate_y(math.radians(self.yaw)) @ view
        return view

    def get_proj_matrix(self):
        aspect = SCREEN_WIDTH / SCREEN_HEIGHT
        return mat_ortho(-self.zoom * aspect, self.zoom * aspect, -self.zoom, self.zoom, -1000.0, 1000.0)

    def get_mouse_target(self, mx, my):
        proj = self.get_proj_matrix(); view = self.get_view_matrix()
        near, far = unproject(mx, my, proj, view)
        direction = far - near
        n = np.linalg.norm(direction)
        if n < 1e-6: return None, (0,0,0), None
        direction /= n
        hit, normal = raycast_voxels(self.world, near, direction)
        if hit:
            place = (hit[0] + normal[0], hit[1] + normal[1], hit[2] + normal[2])
            return hit, normal, place
        return None, (0,0,0), None

    def update_particles(self, dt):
        for p in self.particles[:]:
            p.life -= dt * 2; p.z += p.vz * dt; p.x += p.vx * dt; p.y += p.vy * dt; p.vz -= 10 * dt
            if p.life <= 0: self.particles.remove(p)

    def render_particles(self, mvp, view):
        if not self.particles: return
        verts = []
        for p in self.particles:
            u_offset = 3 / NUM_TEXTURES
            u0 = u_offset + 0.001; u1 = u_offset + (1.0 / NUM_TEXTURES) - 0.001
            v0 = 0.001; v1 = 1.0 - 0.001
            s = 0.15
            verts.extend([
                p.x-s, p.y-s, p.z, u0, v0, 0, 0, 1, p.x+s, p.y-s, p.z, u1, v0, 0, 0, 1,
                p.x+s, p.y+s, p.z, u1, v1, 0, 0, 1, p.x-s, p.y+s, p.z, u0, v1, 0, 0, 1
            ])
            if len(verts) >= 1000 * 4 * 8: break
        arr = np.array(verts, dtype=np.float32)
        glBindVertexArray(self.particle_vao); glBindBuffer(GL_ARRAY_BUFFER, self.particle_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, arr.nbytes, arr)
        glUseProgram(self.shader)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "uMVP"), 1, GL_FALSE, mvp)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "uView"), 1, GL_FALSE, view)
        glDrawArrays(GL_TRIANGLES, 0, len(arr) // 8)
        glBindVertexArray(0)

    def render_highlight(self, mvp):
        if not self.target_block: return
        x, y, z = self.target_block
        e = 0.002
        x0, y0, z0 = x - e, y - e, z - e; x1, y1, z1 = x + 1 + e, y + 1 + e, z + 1 + e
        lines = [
            x0,y0,z0, x1,y0,z0, x1,y0,z0, x1,y1,z0, x1,y1,z0, x0,y1,z0, x0,y1,z0, x0,y0,z0,
            x0,y0,z1, x1,y0,z1, x1,y0,z1, x1,y1,z1, x1,y1,z1, x0,y1,z1, x0,y1,z1, x0,y0,z1,
            x0,y0,z0, x0,y0,z1, x1,y0,z0, x1,y0,z1, x1,y1,z0, x1,y1,z1, x0,y1,z0, x0,y1,z1,
        ]
        arr = np.array(lines, dtype=np.float32)
        glBindVertexArray(self.line_vao); glBindBuffer(GL_ARRAY_BUFFER, self.line_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, arr.nbytes, arr)
        glUseProgram(self.line_shader)
        glUniformMatrix4fv(glGetUniformLocation(self.line_shader, "uMVP"), 1, GL_FALSE, mvp)
        glUniform3f(glGetUniformLocation(self.line_shader, "uColor"), 0.0, 0.0, 0.0)
        glLineWidth(2.0); glDrawArrays(GL_LINES, 0, 24)
        
        if self.place_block:
            x, y, z = self.place_block
            x0, y0, z0 = x - e, y - e, z - e; x1, y1, z1 = x + 1 + e, y + 1 + e, z + 1 + e
            lines = [
                x0,y0,z0, x1,y0,z0, x1,y0,z0, x1,y1,z0, x1,y1,z0, x0,y1,z0, x0,y1,z0, x0,y0,z0,
                x0,y0,z1, x1,y0,z1, x1,y0,z1, x1,y1,z1, x1,y1,z1, x0,y1,z1, x0,y1,z1, x0,y0,z1,
                x0,y0,z0, x0,y0,z1, x1,y0,z0, x1,y0,z1, x1,y1,z0, x1,y1,z1, x0,y1,z0, x0,y1,z1,
            ]
            arr = np.array(lines, dtype=np.float32)
            glBufferSubData(GL_ARRAY_BUFFER, 0, arr.nbytes, arr)
            glUniform3f(glGetUniformLocation(self.line_shader, "uColor"), 1.0, 1.0, 1.0)
            glDrawArrays(GL_LINES, 0, 24)
        glBindVertexArray(0)

    def render(self):
        self.post_processor.bind()
        sky, ambient = get_sky_and_ambient(self.time_of_day)
        glClearColor(sky[0], sky[1], sky[2], 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glEnable(GL_DEPTH_TEST); glEnable(GL_CULL_FACE); glCullFace(GL_BACK)
        proj = self.get_proj_matrix(); view = self.get_view_matrix()
        mvp = proj @ view

        sun_angle = (self.time_of_day - 0.25) * 2 * math.pi
        sun_z = math.sin(sun_angle)
        sun_dir = np.array([math.cos(sun_angle) * 0.7, 0.3, max(sun_z, -0.2)], dtype=np.float32)
        sun_dir /= np.linalg.norm(sun_dir)
        sun_color = np.array([1.0, 1.0, 0.9], dtype=np.float32) * max(0, sun_z)
        moon_color = np.array([0.2, 0.2, 0.4], dtype=np.float32) * max(0, -sun_z)
        total_sun_color = sun_color + moon_color

        glUseProgram(self.shader)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "uMVP"), 1, GL_FALSE, mvp)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "uView"), 1, GL_FALSE, view)
        glUniform1f(glGetUniformLocation(self.shader, "uAmbient"), ambient)
        glUniform3f(glGetUniformLocation(self.shader, "uSunDir"), *sun_dir)
        glUniform3f(glGetUniformLocation(self.shader, "uSunColor"), *total_sun_color)
        glUniform3f(glGetUniformLocation(self.shader, "uFogColor"), sky[0], sky[1], sky[2])
        glUniform1f(glGetUniformLocation(self.shader, "uFogDensity"), 0.015 / self.zoom)
        
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.atlas_id)
        glUniform1i(glGetUniformLocation(self.shader, "uAtlas"), 0)

        if self.wireframe_mode: glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        frustum = get_frustum_planes(mvp)
        cam_pos = np.array([self.cam_x, self.cam_y, self.cam_z])
        visible_chunks = []
        
        for chunk in self.chunks.values():
            if is_chunk_in_frustum(frustum, chunk.cx, chunk.cy):
                visible_chunks.append(chunk)
                if not chunk.is_building and chunk.solid_index_count > 0:
                    glBindVertexArray(chunk.solid_vao)
                    glDrawElements(GL_TRIANGLES, chunk.solid_index_count, GL_UNSIGNED_INT, None)

        trans_chunks = [c for c in visible_chunks if not c.is_building and c.trans_index_count > 0]
        trans_chunks.sort(key=lambda c: -np.sum((np.array([c.cx*16+8, c.cy*16+8, 20]) - cam_pos)**2))

        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); glDepthMask(GL_FALSE)
        for chunk in trans_chunks:
            glBindVertexArray(chunk.trans_vao)
            glDrawElements(GL_TRIANGLES, chunk.trans_index_count, GL_UNSIGNED_INT, None)
        glDepthMask(GL_TRUE); glDisable(GL_BLEND)

        if self.wireframe_mode: glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        glDisable(GL_CULL_FACE)
        self.render_highlight(mvp)
        self.render_particles(mvp, view)

        glUseProgram(0); glBindVertexArray(0)
        self.post_processor.render(1.5)

    def draw_ui(self, surface, font, small_font, clock):
        sky, ambient = get_sky_and_ambient(self.time_of_day)
        
        # Dynamic Hotbar (shows 10 blocks at a time, sliding window)
        hotbar_blocks = list(BLOCK_TYPES.keys())
        start_idx = max(0, self.selected_block - 5)
        end_idx = min(len(hotbar_blocks), start_idx + 10)
        if end_idx - start_idx < 10: start_idx = max(0, end_idx - 10)
        visible_blocks = hotbar_blocks[start_idx:end_idx]
        
        start_x = SCREEN_WIDTH // 2 - len(visible_blocks) * 35
        for i, bid in enumerate(visible_blocks):
            rect = pygame.Rect(start_x + i*70, SCREEN_HEIGHT - 70, 60, 60)
            pygame.draw.rect(surface, (30, 30, 30), rect, border_radius=8)
            color = BLOCK_PARTICLE_COLORS.get(bid, (255,0,255))
            cx, cy = rect.centerx, rect.centery
            pts = [(cx, cy - 15), (cx + 15, cy - 7), (cx, cy), (cx - 15, cy - 7)]
            pygame.draw.polygon(surface, color, pts)
            pygame.draw.polygon(surface, (20,20,20), pts, 1)
            if self.selected_block == bid: pygame.draw.rect(surface, (255, 255, 255), rect, 3, border_radius=8)
            num = font.render(str(i+1 if i < 9 else 0), True, (255,255,255))
            surface.blit(num, (rect.x + 5, rect.y + 5))

        name = BLOCK_TYPES[self.selected_block][0]
        name_surf = font.render(name, True, (255, 255, 255))
        surface.blit(name_surf, (SCREEN_WIDTH // 2 - name_surf.get_width() // 2, SCREEN_HEIGHT - 95))

        cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        pygame.draw.line(surface, (255,255,255), (cx-8, cy), (cx+8, cy), 2)
        pygame.draw.line(surface, (255,255,255), (cx, cy-8), (cx, cy+8), 2)

        if self.show_debug:
            phase = "Night"
            if 0.20 <= self.time_of_day < 0.30: phase = "Dawn"
            elif 0.30 <= self.time_of_day < 0.70: phase = "Day"
            elif 0.70 <= self.time_of_day < 0.82: phase = "Dusk"
            stats = [
                f"FPS: {int(clock.get_fps())}", f"GPU CA: {'CuPy' if USE_GPU else 'NumPy'}",
                f"Renderer: OpenGL 3.3 (HDR FBO + Multithreading)", f"Voxels: {np.count_nonzero(self.world)}",
                f"Block Types: {len(BLOCK_TYPES)}", f"Particles: {len(self.particles)}", 
                f"Chunks: {len(self.chunks)}", f"Cam: ({self.cam_x:.1f}, {self.cam_y:.1f}, {self.cam_z:.1f})  Zoom: {self.zoom:.1f}",
                f"Yaw: {self.yaw:.1f}  Pitch: {self.pitch:.1f}", f"Time: {self.time_of_day:.2f} ({phase})  Ambient: {ambient:.2f}",
                f"Target: {self.target_block}",
            ]
            for i, text in enumerate(stats):
                surface.blit(small_font.render(text, True, (255, 255, 255)), (10, 10 + i * 20))

        if self.show_help:
            help_lines = [
                "Controls:", "WASD / Arrows : Pan camera", "Q / E : Rotate camera (yaw)",
                "Space / Shift : Move camera up / down", "Ctrl : Sprint", "Mouse Wheel : Zoom",
                "Left Click : Break block", "Right Click : Place block", "1-9, 0 : Select block in window",
                "[ / ] : Cycle selected block (scrolls hotbar)", "F : Toggle wireframe", "F3 : Toggle debug overlay",
                "F2 : Screenshot", "F5 / F9 : Save / Load world", "R : Regenerate world (new seed)",
                "T : Pause / resume time", "H : Toggle this help", "ESC : Quit",
            ]
            box_w, box_h = 380, len(help_lines) * 22 + 20
            box = pygame.Rect(SCREEN_WIDTH - box_w - 20, 20, box_w, box_h)
            s = pygame.Surface((box_w, box_h), pygame.SRCALPHA); s.fill((0, 0, 0, 180))
            surface.blit(s, box.topleft); pygame.draw.rect(surface, (255, 255, 255), box, 1, border_radius=6)
            for i, line in enumerate(help_lines):
                surface.blit(small_font.render(line, True, (255, 255, 255)), (box.x + 12, box.y + 10 + i * 22))

        hint = small_font.render("H = Help   F3 = Debug   F5/F9 = Save/Load   R = Regen   [ / ] = Change Block", True, (200, 200, 200))
        surface.blit(hint, (10, SCREEN_HEIGHT - 25))

def main():
    pygame.init()
    try:
        pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF, vsync=1)
    except TypeError:
        pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("OpenGL Voxel Engine - 35 Blocks & Biomes")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18, bold=True)
    small_font = pygame.font.SysFont("Arial", 14)

    engine = VoxelEngine()
    engine.generate_terrain()
    ui_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        mx, my = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.MOUSEWHEEL:
                engine.zoom -= event.y * 1.0
                engine.zoom = max(4.0, min(engine.zoom, 50.0))
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if engine.target_block:
                    if event.button == 1: engine.set_voxel(*engine.target_block, 0)
                    elif event.button == 3 and engine.place_block: engine.set_voxel(*engine.place_block, engine.selected_block)
            elif event.type == pygame.KEYDOWN:
                # Check for number keys 1-0 for hotbar selection
                digit = None
                if pygame.K_1 <= event.key <= pygame.K_9: digit = event.key - pygame.K_1
                elif event.key == pygame.K_0: digit = 9
                
                if digit is not None:
                    hotbar_blocks = list(BLOCK_TYPES.keys())
                    start_idx = max(0, engine.selected_block - 5)
                    end_idx = min(len(hotbar_blocks), start_idx + 10)
                    if end_idx - start_idx < 10: start_idx = max(0, end_idx - 10)
                    visible_blocks = hotbar_blocks[start_idx:end_idx]
                    if digit < len(visible_blocks):
                        engine.selected_block = visible_blocks[digit]
                elif event.key == pygame.K_LEFTBRACKET: 
                    engine.selected_block = max(1, engine.selected_block - 1)
                elif event.key == pygame.K_RIGHTBRACKET: 
                    engine.selected_block = min(len(BLOCK_TYPES), engine.selected_block + 1)
                elif event.key == pygame.K_t: engine.time_paused = not engine.time_paused
                elif event.key == pygame.K_f: engine.wireframe_mode = not engine.wireframe_mode
                elif event.key == pygame.K_F3: engine.show_debug = not engine.show_debug
                elif event.key == pygame.K_h: engine.show_help = not engine.show_help
                elif event.key == pygame.K_F2:
                    snap = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
                    pygame.image.save(snap, f"screenshot_{int(time.time())}.png")
                elif event.key == pygame.K_F5: engine.save_world()
                elif event.key == pygame.K_F9: engine.load_world()
                elif event.key == pygame.K_r:
                    engine.seed = random.randint(0, 99999)
                    engine.generate_terrain()
                elif event.key == pygame.K_ESCAPE: running = False

        keys = pygame.key.get_pressed()
        sprint = 2.5 if (keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]) else 1.0
        pan_speed = 25 * dt * sprint
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            engine.cam_x -= pan_speed * math.cos(math.radians(engine.yaw))
            engine.cam_y -= pan_speed * math.sin(math.radians(engine.yaw))
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            engine.cam_x += pan_speed * math.cos(math.radians(engine.yaw))
            engine.cam_y += pan_speed * math.sin(math.radians(engine.yaw))
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            engine.cam_x -= pan_speed * math.sin(math.radians(engine.yaw))
            engine.cam_y += pan_speed * math.cos(math.radians(engine.yaw))
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            engine.cam_x += pan_speed * math.sin(math.radians(engine.yaw))
            engine.cam_y -= pan_speed * math.cos(math.radians(engine.yaw))
        if keys[pygame.K_SPACE]: engine.cam_z += pan_speed
        if keys[pygame.K_LSHIFT]: engine.cam_z -= pan_speed
        if keys[pygame.K_q]: engine.yaw -= 60 * dt
        if keys[pygame.K_e]: engine.yaw += 60 * dt

        if not engine.time_paused:
            engine.time_of_day += dt * 0.03
            if engine.time_of_day > 1.0: engine.time_of_day = 0.0

        engine.target_block, engine.target_normal, engine.place_block = engine.get_mouse_target(mx, my)
        engine.update_particles(dt)
        engine.update_chunks()

        engine.render()

        ui_surface.fill((0, 0, 0, 0))
        engine.draw_ui(ui_surface, font, small_font, clock)

        tex_data = pygame.image.tostring(ui_surface, "RGBA", True)
        glBindTexture(GL_TEXTURE_2D, engine.ui_texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, tex_data)

        glDisable(GL_DEPTH_TEST); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glUseProgram(engine.ui_shader)
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, engine.ui_texture)
        glUniform1i(glGetUniformLocation(engine.ui_shader, "uTexture"), 0)
        glBindVertexArray(engine.ui_vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0); glDisable(GL_BLEND)

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()