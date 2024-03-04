import math
import sys, os

# Disable print
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Enable print
def enablePrint():
    sys.stdout = sys.__stdout__


blockPrint()
import pygame
enablePrint()

import numpy as np

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 560, 560
NUM_BOX = 28
BOX_SIZE = WIDTH // NUM_BOX
ROWS, COLS = HEIGHT // BOX_SIZE, WIDTH // BOX_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Create a window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
screen.fill(BLACK)
pygame.display.set_caption("press q to quit program")

# Initialize grid
grid = np.zeros((ROWS, COLS))

def inGrid(row, col):
    return 0 <= row < ROWS and 0 <= col < COLS

def draw(x, y, radius):
    row, col = y // BOX_SIZE, x // BOX_SIZE
    if inGrid(row, col):
        grid[row][col] = min(1, max(grid[row][col] + 0.3, grid[row][col]))
        for i in range(col - 1, col+2):
            for j in range(row - 1, row +2):
                if distance((i+0.5)*BOX_SIZE, (j+0.5)*BOX_SIZE, x,y) <= radius:
                    if inGrid(j, i):
                        grid[j][i] = min(1, max(grid[j][i] + 0.08, grid[j][i]))
                    
def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def color(value):
    value = value
    return (255*value, 255*value, 255*value)  # Return a tuple (R, G, B, A)

# Main loop
running = True
drawing = False
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            # Save the grid array when the window is closed
            np.savetxt('output/grid_array.txt', grid, fmt='%f')
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                # Save a custom array with the first element as 'q'
                custom_array = ['q'] 
                np.savetxt('output/grid_array.txt', custom_array, fmt='%s')
                running = False
            elif event.key == pygame.K_ESCAPE or event.key == pygame.K_RETURN:
                np.savetxt('output/grid_array.txt', grid, fmt='%f')
                running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        elif drawing:
            x, y = pygame.mouse.get_pos()
            draw(x, y, BOX_SIZE + 10)
                            


    # Draw grid
    for i in range(ROWS):
        for j in range(COLS):
            if grid[i, j] > 0:
                pygame.draw.rect(screen, color(grid[i, j]), (j * BOX_SIZE, i * BOX_SIZE, BOX_SIZE, BOX_SIZE))

    pygame.display.flip()

# Quit pygame
pygame.quit()
