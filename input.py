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

def increaseVal(row, col):
    grid[row][col] = 1
    for i in range(-1, 2):
        for j in range(-1, 2):
            if inGrid(row + i, col + j) and (i != 0 or j != 0):
                grid[row + i][col + j] = min(1, max(grid[row+i][col+j] + 0.04, grid[row+i][col+j])); 

def color(value):
    return (255*value, 255*value, 255*value)  # Return a tuple (R, G, B, A)

# Main loop
running = True
drawing = False
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            # Save the grid array when the window is closed
            np.savetxt('grid_array.txt', grid, fmt='%f')
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                # Save a custom array with the first element as 'q'
                custom_array = ['q'] 
                np.savetxt('grid_array.txt', custom_array, fmt='%s')
                running = False
            elif event.key == pygame.K_ESCAPE or event.key == pygame.K_RETURN:
                np.savetxt('grid_array.txt', grid, fmt='%f')
                running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        elif event.type == pygame.MOUSEMOTION and drawing:
            x, y = pygame.mouse.get_pos()
            row, col = y // BOX_SIZE, x // BOX_SIZE
            if inGrid(row, col):
                increaseVal(row, col)
                            


    # Draw grid
    for i in range(ROWS):
        for j in range(COLS):
            if grid[i, j] > 0:
                pygame.draw.rect(screen, color(grid[i, j]), (j * BOX_SIZE, i * BOX_SIZE, BOX_SIZE, BOX_SIZE))

    pygame.display.flip()

# Quit pygame
pygame.quit()
