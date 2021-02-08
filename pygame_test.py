import pygame
from pygame.locals import *

# pygame.init()
screen = pygame.display.set_mode((640, 240))


class Color:
    BLACK = (0, 0, 0)
    GRAY = (127, 127, 127)
    WHITE = (255, 255, 255)

    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)

    YELLOW = (255, 255, 0)
    CYAN = (0, 255, 255)
    MAGENTA = (255, 0, 255)


# The easiest way to decode many key
key_dict = {K_k: Color.BLACK, K_r: Color.RED, K_g: Color.GREEN, K_b: Color.BLUE,
            K_y: Color.YELLOW, K_c: Color.CYAN, K_m: Color.MAGENTA, K_w: Color.WHITE}

pygame.init()
background = Color.GRAY
caption = "Start"

running = True
while running:
    for event in pygame.event.get():
        # Quit
        if event.type == pygame.QUIT:
            running = False

        # If the keyboard is pressed
        if event.type == KEYDOWN:
            if event.key in key_dict:
                background = key_dict[event.key]
                caption = 'background color = ' + str(background)

        pygame.display.set_caption(caption)
        screen.fill(background)
        pygame.display.update()
pygame.quit()

# 1.9 Explore a simple ball game
