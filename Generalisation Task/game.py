import pygame
import sys
import random
import math
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
import os

# Parse arguments
parser = argparse.ArgumentParser(description='Reaching Game')
parser.add_argument('--target_mode', type=str, default='dynamic', help='Mode for angular shift of target: random, fix, dynamic')
parser.add_argument('--mask_mode', action="store_true", help='Turn on mask mode')
parser.add_argument('--debug', action="store_true", help='Turn on debug mode')
parser.add_argument('-e', '--experiment_design', type=str, default='generalisation', help='Experiment design')
parser.add_argument('-n', '--name', type=str, default='test', help='Name of the experiment')
args = parser.parse_args()

# Game parameters
SCREEN_X, SCREEN_Y = 3456, 2234 # your screen resolution
WIDTH, HEIGHT = SCREEN_X // 1.5  , SCREEN_Y // 1.5 # be aware of monitor scaling on windows (150%)
CIRCLE_SIZE = 20
TARGET_SIZE = CIRCLE_SIZE
TARGET_RADIUS = 300
MASK_RADIUS = 0.66 * TARGET_RADIUS
ATTEMPTS_LIMIT = 200
START_POSITION = (WIDTH // 2, HEIGHT // 2)
START_ANGLE = 0
PERTURBATION_ANGLE= 30
TIME_LIMIT = 1000 # time limit in ms

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
black = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Initialize Pygame
pygame.init()

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("Reaching Game")


# Initialize game metrics
score = 0
attempts = 0
new_target = None
start_time = 0

new_target = None
start_target=math.radians(START_ANGLE)
move_faster = False 
clock = pygame.time.Clock()

# Initialize game modes
# experiment_design = "base"
mask_mode = args.mask_mode  # Turn on mask mode
target_mode = args.target_mode  # Mode for angular shift of target: random, fix, dynamic
perturbation_mode= False
perturbation_type= 'sudden' # Mode for angular shift of controll: random, gradual or sudden
perturbation_angle = math.radians(PERTURBATION_ANGLE)  # Angle between mouse_pos and circle_pos
perturbed_mouse_angle = 0
gradual_step = 0
gradual_attempts = 1
perturbation_rand=random.uniform(-math.pi/4, +math.pi/4)

error_angles = []  # List to store error angles
move_faster_events = [] # List to store move_faster events

# Flag for showing mouse position and deltas
show_mouse_info = False

# Function to generate a new target position
def generate_target_position(angle=None):
    if target_mode == 'random':
        angle = random.uniform(0, 2 * math.pi)

    elif target_mode == 'fix':   
        angle=start_target; 
        
    elif target_mode == 'dynamic':   
        angle=math.radians(angle); 

    new_target_x = WIDTH // 2 + TARGET_RADIUS * math.sin(angle)
    new_target_y = HEIGHT // 2 + TARGET_RADIUS * -math.cos(angle) # zero-angle at the top
    return [new_target_x, new_target_y]

# Function to check if the current target is reached
def check_target_reached():
    if new_target:
        distance = math.hypot(circle_pos[0] - new_target[0], circle_pos[1] - new_target[1])
        return distance <= CIRCLE_SIZE
    return False

# Function to check if player is at starting position and generate new target
def at_start_position_and_generate_target(mouse_pos):
    distance = math.hypot(mouse_pos[0] - START_POSITION[0], mouse_pos[1] - START_POSITION[1])
    if distance <= CIRCLE_SIZE:
        return True
    return False

# Function to make blocks based on conditions
def make_blocks(conditions, num_control, num_perturb):
    blocks = []
    i = 0
    for condition in conditions:
        if len(condition) == 2:
            blocks.append((i, i + num_control - 1))
            i += num_control
        else:
            blocks.append((i, i + num_perturb - 1))
            i += num_perturb
    return blocks

conditions = [(False, 0), (True, 'sudden', 30, 0), (False, 0), (True, 'sudden', 30, 0), (False, 0)]
basic_experment = {b: c for b, c in zip(make_blocks(conditions, 20, 60), conditions)}

conditions = [(False, 30), (True, 'sudden', 30, 30), (False, 30), 
              (False, 60), (True, 'sudden', 30, 60), (False, 60), 
              (False, 45), (True, 'sudden', 30, 45), (False, 45), 
              (False, 130), (True, 'sudden', 30, 130), (False, 130)]
generalisation_experiment = {b: c for b, c in zip(make_blocks(conditions, 20, 60), conditions)}

conditions = [(False, 0), (True, 'sudden', 30, 0), (False, 0), (True, 'sudden', 30, 0), (False, 0),
              (False, 0), (True, 'sudden', 30, 0), (False, 0), (True, 'sudden', 30, 0), (False, 0)]
finger_experiment = {b: c for b, c in zip(make_blocks(conditions, 20, 60), conditions)}

experiments = {"base": basic_experment, "generalisation": generalisation_experiment, "fingers": finger_experiment}

# Function to define perturbation mode and type on current attempt based on experiment design dictionary
def get_current_state(experiment_design, attempts):
    for (start, end), values in experiment_design.items():
        if start <= attempts <= end:
            if len(values) == 2:
                return False, None, 0, values[1], True
            else:
                return values[0], values[1], math.radians(values[2]), values[3], True
    return False, None, 0, 0, False

experiment = experiments[args.experiment_design]

# Main game loop
running = True
while running:
    screen.fill(BLACK)
             
    # Design experiment
    perturbation_mode, perturbation_type, perturbation_angle, target_angle, running = get_current_state(experiment, attempts)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:  # Press 'esc' to close the experiment
                running = False
            elif event.key == pygame.K_4: # Press '4' to test pertubation_mode
                perturbation_mode = True
            elif event.key == pygame.K_5: # Press '5' to end pertubation_mode
                perturbation_mode = False
            elif event.key == pygame.K_h:  # Press 'h' to toggle mouse info display
                show_mouse_info = not show_mouse_info
    
    # Hide the mouse cursor
    pygame.mouse.set_visible(False)
    # Get mouse position
    mouse_pos = pygame.mouse.get_pos()

    # Calculate distance from START_POSITION to mouse_pos
    deltax = mouse_pos[0] - START_POSITION[0]
    deltay = mouse_pos[1] - START_POSITION[1]
    distance = math.hypot(deltax, deltay)
    mouse_angle = math.atan2(deltay, deltax) 

    # TASK1: CALCULATE perturbed_mouse_pos
    # PRESS 'h' in game for a hint
    if perturbation_mode:
        if perturbation_type == 'sudden':
            #sudden clockwise perturbation of perturbation_angle
            perturbed_mouse_angle = mouse_angle + perturbation_angle

        elif perturbation_type == 'gradual':   
            #gradual counterclockwise perturbation of perturbation_angle in 10 steps, with perturbation_angle/10, each step lasts 3 attempts
            perturbed_mouse_angle = mouse_angle - perturbation_angle * min(gradual_attempts // 3, 10) / 10 
 
        perturbed_mouse_pos = [math.cos(perturbed_mouse_angle) * distance + START_POSITION[0], 
                               math.sin(perturbed_mouse_angle) * distance + START_POSITION[1]]

        circle_pos = perturbed_mouse_pos
    else:
        circle_pos = pygame.mouse.get_pos()
    
    # Check if target is hit or missed
    # hit if circle touches target's center
    if check_target_reached():
        score += 1
        # attempts += 1

        # CALCULATE AND SAVE ERRORS between target and circle end position for a hit
        error_angle = error_angle = math.atan2(circle_pos[1] - START_POSITION[0], circle_pos[0] - START_POSITION[1]) - math.atan2(new_target[1] - START_POSITION[0], new_target[0] - START_POSITION[1])
        error_angles.append(error_angle)
        move_faster_events.append(move_faster)

        new_target = None  # Set target to None to indicate hit
        start_time = 0  # Reset start_time after hitting the target
        # if perturbation_type == 'gradual' and perturbation_mode:   
        #     gradual_attempts += 1

    #miss if player leaves the target_radius + 1% tolerance
    elif new_target and math.hypot(circle_pos[0] - START_POSITION[0], circle_pos[1] - START_POSITION[1]) > TARGET_RADIUS*1.01:
        # attempts += 1

        # CALCULATE AND SAVE ERRORS between target and circle end position for a miss
        error_angle = math.atan2(circle_pos[1] - START_POSITION[0], circle_pos[0] - START_POSITION[1]) - math.atan2(new_target[1] - START_POSITION[0], new_target[0] - START_POSITION[1])
        error_angles.append(error_angle)
        move_faster_events.append(move_faster)

        new_target = None  # Set target to None to indicate miss
        start_time = 0  # Reset start_time after missing the target

        # if perturbation_type == 'gradual' and perturbation_mode:   
        #     gradual_attempts += 1

    # Check if player moved to the center and generate new target
    if not new_target and at_start_position_and_generate_target(mouse_pos):
        attempts += 1
        new_target = generate_target_position(target_angle)
        move_faster = False
        start_time = pygame.time.get_ticks()  # Start the timer for the attempt
        if perturbation_type == 'gradual' and perturbation_mode:   
            gradual_attempts += 1

    # Check if time limit for the attempt is reached
    current_time = pygame.time.get_ticks()
    if start_time != 0 and (current_time - start_time) > TIME_LIMIT:
        move_faster = True
        start_time = 0  # Reset start_time
        
    # Show 'MOVE FASTER!'
    if move_faster:
        font = pygame.font.Font(None, 36)
        text = font.render('MOVE FASTER!', True, black)
        text_rect = text.get_rect(center=(START_POSITION))
        screen.blit(text, text_rect)

# Generate playing field
    # Draw current target
    if new_target:
        pygame.draw.circle(screen, BLUE, new_target, TARGET_SIZE // 2)

    # Draw circle cursor
    if mask_mode:
        if distance < MASK_RADIUS:
            pygame.draw.circle(screen, WHITE, circle_pos, CIRCLE_SIZE // 2)
    else:
        pygame.draw.circle(screen, WHITE, circle_pos, CIRCLE_SIZE // 2)
    
    # Draw start position
    pygame.draw.circle(screen, WHITE, START_POSITION, 5)        

    # Show score
    font = pygame.font.Font(None, 36)
    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (10, 10))

    # Show attempts
    font = pygame.font.Font(None, 36)
    score_text = font.render(f"Attempts: {attempts}", True, WHITE)
    screen.blit(score_text, (10, 30))
    
    if args.experiment_design == 'fingers':
        font = pygame.font.Font(None, 45)
        if attempts < 25:
            finger_text = font.render(f"Use your Index finger Only!", True, black)
            screen.blit(finger_text, (WIDTH//2 - 200, 90))
        elif attempts == 25:
            finger_text = font.render(f"Use your Middle finger Now!", True, black)
            text_rect = finger_text.get_rect(center=(START_POSITION))
            screen.blit(finger_text, text_rect)
        else:
            finger_text = font.render(f"Use your Middle finger Only!", True, black)
            screen.blit(finger_text, (WIDTH//2 - 200, 90))
        

    if show_mouse_info:
        mouse_info_text = font.render(f"Mouse: x={mouse_pos[0]}, y={mouse_pos[1]}", True, WHITE)
        delta_info_text = font.render(f"Delta: Δx={deltax}, Δy={deltay}", True, WHITE)
        mouse_angle_text = font.render(f"Mouse_Ang: {np.rint(np.degrees(mouse_angle))}", True, WHITE)
        screen.blit(mouse_info_text, (10, 60))
        screen.blit(delta_info_text, (10, 90))
        screen.blit(mouse_angle_text, (10, 120))
        
    if args.debug and len(error_angles) > 0:
        font = pygame.font.Font(None, 36)
        error_text = font.render(f"Previous Error: {np.rint(np.degrees(error_angles[-1]))}", True, WHITE)
        screen.blit(error_text, (10, 240))
        
        if perturbation_mode:
            perturbation_text = font.render(f"Perturbation: {perturbation_type}", True, WHITE)
            screen.blit(perturbation_text, (10, 150))
            if perturbation_type == 'gradual':
                gradual_text = font.render(f"Gradual step: {gradual_step + 1} and gradual attempt: {gradual_attempts}", True, WHITE)
                angle_text = font.render(f"Perturbation angle: {np.rint(np.degrees((gradual_step + 1) * perturbation_angle / 10))}", True, WHITE)
                screen.blit(gradual_text, (10, 180))
                screen.blit(angle_text, (10, 210))
                
        target_angle_text = font.render(f"Target Angle: {target_angle}", True, WHITE)
        screen.blit(target_angle_text, (10, 270))

    # Update display
    pygame.display.flip()
    clock.tick(60)

# Quit Pygame
pygame.quit()

os.makedirs(args.experiment_design, exist_ok=True)
folder_path = os.path.join(args.experiment_design, args.name)
os.makedirs(folder_path, exist_ok=True)

# Save error angles and move_faster events to csv files
error_angles = np.degrees(np.array(error_angles))
move_faster_events = np.array(move_faster_events)

name_exp = args.name + "_" + args.experiment_design
np.savetxt(os.path.join(folder_path, "error_angles_" + name_exp +'.csv'), error_angles, delimiter=",")
np.savetxt(os.path.join(folder_path, "move_faster_" + name_exp +'.csv'), move_faster_events, delimiter=",")

# plt.show()

sys.exit()