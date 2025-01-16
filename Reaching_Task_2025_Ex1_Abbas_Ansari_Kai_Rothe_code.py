import pygame
import sys
import random
import math
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

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
experiment_design = "base"
mask_mode= False
target_mode = 'fix'  # Mode for angular shift of target: random, fix, dynamic
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
def generate_target_position():
    if target_mode == 'random':
        angle = random.uniform(0, 2 * math.pi)

    elif target_mode == 'fix':   
        angle=start_target;  

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

# Main game loop
running = True
while running:
    screen.fill(BLACK)

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
            
    # Design experiment
    if experiment_design == "base":
        if attempts == 1:
            perturbation_mode = False
        elif attempts == 40:
            perturbation_mode = True
            perturbation_type = 'gradual' 
        elif attempts == 80:
            perturbation_mode = False
        elif attempts == 120:
            perturbation_mode = True    
            perturbation_type = 'sudden'         
        elif attempts == 160:
            perturbation_mode = False
        elif attempts >= ATTEMPTS_LIMIT:
            running = False      
    elif experiment_design == "transfer_learning":
        if attempts == 1:
            perturbation_mode = False
        elif attempts == 40:
            perturbation_mode = True
            perturbation_type = 'gradual'
            perturbation_angle = math.radians(30)
        elif attempts == 80:
            perturbation_mode = False
        elif attempts == 120:
            perturbation_mode = True    
            perturbation_type = 'gradual'
            perturbation_angle = math.radians(-30)         
        elif attempts == 160:
            perturbation_mode = False
        elif attempts == 200:
            perturbation_mode = True    
            perturbation_type = 'gradual'
            perturbation_angle = math.radians(30)
        elif attempts == 240:
            perturbation_mode = False
        elif attempts == 280:
            perturbation_mode = True    
            perturbation_type = 'gradual'
            perturbation_angle = math.radians(30)
        elif attempts >= 320:
            running = False    

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
        attempts += 1

        # CALCULATE AND SAVE ERRORS between target and circle end position for a hit
        error_angle = error_angle = math.atan2(circle_pos[1] - START_POSITION[0], circle_pos[0] - START_POSITION[1]) - math.atan2(new_target[1] - START_POSITION[0], new_target[0] - START_POSITION[1])
        error_angles.append(error_angle)
        move_faster_events.append(move_faster)

        new_target = None  # Set target to None to indicate hit
        start_time = 0  # Reset start_time after hitting the target
        if perturbation_type == 'gradual' and perturbation_mode:   
            gradual_attempts += 1

    #miss if player leaves the target_radius + 1% tolerance
    elif new_target and math.hypot(circle_pos[0] - START_POSITION[0], circle_pos[1] - START_POSITION[1]) > TARGET_RADIUS*1.01:
        attempts += 1

        # CALCULATE AND SAVE ERRORS between target and circle end position for a miss
        error_angle = math.atan2(circle_pos[1] - START_POSITION[0], circle_pos[0] - START_POSITION[1]) - math.atan2(new_target[1] - START_POSITION[0], new_target[0] - START_POSITION[1])
        error_angles.append(error_angle)
        move_faster_events.append(move_faster)

        new_target = None  # Set target to None to indicate miss
        start_time = 0  # Reset start_time after missing the target

        if perturbation_type == 'gradual' and perturbation_mode:   
            gradual_attempts += 1

    # Check if player moved to the center and generate new target
    if not new_target and at_start_position_and_generate_target(mouse_pos):
        new_target = generate_target_position()
        move_faster = False
        start_time = pygame.time.get_ticks()  # Start the timer for the attempt

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

    if show_mouse_info:
        mouse_info_text = font.render(f"Mouse: x={mouse_pos[0]}, y={mouse_pos[1]}", True, WHITE)
        delta_info_text = font.render(f"Delta: Δx={deltax}, Δy={deltay}", True, WHITE)
        mouse_angle_text = font.render(f"Mouse_Ang: {np.rint(np.degrees(mouse_angle))}", True, WHITE)
        screen.blit(mouse_info_text, (10, 60))
        screen.blit(delta_info_text, (10, 90))
        screen.blit(mouse_angle_text, (10, 120))

    # Update display
    pygame.display.flip()
    clock.tick(60)

# Quit Pygame
pygame.quit()

## TASK 2, CALCULATE, PLOT AND SAVE (e.g. export as .csv) ERRORS from error_angles

error_angles = np.degrees(np.array(error_angles))
move_faster_events = np.array(move_faster_events)
attempts = np.arange(1, len(error_angles) + 1)

if experiment_design == "base":
    motor_variabilities = []
    for i in [0, 40, 80, 120]:
        motor_variabilities.append(np.var(error_angles[i:i+40]))
    motor_variabilities = np.array(motor_variabilities)
    
    fig = plt.figure()
    plt.plot(attempts[~move_faster_events], error_angles[~move_faster_events], 'o', linestyle="--", color = "blue", label = "Fast Enough")
    plt.plot(attempts[move_faster_events], error_angles[move_faster_events], 'x', linestyle="", color = "red", label = "Too Slow")

    plt.axvline(x=40, color='black', linestyle='-')
    plt.text(41, plt.ylim()[1] + 1, 'gradual\npertubation', color='black', verticalalignment='top', horizontalalignment='left', weight='bold')
    plt.axvline(x=80, color='black', linestyle='-')
    plt.text(81, plt.ylim()[1] + 1, 'no\npertubation', color='black', verticalalignment='top', horizontalalignment='left', weight='bold')
    plt.axvline(x=120, color='black', linestyle='-')
    plt.text(121, plt.ylim()[1] + 1, 'sudden\npertubation', color='black', verticalalignment='top', horizontalalignment='left', weight='bold')
    plt.axvline(x=160, color='black', linestyle='-')
    plt.text(161, plt.ylim()[1] + 1, 'no\npertubation', color='black', verticalalignment='top', horizontalalignment='left', weight='bold')

    plt.ylim(plt.ylim()[0], plt.ylim()[1] + 1.5)
    plt.xlim(0, ATTEMPTS_LIMIT)
elif experiment_design == "transfer_learning":
    motor_variabilities = []
    for i in [0, 40, 80, 120, 160, 200, 240, 280]:
        motor_variabilities.append(np.var(error_angles[i:i+40]))
    motor_variabilities = np.array(motor_variabilities)

    fig = plt.figure(figsize=(15, 5))

    plt.plot(attempts[~move_faster_events], error_angles[~move_faster_events], 'o', linestyle="--", color = "blue", label = "Fast Enough")
    plt.plot(attempts[move_faster_events], error_angles[move_faster_events], 'x', linestyle="", color = "red", label = "Too Slow")

    plt.axvline(x=1, color='black', linestyle='-')
    plt.text(2, plt.ylim()[1] + 5, '1) R, 0°', color='black', verticalalignment='top', horizontalalignment='left', weight='bold')
    plt.axvline(x=40, color='black', linestyle='-')
    plt.text(41, plt.ylim()[1] + 5, '2) CONTROL:\nR, gradual +30°', color='black', verticalalignment='top', horizontalalignment='left', weight='bold')
    plt.axvline(x=80, color='black', linestyle='-')
    plt.text(81, plt.ylim()[1] + 5, '3) R, 0°\n', color='black', verticalalignment='top', horizontalalignment='left', weight='bold')
    plt.axvline(x=120, color='black', linestyle='-')
    plt.text(121, plt.ylim()[1] + 5, '4) UNLEARNING:\nR, gradual -30°', color='black', verticalalignment='top', horizontalalignment='left', weight='bold')
    plt.axvline(x=160, color='black', linestyle='-')
    plt.text(161, plt.ylim()[1] + 5, '5) L, 0°', color='black', verticalalignment='top', horizontalalignment='left', weight='bold')
    plt.axvline(x=200, color='black', linestyle='-')
    plt.text(201, plt.ylim()[1] + 5, '6) ADAPTATION:\nL, gradual +30°', color='black', verticalalignment='top', horizontalalignment='left', weight='bold')
    plt.axvline(x=240, color='black', linestyle='-')
    plt.text(241, plt.ylim()[1] + 5, '7) R, 0°', color='black', verticalalignment='top', horizontalalignment='left', weight='bold')
    plt.axvline(x=280, color='black', linestyle='-')
    plt.text(281, plt.ylim()[1] + 5, "8) TRANSFER?\nR, gradual +30°", color='black', verticalalignment='top', horizontalalignment='left', weight='bold')

    plt.ylim(plt.ylim()[0], plt.ylim()[1] + 5.5)
    plt.xlim(0, 320)
plt.grid()
plt.xlabel('Attempt')
plt.ylabel('Error Angle [°]')
plt.legend(title='Movement Speed', loc = "lower right")
plt.tight_layout()

date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
np.savetxt("error_angles_" + date_str +'.csv', error_angles, delimiter=",")
np.savetxt("move_faster_" + date_str +'.csv', move_faster_events, delimiter=",")
np.savetxt("motor_variabilities_" + date_str +'.csv', move_faster_events, delimiter=",")
plt.savefig('errors' + date_str +'.png')

plt.show()

sys.exit()
