import pygame
import math
import numpy as np
import argparse
import pickle as pkl
import os
import time

# Parse arguments
parser = argparse.ArgumentParser(description="The Bavarian Game")
parser.add_argument('--debug', action="store_true", help='Turn on debug mode')
parser.add_argument('--backward', action="store_true", help='Run the backward block structure')
parser.add_argument('--variable', action="store_true", help='Run the variable beer block structure')
parser.add_argument('-n', '--name', type=str, default='test', help='Name of the experiment')
parser.add_argument('-no_perturb', type=int, default=10, help='Number of non perturbed trials')
parser.add_argument('-perturb', type=int, default=30, help='Number of perturbed trials')
args = parser.parse_args()

# Initialize Pygame
pygame.init()

# Screen settings
# Full screen mode
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
SCREEN_WIDTH, SCREEN_HEIGHT = screen.get_size()
pygame.display.set_caption("Beer Pint Game")

# Constants
TABLE_WIDTH = SCREEN_WIDTH - 100
TABLE_HEIGHT = int(SCREEN_HEIGHT * 0.9)
FREE_ZONE_RADIUS = 110
START_POS = (SCREEN_WIDTH // 2, SCREEN_HEIGHT - FREE_ZONE_RADIUS - 10)

# Colors
WHITE = (255, 255, 255)
DARK_GREEN = (0, 100, 0)
LIGHT_GREEN = (144, 238, 144)
DARK_RED = (139, 0, 0)
LIGHT_RED = (255, 182, 193)
DARK_BROWN = (120, 66, 40)
LIGHT_BLUE = (173, 216, 230)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)
GREEN_LAMP = (0, 255, 0)
RED_LAMP = (255, 0, 0)

# Game settings
BASE_FRICTION = 0.99
MAX_FRICTION = 0.977
ZONE_WIDTH = int(TABLE_WIDTH * 0.95)
ZONE_HEIGHT = 150

# Rectangles and triangles
SCORING_RECT = pygame.Rect(
    (SCREEN_WIDTH - ZONE_WIDTH) // 2,
    int(TABLE_HEIGHT * 0.2),
    ZONE_WIDTH,
    ZONE_HEIGHT,
)

TABLE_RECT = pygame.Rect((SCREEN_WIDTH - TABLE_WIDTH) // 2, (SCREEN_HEIGHT - TABLE_HEIGHT) // 2, TABLE_WIDTH, TABLE_HEIGHT)
SCREEN_RECT = pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)

GREEN_TRIANGLE = [
    SCORING_RECT.topleft,
    SCORING_RECT.topright,
    SCORING_RECT.bottomleft
]
RED_TRIANGLE = [
    SCORING_RECT.bottomright,
    SCORING_RECT.bottomleft,
    SCORING_RECT.topright
]

# Game variables
pint_pos = list(START_POS)
pint_velocity = [0, 0]
pint_radius = 15
friction = BASE_FRICTION
launched = False
stopped = False
waiting_for_mouse = True
perturbation_active = False
score = 0
scores = []
feedback_mode = False
feedback_type = None
trajectory = []
perturbation_force=0
force_increment=0.2
end_pos = [0, 0]
current_block = 1
show_info=0
trial_positions = []
last_trajectory=[]
noise_mu, noise_std = 5, 3
fraction_beer_in_pint = 1.0
cur_noise = 0

# Font setup
font = pygame.font.SysFont(None, 36)

# BUILD FIELD
def draw_playfield(mask_pint=False):
    """Draw the game playfield."""
    screen.fill(WHITE)

    # Draw the table
    pygame.draw.rect(screen, DARK_BROWN, TABLE_RECT)

    # Draw free movement zone
    pygame.draw.circle(screen, LIGHT_BLUE, START_POS, FREE_ZONE_RADIUS)
    pygame.draw.circle(screen, BLACK, START_POS, FREE_ZONE_RADIUS, 3)

    # Draw scoring areas with precomputed gradients
    screen.blit(green_gradient, SCORING_RECT.topleft)
    screen.blit(red_gradient, SCORING_RECT.topleft)

    # Optionally mask the beer pint
    if not mask_pint:
        # Draw the beer pint with size of yellow based on fraction_beer_in_pint
        pygame.draw.circle(screen, WHITE, (int(pint_pos[0]), int(pint_pos[1])), pint_radius + 2)
        pygame.draw.circle(screen, YELLOW, (int(pint_pos[0]), int(pint_pos[1])), pint_radius*fraction_beer_in_pint)
        # pygame.draw.circle(screen, YELLOW, (int(pint_pos[0]), int(pint_pos[1])), pint_radius)
        # pygame.draw.circle(screen, WHITE, (int(pint_pos[0]), int(pint_pos[1])), pint_radius + 2, 2)


# Precompute gradient surfaces
def create_gradient_surface(points, start_color, end_color, reference_point):
    """Generate a gradient surface for a triangular region."""
    max_distance = max(math.dist(reference_point, p) for p in points)
    surface = pygame.Surface((SCORING_RECT.width, SCORING_RECT.height), pygame.SRCALPHA)

    for y in range(surface.get_height()):
        for x in range(surface.get_width()):
            world_x = SCORING_RECT.left + x
            world_y = SCORING_RECT.top + y
            if point_in_polygon((world_x, world_y), points):
                distance = math.dist((world_x, world_y), reference_point)
                factor = min(distance / max_distance, 1.0)
                color = interpolate_color(start_color, end_color, factor)
                surface.set_at((x, y), color + (255,))  # Add alpha
    return surface

def interpolate_color(start_color, end_color, factor):
    """Interpolate between two colors."""
    return tuple(int(start + (end - start) * factor) for start, end in zip(start_color, end_color))

# PINT_MOVEMENTS
def handle_mouse_input():
    """Handle mouse interactions with the pint."""
    global pint_pos, pint_velocity, launched, waiting_for_mouse, noise
    mouse_pos = pygame.mouse.get_pos()
    distance = math.dist(mouse_pos, START_POS)
    if waiting_for_mouse:    
        if distance <= pint_radius:  # Mouse touching the pint
            waiting_for_mouse = False
    elif distance <= FREE_ZONE_RADIUS:
        pint_pos[0], pint_pos[1] = mouse_pos
    else:
        pint_velocity = calculate_velocity(pint_pos, mouse_pos)
        if perturbation_active:
            apply_perturbation()
        launched = True

def calculate_velocity(start_pos, mouse_pos):
    global noise, cur_noise
    cur_noise = 0 if not noise else np.random.normal(noise[0], noise[1], 1).astype(float)[0]
    dx = mouse_pos[0] - start_pos[0]
    dy = mouse_pos[1] - start_pos[1]
    speed = math.sqrt(dx**2 + dy**2) / 10
    angle = math.atan2(dy, dx) + math.radians(cur_noise)
    return [speed * math.cos(angle), speed * math.sin(angle)]

def apply_friction():
    global pint_velocity, fraction_beer_in_pint, friction
    friction = BASE_FRICTION - (BASE_FRICTION - MAX_FRICTION) * fraction_beer_in_pint
    pint_velocity[0] *= friction
    pint_velocity[1] *= friction

def update_perturbation():
    """Adjust the perturbation force based on gradual or sudden mode."""
    global perturbation_force, trial_in_block
    
    if gradual_perturbation and perturbation_active:
        # Increment force every 3 trials (or however you want to adjust the frequency)
        if trial_in_block % 3 == 0 and trial_in_block != 0:
            perturbation_force += force_increment  # Increase perturbation force gradually after each set of 3 trials
            print(f"Gradual perturbation force updated to: {perturbation_force}")
    # Sudden perturbation: No updates needed (force remains constant)
    
def apply_perturbation():
    """Apply perturbation to the pint's movement."""
    if perturbation_active:
        pint_velocity[0] += perturbation_force # Add rightward force

# CHECK & SCORE
def check_stopped():
    global pint_pos, stopped, launched
    # print("pint_pos", pint_pos)
    if abs(pint_velocity[0]) < 0.1 and abs(pint_velocity[1]) < 0.1 and launched:
        stopped = True
        launched = False
    elif not SCREEN_RECT.collidepoint(*pint_pos):
        stopped = True
        launched = False
        

def point_in_polygon(point, polygon):
    """Check if a point is inside a polygon."""
    x, y = point
    n = len(polygon)
    inside = False
    px, py = polygon[0]
    for i in range(1, n + 1):
        sx, sy = polygon[i % n]
        if y > min(py, sy):
            if y <= max(py, sy):
                if x <= max(px, sx):
                    if py != sy:
                        xinters = (y - py) * (sx - px) / (sy - py) + px
                    if px == sx or x <= xinters:
                        inside = not inside
        px, py = sx, sy
    return inside

def calculate_score():
    """Calculate and update the score."""
    global pint_pos, stopped, end_pos, score, trial_counter, trial_positions
    if stopped:  # Only calculate score once per trial
        if point_in_polygon(pint_pos, GREEN_TRIANGLE):
            reference_point = SCORING_RECT.topleft
            distance = math.dist(pint_pos, reference_point)
            max_distance = max(math.dist(p, reference_point) for p in GREEN_TRIANGLE)
            score += calculate_edge_score(distance, max_distance)
        elif point_in_polygon(pint_pos, RED_TRIANGLE):
            reference_point = SCORING_RECT.bottomright
            distance = math.dist(pint_pos, reference_point)
            max_distance = max(math.dist(p, reference_point) for p in RED_TRIANGLE)
            score -= calculate_edge_score(distance, max_distance)
        elif not TABLE_RECT.collidepoint(*pint_pos):
            score -= 50  # Penalty for missing
            display_message("Too far!")
        
        # Append trial position and current block number
        trial_positions.append((pint_pos[0], pint_pos[1], current_block))
        scores.append(score)
        reset_pint()
        handle_trial_end()
        
        

def calculate_edge_score(distance, max_distance):
    """
    Calculate the score based on distance to the reference point.
    100 points for the closest edge, 10 points for the farthest edge.
    """
    normalized_distance = min(distance / max_distance, 1.0)  # Normalize to [0, 1]
    return int(100 - 90 * normalized_distance)  # Scale between 100 and 10

def display_message(text, flash_message=False, duration=1000):
    """Display a message on the screen, flash the message with colours if specified."""
    if flash_message:
        font = pygame.font.Font(None, 52)
        screen = pygame.display.get_surface()
        text_surface = font.render(text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2 - 100))
    
        colors = [DARK_RED, RED_LAMP]
        end_time = time.time() + duration / 1000
        while time.time() < end_time:
            for color in colors:
                text_surface = font.render(text, True, color)
                screen.blit(text_surface, text_rect)
                pygame.display.flip()
                pygame.time.delay(500)
    else:
        font = pygame.font.Font(None, 36)
        screen = pygame.display.get_surface()
        text_surface = font.render(text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
        screen.blit(text_surface, text_rect)
        pygame.display.flip()
        pygame.time.delay(duration)
        

def reset_pint():
    """Reset the pint to the starting position."""
    global pint_pos, end_pos, last_trajectory, pint_velocity, launched, stopped, waiting_for_mouse, trajectory
    trajectory.append(last_trajectory)
    pint_pos[:] = START_POS
    pint_velocity[:] = [0, 0]
    launched = False
    stopped = False
    waiting_for_mouse = True
    last_trajectory=[]


#TASK 1: IMPLEMENT FEEDBACK MODES

def draw_feedback():
    """Display feedback based on the feedback type."""
    global feedback_type, last_trajectory, trajectory
    if not last_trajectory and trajectory:
        if feedback_type == 'trajectory':
            pygame.draw.lines(screen, YELLOW, False, trajectory[-1], 2)
        elif feedback_type == 'endpos':
            end_pos = trajectory[-1][-1]
            pygame.draw.circle(screen, WHITE, (int(end_pos[0]), int(end_pos[1])), pint_radius + 2, 2)
        elif feedback_type == 'rl':
            end_pos = trajectory[-1][-1]
            color = GREEN_LAMP if point_in_polygon(end_pos, GREEN_TRIANGLE) else RED_LAMP
            pygame.draw.circle(screen, color, START_POS, FREE_ZONE_RADIUS+10, 10)

# Precompute gradient surfaces
green_gradient = create_gradient_surface(GREEN_TRIANGLE, DARK_GREEN, LIGHT_GREEN, SCORING_RECT.topleft)
red_gradient = create_gradient_surface(RED_TRIANGLE, DARK_RED, LIGHT_RED, SCORING_RECT.bottomright)


#Design Experiment
def setup_block(block_number):
    """Set up block parameters."""
    global perturbation_active, feedback_mode, feedback_type, perturbation_force, trial_in_block, gradual_perturbation, noise
    
    block = block_structure[block_number - 1]
    feedback_type = block['feedback'] if block['feedback'] else None
    feedback_mode = feedback_type is not None
    noise = block.get('noise', None)

    perturbation_active = block['perturbation']
    trial_in_block = 0

    # Apply global perturbation mode to set gradual or sudden
    if perturbation_active:
        if block['gradual']:  # Gradual perturbation
            gradual_perturbation = True
            perturbation_force = block.get('initial_force', 0)  # Use the initial force for gradual perturbation
        else:  # Sudden perturbation
            gradual_perturbation = False
            perturbation_force = block.get('sudden_force', 10.0)  # Use the sudden force for sudden perturbation

def handle_trial_end():
    """Handle end-of-trial events."""
    global trial_in_block, current_block, running

    trial_in_block += 1

    # Update perturbation force for gradual perturbation
    if perturbation_active and gradual_perturbation:
        update_perturbation()

    # Transition to the next block if trials in the current block are complete
    if trial_in_block >= block_structure[current_block - 1]['num_trials']:
        current_block += 1
        if current_block > len(block_structure):
            running = False  # End experiment
        else:
            if current_block != 1 and (current_block-1)% 3 == 0:
                # Show a message "Drinking beer... The pint just got lighter!" and decrease fraction_beer_in_pint   
                # The message must flash above the board for 2 seconds, implement a custom function for this
                display_message("Drinking beer... The pint just got lighter!", flash_message=True, duration=2000)
                global fraction_beer_in_pint
                fraction_beer_in_pint = 1 - 0.8*current_block/len(block_structure)
            
            setup_block(current_block)

# TASK1: Define the experiment blocks
if args.debug:
    args.no_perturb = 2
    args.perturb = 1

no_perturb_trials = args.no_perturb
perturb_trials = args.perturb
none_feedback_block = [
    #Normal visual feedback
    {'feedback': None, 'perturbation': False, 'gradual': False, 'num_trials': no_perturb_trials},  # 10 trials without perturbation
    {'feedback': None, 'perturbation': True, 'gradual': False, 'num_trials': perturb_trials, 'initial_force': 0.2, 'sudden_force': 2.0},  # 30 trials with gradual perturbation
    {'feedback': None, 'perturbation': False, 'gradual': False, 'num_trials': no_perturb_trials},  # 10 trials without perturbation
]

trajectory_feedback_block = [
    # ADD Trajectory feedback
    {'feedback': "trajectory", 'perturbation': False, 'gradual': False, 'num_trials': no_perturb_trials},  # 10 trials without perturbation
    {'feedback': "trajectory", 'perturbation': True, 'gradual': True, 'num_trials': perturb_trials, 'initial_force': 0.2, 'sudden_force': 2.0},  # 30 trials with gradual perturbation
    {'feedback': "trajectory", 'perturbation': False, 'gradual': False, 'num_trials': no_perturb_trials},  # 10 trials without perturbation
]

endpos_feedback_block = [
    # ADD End position feedback
    {'feedback': "endpos", 'perturbation': False, 'gradual': False, 'num_trials': no_perturb_trials},  # 10 trials without perturbation
    {'feedback': "endpos", 'perturbation': True, 'gradual': True, 'num_trials': perturb_trials, 'initial_force': 0.2, 'sudden_force': 2.0},  # 30 trials with gradual perturbation
    {'feedback': "endpos", 'perturbation': False, 'gradual': False, 'num_trials': no_perturb_trials},  # 10 trials without perturbation
]

rl_feedback_block = [
    # ADD RL feedback
    {'feedback': "rl", 'perturbation': False, 'gradual': False, 'num_trials': no_perturb_trials},  # 10 trials without perturbation
    {'feedback': "rl", 'perturbation': True, 'gradual': True, 'num_trials': perturb_trials, 'initial_force': 0.2, 'sudden_force': 2.0},  # 30 trials with gradual perturbation
    {'feedback': "rl", 'perturbation': False, 'gradual': False, 'num_trials': no_perturb_trials},  # 10 trials without perturbation
]

forward_block_structure = none_feedback_block + trajectory_feedback_block + endpos_feedback_block + rl_feedback_block
reverse_block_structure = rl_feedback_block + endpos_feedback_block + trajectory_feedback_block + none_feedback_block
debug_block_structure = none_feedback_block + none_feedback_block + none_feedback_block + none_feedback_block
noises = [None, (1, 2), (3,4), (5,6), None]
endpos_noise_block_structure = []
for noise in noises:
    endpos_noise_block_structure += [
        {'feedback': "endpos", 'perturbation': False, 'gradual': False, 'num_trials': no_perturb_trials, 'noise': noise},  # 10 trials without perturbation
        {'feedback': "endpos", 'perturbation': True, 'gradual': False, 'num_trials': perturb_trials, 'initial_force': 0.2, 'sudden_force': 2.0, 'noise': noise},  # 30 trials with sudden perturbation
        {'feedback': "endpos", 'perturbation': False, 'gradual': False, 'num_trials': no_perturb_trials, 'noise': noise},  # 10 trials without perturbation
    ]
block_structure = forward_block_structure if not args.backward else reverse_block_structure
block_structure = endpos_noise_block_structure if args.variable else block_structure
block_structure = debug_block_structure if args.debug and not args.variable else block_structure
current_block = 1
setup_block(current_block)

# Main game loop
clock = pygame.time.Clock()
running = True
while running:
# Determine if the beer pint should be masked
    mask_pint = launched and feedback_mode and feedback_type in ('trajectory', 'rl', 'endpos')

    # Draw playfield with optional masking
    draw_playfield(mask_pint=mask_pint)

    # Display score (only for feedbacks where score is not relevant)
    if feedback_type not in ('rl', 'endpos', 'trajectory'):
        score_text = font.render(f"Score: {score}", True, BLACK)
        screen.blit(score_text, (10, 10))

    # Handle Keyboard events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_4:
                perturbation_mode = True
            elif event.key == pygame.K_5:
                perturbation_mode = False
            elif event.key == pygame.K_1:
                feedback_type = 'trajectory'
                feedback_mode = True
            elif event.key == pygame.K_2:
                feedback_type = 'endpos'
                feedback_mode = True
            elif event.key == pygame.K_3:
                feedback_type = 'rl'
                feedback_mode = True
            elif event.key == pygame.K_0:
                feedback_mode = False
            elif event.key == pygame.K_i:  # Press 'i' to toggle info display
                show_info = not show_info    
            elif event.key == pygame.K_SPACE:  # Start the next experimental block
                current_block += 1
                if current_block > len(block_structure):
                    running = False  # End the experiment
                else:
                    setup_block(current_block)
    if not launched:
        handle_mouse_input()
    else:
        pint_pos[0] += pint_velocity[0]
        pint_pos[1] += pint_velocity[1]
        last_trajectory.append((int(pint_pos[0]), int(pint_pos[1])))
        apply_friction()
        check_stopped()
        calculate_score()

    # Draw feedback if applicable
    draw_feedback()

    if show_info:
        fb_info_text = font.render(f"Feedback: {feedback_type}", True, BLACK)
        pt_info_text = font.render(f"Perturbation:{perturbation_active}", True, BLACK)
        pf_info_text = font.render(f"Perturbation_force:{perturbation_force}", True, BLACK)
        tib_text = font.render(f"Trial_in_block: {trial_in_block}", True, BLACK)
        beer_text = font.render(f"Beer: {fraction_beer_in_pint:.2f}", True, BLACK)
        friction_text = font.render(f"Friction: {friction:.4f}", True, BLACK)
        noise_text = font.render(f"Noise: {0 if not cur_noise else cur_noise:.2f}", True, BLACK)
        screen.blit(fb_info_text, (10, 60))
        screen.blit(pt_info_text, (10, 90))
        screen.blit(pf_info_text, (10, 120))
        screen.blit(tib_text, (10, 150))    
        screen.blit(beer_text, (10, 180))
        screen.blit(friction_text, (10, 210))
        screen.blit(noise_text, (10, 240))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()

folder_path = os.path.join("forward" if not args.backward else "backward" , args.name)
os.makedirs(folder_path, exist_ok=True)
np.savetxt(os.path.join(folder_path, "trial_positions.csv"), np.array(trial_positions), delimiter=",")
np.savetxt(os.path.join(folder_path, "scores.csv"), np.array(scores), delimiter=",")
np.savetxt(os.path.join(folder_path, "geometry_specs.csv"), np.array([SCREEN_WIDTH, SCREEN_HEIGHT]), delimiter=",")
pkl.dump(trajectory, open(os.path.join(folder_path, "trajectories.pkl"), "wb"))