import pygame
import sys
import random
import pickle
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import pygame.gfxdraw
from pygame import Surface
import math

# Constants
WIDTH, HEIGHT = 800, 600
GRID_SIZE = 10
CELL_SIZE = 30
MARGIN = 20
SHIP_SIZES = [5, 4, 3, 3, 2]
FPS = 60
MAX_ROUNDS = 3

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

# Add these new colors
NAVY_BLUE = (0, 48, 73)
LIGHT_BLUE = (134, 187, 216)
CREAM = (255, 246, 228)
ORANGE = (255, 155, 66)
DARK_RED = (204, 41, 54)
BOARD_COLOR = (238, 238, 238)

# Add these constants
ANIMATION_SPEED = 5
ROUND_DISPLAY_TIME = 2000  # 2 seconds

# Paths for saving data
DATA_FILE = "ship_placements.pkl"
MODEL_FILE = "rf_model.pkl"

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Battleship with Adaptive AI")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

def draw_text(text, x, y, color=BLACK):
    img = font.render(text, True, color)
    screen.blit(img, (x, y))

def draw_grid(x_offset, y_offset, board, show_ships=False):
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            rect = pygame.Rect(x_offset + col*CELL_SIZE, y_offset + row*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 1)
            cell = board[row][col]
            if cell == 1 and show_ships:
                pygame.draw.rect(screen, GRAY, rect)
            elif cell == 2:
                pygame.draw.rect(screen, RED, rect)
            elif cell == 3:
                pygame.draw.rect(screen, WHITE, rect)
            elif cell == 4:
                pygame.draw.rect(screen, BLUE, rect)

def random_ship_placement():
    board = [[0]*GRID_SIZE for _ in range(GRID_SIZE)]
    for ship_len in SHIP_SIZES:
        placed = False
        while not placed:
            orientation = random.choice(['H', 'V'])
            if orientation == 'H':
                row = random.randint(0, GRID_SIZE-1)
                col = random.randint(0, GRID_SIZE-ship_len)
                if all(board[row][col+i] == 0 for i in range(ship_len)):
                    for i in range(ship_len):
                        board[row][col+i] = 1
                    placed = True
            else:
                row = random.randint(0, GRID_SIZE-ship_len)
                col = random.randint(0, GRID_SIZE-1)
                if all(board[row+i][col] == 0 for i in range(ship_len)):
                    for i in range(ship_len):
                        board[row+i][col] = 1
                    placed = True
    return board

def is_valid_placement(board, ship_len, row, col, orientation):
    if orientation == 'H':
        if col + ship_len > GRID_SIZE:
            return False
        for i in range(ship_len):
            if board[row][col+i] != 0:
                return False
    else:
        if row + ship_len > GRID_SIZE:
            return False
        for i in range(ship_len):
            if board[row+i][col] != 0:
                return False
    return True

def place_ship(board, ship_len, row, col, orientation):
    if orientation == 'H':
        for i in range(ship_len):
            board[row][col+i] = 1
    else:
        for i in range(ship_len):
            board[row+i][col] = 1

def all_ships_sunk(board):
    for row in board:
        if 1 in row:
            return False
    return True

def get_adjacent_cells(row, col):
    adj = []
    for r in range(row-1, row+2):
        for c in range(col-1, col+2):
            if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                if (r != row or c != col):
                    adj.append((r,c))
    return adj

class ModernUI:
    def __init__(self):
        self.round_animation_alpha = 0
        self.show_round_screen = False
        self.round_start_time = 0
        self.fonts = {
            'large': pygame.font.Font(None, 74),
            'medium': pygame.font.Font(None, 48),
            'small': pygame.font.Font(None, 32)
        }

    def draw_rounded_rect(self, surface, color, rect, radius=15):
        """Draw a rounded rectangle"""
        pygame.gfxdraw.filled_circle(surface, rect.left + radius, rect.top + radius, radius, color)
        pygame.gfxdraw.filled_circle(surface, rect.right - radius - 1, rect.top + radius, radius, color)
        pygame.gfxdraw.filled_circle(surface, rect.left + radius, rect.bottom - radius - 1, radius, color)
        pygame.gfxdraw.filled_circle(surface, rect.right - radius - 1, rect.bottom - radius - 1, radius, color)
        
        pygame.draw.rect(surface, color, (rect.left + radius, rect.top, rect.width - 2*radius, rect.height))
        pygame.draw.rect(surface, color, (rect.left, rect.top + radius, rect.width, rect.height - 2*radius))

    def draw_board_background(self, surface, rect):
        """Draw a stylized board background"""
        self.draw_rounded_rect(surface, BOARD_COLOR, rect)
        # Add subtle grid lines
        for i in range(GRID_SIZE + 1):
            x = rect.left + i * CELL_SIZE
            y = rect.top + i * CELL_SIZE
            pygame.draw.line(surface, GRAY, (x, rect.top), (x, rect.bottom), 1)
            pygame.draw.line(surface, GRAY, (rect.left, y), (rect.right, y), 1)

    def draw_cell(self, surface, rect, cell_type, alpha=255):
        """Draw a styled cell"""
        if cell_type == 1:  # Ship
            color = GRAY
        elif cell_type == 2:  # Hit
            color = DARK_RED
        elif cell_type == 3:  # Miss
            color = LIGHT_BLUE
        elif cell_type == 4:  # Water
            color = NAVY_BLUE
        else:
            return

        s = Surface((CELL_SIZE, CELL_SIZE))
        s.fill(color)
        s.set_alpha(alpha)
        surface.blit(s, rect)

class BattleshipGame:
    def __init__(self):
        self.player_board = [[0]*GRID_SIZE for _ in range(GRID_SIZE)]
        self.ai_board = [[0]*GRID_SIZE for _ in range(GRID_SIZE)]
        self.player_guess_board = [[0]*GRID_SIZE for _ in range(GRID_SIZE)]
        self.ai_guess_board = [[0]*GRID_SIZE for _ in range(GRID_SIZE)]
        self.player_ships_to_place = SHIP_SIZES[:]
        self.placing_ship_index = 0
        self.placing_orientation = 'H'
        self.phase = 'round_start'  # changed initial phase to round_start for first round overlay
        self.message = "Place your ships"
        self.ai_hits = []
        self.ai_targets = []
        self.player_hits = []
        self.round = 1
        self.player_wins = 0
        self.ai_wins = 0
        self.load_data()
        self.model = self.load_model()
        self.player_ship_positions = []  # For ML training
        self.ai_ship_positions = []  # For ML training
        self.ui = ModernUI()
        self.round_start_time = pygame.time.get_ticks()  # track round start time for overlay

    def load_data(self):
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'rb') as f:
                self.ship_placement_data = pickle.load(f)
        else:
            self.ship_placement_data = []

    def save_data(self):
        with open(DATA_FILE, 'wb') as f:
            pickle.dump(self.ship_placement_data, f)

    def load_model(self):
        if os.path.exists(MODEL_FILE):
            with open(MODEL_FILE, 'rb') as f:
                model = pickle.load(f)
            return model
        else:
            return None

    def save_model(self):
        if self.model:
            with open(MODEL_FILE, 'wb') as f:
                pickle.dump(self.model, f)

    def reset_boards(self):
        self.player_board = [[0]*GRID_SIZE for _ in range(GRID_SIZE)]
        self.ai_board = random_ship_placement()
        self.player_guess_board = [[0]*GRID_SIZE for _ in range(GRID_SIZE)]
        self.ai_guess_board = [[0]*GRID_SIZE for _ in range(GRID_SIZE)]
        self.player_ships_to_place = SHIP_SIZES[:]
        self.placing_ship_index = 0
        self.placing_orientation = 'H'
        self.phase = 'round_start'  # changed phase to round_start to show overlay at round start
        self.message = "Place your ships"
        self.ai_hits = []
        self.ai_targets = []
        self.player_hits = []
        self.player_ship_positions = []
        self.ai_ship_positions = []
        self.round_start_time = pygame.time.get_ticks()  # reset round start time for overlay

    def handle_player_placement(self, pos):
        if self.phase != 'placing':
            return
        x, y = pos
        # Calculate centered positions for player board
        half_width = WIDTH // 2
        board_width = GRID_SIZE * CELL_SIZE
        board_height = GRID_SIZE * CELL_SIZE
        player_x = (half_width - board_width) // 2
        y_offset = (HEIGHT - board_height) // 2

        grid_x = (x - player_x) // CELL_SIZE
        grid_y = (y - y_offset) // CELL_SIZE
        if 0 <= grid_x < GRID_SIZE and 0 <= grid_y < GRID_SIZE:
            ship_len = self.player_ships_to_place[self.placing_ship_index]
            if is_valid_placement(self.player_board, ship_len, grid_y, grid_x, self.placing_orientation):
                place_ship(self.player_board, ship_len, grid_y, grid_x, self.placing_orientation)
                # Save ship positions for ML training
                positions = []
                if self.placing_orientation == 'H':
                    for i in range(ship_len):
                        positions.append((grid_y, grid_x + i))
                else:
                    for i in range(ship_len):
                        positions.append((grid_y + i, grid_x))
                self.player_ship_positions.append(positions)
                self.placing_ship_index += 1
                if self.placing_ship_index >= len(self.player_ships_to_place):
                    self.phase = 'player_turn'
                    self.message = "Your turn to guess"
            else:
                self.message = "Invalid placement"

    def player_guess(self, pos):
        if self.phase != 'player_turn':
            return
        x, y = pos
        # Calculate centered positions for AI board
        half_width = WIDTH // 2
        board_width = GRID_SIZE * CELL_SIZE
        board_height = GRID_SIZE * CELL_SIZE
        ai_x = half_width + (half_width - board_width) // 2
        y_offset = (HEIGHT - board_height) // 2

        grid_x = (x - ai_x) // CELL_SIZE
        grid_y = (y - y_offset) // CELL_SIZE
        if 0 <= grid_x < GRID_SIZE and 0 <= grid_y < GRID_SIZE:
            if self.player_guess_board[grid_y][grid_x] == 0:
                if self.ai_board[grid_y][grid_x] == 1:
                    self.player_guess_board[grid_y][grid_x] = 2  # hit
                    self.ai_board[grid_y][grid_x] = 2
                    self.player_hits.append((grid_y, grid_x))
                    self.message = "Hit!"
                else:
                    self.player_guess_board[grid_y][grid_x] = 3  # miss
                    self.ai_board[grid_y][grid_x] = 3
                    self.message = "Miss!"
                if all_ships_sunk(self.ai_board):
                    self.player_wins += 1
                    self.message = f"You won round {self.round}!"
                    self.phase = 'round_over'
                else:
                    self.phase = 'ai_turn'

    def ai_guess(self):
        if self.phase != 'ai_turn':
            return
        # AI guessing logic with ML heatmap
        guess = self.get_ai_guess()
        row, col = guess
        if self.player_board[row][col] == 1:
            self.ai_guess_board[row][col] = 2  # hit
            self.player_board[row][col] = 2
            self.ai_hits.append((row, col))
            self.ai_targets.extend(get_adjacent_cells(row, col))
            self.message = "AI hit your ship!"
        else:
            self.ai_guess_board[row][col] = 3  # miss
            self.player_board[row][col] = 3
            self.message = "AI missed!"
        if all_ships_sunk(self.player_board):
            self.ai_wins += 1
            self.message = f"AI won round {self.round}!"
            self.phase = 'round_over'
        else:
            self.phase = 'player_turn'

    def get_ai_guess(self):
        # If there are targets from previous hits, prioritize them
        while self.ai_targets:
            target = self.ai_targets.pop(0)
            r, c = target
            if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                if self.ai_guess_board[r][c] == 0:
                    return (r, c)
        # Otherwise, use ML model heatmap if available
        if self.model and len(self.ship_placement_data) >= 5:
            heatmap = self.generate_heatmap()
            # Choose the cell with highest probability that is not guessed yet
            candidates = [(r, c, heatmap[r][c]) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if self.ai_guess_board[r][c] == 0]
            if candidates:
                candidates.sort(key=lambda x: x[2], reverse=True)
                return (candidates[0][0], candidates[0][1])
        # Fallback: random guess
        while True:
            r = random.randint(0, GRID_SIZE-1)
            c = random.randint(0, GRID_SIZE-1)
            if self.ai_guess_board[r][c] == 0:
                return (r, c)

    def generate_heatmap(self):
        # Generate feature matrix for all cells
        X = []
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                features = self.extract_features(r, c)
                X.append(features)
        X = np.array(X)
        probs = self.model.predict_proba(X)[:,1]
        heatmap = probs.reshape((GRID_SIZE, GRID_SIZE))
        return heatmap

    def extract_features(self, r, c):
        # Features for ML model
        is_corner = 1 if (r == 0 or r == GRID_SIZE-1) and (c == 0 or c == GRID_SIZE-1) else 0
        dist_center = np.sqrt((r - GRID_SIZE/2)**2 + (c - GRID_SIZE/2)**2)
        # Count how many times this cell had a ship in past games
        count = 0
        for game in self.ship_placement_data:
            for ship in game:
                if (r, c) in ship:
                    count += 1
        return [is_corner, dist_center, count]

    def train_model(self):
        if len(self.ship_placement_data) < 5:
            return
        X = []
        y = []
        for game in self.ship_placement_data:
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    features = self.extract_features(r, c)
                    X.append(features)
                    # Label 1 if cell has ship in this game, else 0
                    label = 0
                    for ship in game:
                        if (r, c) in ship:
                            label = 1
                            break
                    y.append(label)
        X = np.array(X)
        y = np.array(y)
        clf = RandomForestClassifier(n_estimators=50)
        clf.fit(X, y)
        self.model = clf
        self.save_model()

    def end_round(self):
        # Save player ship positions for training
        self.ship_placement_data.append(self.player_ship_positions)
        self.save_data()
        self.train_model()
        self.round += 1
        if self.round > MAX_ROUNDS or self.player_wins == 2 or self.ai_wins == 2:
            self.phase = 'game_over'
            if self.player_wins > self.ai_wins:
                self.message = f"You won the game! {self.player_wins} - {self.ai_wins}"
            elif self.ai_wins > self.player_wins:
                self.message = f"AI won the game! {self.ai_wins} - {self.player_wins}"
            else:
                self.message = f"Game tied! {self.player_wins} - {self.ai_wins}"
        else:
            self.reset_boards()
            self.message = f"Round {self.round}: Place your ships"
            self.phase = 'round_start'  # changed to round_start to show overlay at new round
            self.round_start_time = pygame.time.get_ticks()

    def draw(self):
        screen.fill(NAVY_BLUE)

        # Calculate centered positions for boards in their halves
        half_width = WIDTH // 2
        board_width = GRID_SIZE * CELL_SIZE
        board_height = GRID_SIZE * CELL_SIZE
        player_x = (half_width - board_width) // 2
        ai_x = half_width + (half_width - board_width) // 2
        y_offset = (HEIGHT - board_height) // 2

        # Draw board backgrounds
        player_board_rect = pygame.Rect(player_x, y_offset, board_width, board_height)
        ai_board_rect = pygame.Rect(ai_x, y_offset, board_width, board_height)
        self.ui.draw_board_background(screen, player_board_rect)
        self.ui.draw_board_background(screen, ai_board_rect)

        # Draw boards with enhanced styling
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                # Player board
                rect = pygame.Rect(player_x + col*CELL_SIZE, y_offset + row*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                self.ui.draw_cell(screen, rect, self.player_board[row][col])
                
                # AI board
                rect = pygame.Rect(ai_x + col*CELL_SIZE, y_offset + row*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                self.ui.draw_cell(screen, rect, self.player_guess_board[row][col])

        # Draw round start overlay for 3 seconds
        if self.phase == 'round_start':
            current_time = pygame.time.get_ticks()
            elapsed = current_time - self.round_start_time
            if elapsed < 3000:
                overlay = pygame.Surface((WIDTH, HEIGHT))
                overlay.fill(BLACK)
                overlay.set_alpha(230)
                screen.blit(overlay, (0, 0))
                round_text = f"ROUND {self.round}"
                text_surface = self.ui.fonts['large'].render(round_text, True, CREAM)
                text_rect = text_surface.get_rect(center=(WIDTH//2, HEIGHT//2))
                screen.blit(text_surface, text_rect)
            else:
                # After 3 seconds, move to placing phase for all rounds
                self.phase = 'placing'
                self.message = "Place your ships"

        # Draw round transition screen for round_over or placing after round 1 removed to avoid conflict

        # Draw game info with enhanced styling
        title_font = self.ui.fonts['medium']
        info_font = self.ui.fonts['small']

        # Draw titles
        player_title = title_font.render("Your Board", True, CREAM)
        ai_title = title_font.render("Opponent's Board", True, CREAM)
        screen.blit(player_title, (player_x, y_offset - 40))
        screen.blit(ai_title, (ai_x, y_offset - 40))

        # Draw game status
        status_text = info_font.render(self.message, True, ORANGE)
        round_text = info_font.render(f"Round: {self.round}/{MAX_ROUNDS}", True, CREAM)
        score_text = info_font.render(f"Score: You {self.player_wins} - {self.ai_wins} AI", True, CREAM)

        screen.blit(status_text, (player_x, HEIGHT - 40))
        screen.blit(round_text, (WIDTH//2 - round_text.get_width()//2, HEIGHT - 70))
        screen.blit(score_text, (WIDTH - player_x - score_text.get_width(), HEIGHT - 40))

        # Draw ship placement info
        if self.phase == 'placing':
            ship_len = self.player_ships_to_place[self.placing_ship_index]
            placement_text = info_font.render(
                f"Place ship of length {ship_len} ({self.placing_orientation}) - Press R to rotate", 
                True, CREAM
            )
            screen.blit(placement_text, (player_x, HEIGHT - 100))

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if self.phase == 'placing' and event.key == pygame.K_r:
                        self.placing_orientation = 'V' if self.placing_orientation == 'H' else 'H'
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.phase == 'placing':
                        self.handle_player_placement(event.pos)
                    elif self.phase == 'player_turn':
                        self.player_guess(event.pos)
                    elif self.phase == 'round_over':
                        self.end_round()
                    elif self.phase == 'game_over':
                        # Restart game
                        self.round = 1
                        self.player_wins = 0
                        self.ai_wins = 0
                        self.reset_boards()
                        self.message = "Place your ships"
                        self.phase = 'round_start'  # changed to round_start for overlay on restart
                        self.round_start_time = pygame.time.get_ticks()
            if self.phase == 'ai_turn':
                pygame.time.wait(500)
                self.ai_guess()
            self.draw()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = BattleshipGame()
    game.run()

#This is the main code for the school project by Abhinav Raj of class 9th A of Air Force School Jalahalli
#Thanks a lot for reviewing
