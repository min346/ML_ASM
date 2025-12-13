import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pickle
import time
from matplotlib.patches import Rectangle
from typing import Optional
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional


# ============================================================
# 1) ENVIRONMENT (grid-based)
#    If you already have ParkingGrid + env_builders in your project,
#    you can delete this section and import them instead.
# ============================================================

State = Tuple[int, int]  # (row, col)

@dataclass
class StepInfo:
    is_collision: bool = False
    is_boundary: bool = False
    is_parked: bool = False

class ParkingGrid:
    # default environment
    def __init__(self,
                 size=10,
                 start=(0,0),
                 parking_spots=[(9,9)],
                 obstacles=None,
                 moving_obstacles=None,
                 move_penalty=-2,
                 collision_penalty=-50,
                 park_reward=200,
                 boundary_penalty=-20,
                 reward_shaping=True,
                 shaping_coeff=0.1,      # ↓ reduced (IMPORTANT)
                 slip_prob=0.0):
        
        self.size = size
        self.start = start
        
        # convert to sets for faster lookup
        self.parking_spots = set(parking_spots)
        self.static_obstacles = set(obstacles) if obstacles else set()
        self.moving_obstacles = set(moving_obstacles) if moving_obstacles else set()
        self.obstacles = self.static_obstacles | self.moving_obstacles

        # rewards
        self.move_penalty = move_penalty
        self.collision_penalty = collision_penalty
        self.park_reward = park_reward
        self.boundary_penalty = boundary_penalty

        # shaping & dynamics
        self.reward_shaping = reward_shaping
        self.shaping_coeff = shaping_coeff
        self.slip_prob = slip_prob

        self.reset()

    def reset(self):
        # always use fixed start position
        self.state = self.start
        self.steps_taken = 0                 # track steps
        self.prev_action = None              # NEW: for turn penalty
        self.visit_count = {}                # NEW: for revisit penalty
        return self.state

    # check inside grid
    def _in_bounds(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    # used for reward shaping (distance to nearest parking spot)
    def _nearest_goal_distance(self, pos):
        dists = [abs(pos[0] - g[0]) + abs(pos[1] - g[1]) for g in self.parking_spots]
        return min(dists) if dists else 0

    # simple left-moving obstacle behavior
    def _update_moving_obstacles(self):
        if not self.moving_obstacles:
            return

        new_positions = set()
        for (x, y) in self.moving_obstacles:
            nx, ny = x, y - 1
            if not self._in_bounds(nx, ny) or (nx, ny) in self.static_obstacles:
                nx, ny = x, self.size - 1
            new_positions.add((nx, ny))

        self.moving_obstacles = new_positions
        self.obstacles = self.static_obstacles | self.moving_obstacles

    def step(self, action):
        # update moving obstacles before the agent moves
        self._update_moving_obstacles()

        self.steps_taken += 1

        # stochastic slip
        if self.slip_prob > 0 and np.random.rand() < self.slip_prob:
            action = np.random.randint(4)

        x, y = self.state

        # 0 = up, 1 = down, 2 = left, 3 = right
        if action == 0:
            nx, ny = x - 1, y
        elif action == 1:
            nx, ny = x + 1, y
        elif action == 2:
            nx, ny = x, y - 1
        elif action == 3:
            nx, ny = x, y + 1
        else:
            raise ValueError("Invalid action")

        info = {"is_collision": False, "is_boundary": False, "is_parked": False}
        done = False

        # Boundary check
        if not self._in_bounds(nx, ny):
            reward = self.boundary_penalty
            info["is_boundary"] = True
            return self.state, reward, done, info

        next_state = (nx, ny)

        # Obstacle / collision check
        if next_state in self.obstacles:
            reward = self.collision_penalty
            info["is_collision"] = True
            return self.state, reward, done, info

        # Parking success
        if next_state in self.parking_spots:
            reward = self.park_reward
            done = True
            self.state = next_state
            info["is_parked"] = True
            return next_state, reward, done, info

        # ---------------- NORMAL MOVE ----------------
        reward = self.move_penalty

        # NEW: turning penalty (kills zig-zag)
        if self.prev_action is not None and action != self.prev_action:
            reward -= 1.0

        self.prev_action = action

        # NEW: revisit penalty (kills oscillation)
        self.visit_count[next_state] = self.visit_count.get(next_state, 0) + 1
        if self.visit_count[next_state] > 1:
            reward -= 1.5

        # NEW: early anti-wandering
        if self.steps_taken > 20:
            reward -= 1
        if self.steps_taken > 50:
            reward -= 2

        # reward shaping (weaker, safer)
        if self.reward_shaping:
            d_before = self._nearest_goal_distance((x, y))
            d_after = self._nearest_goal_distance(next_state)
            reward += self.shaping_coeff * (d_before - d_after)

        self.state = next_state
        return next_state, reward, done, info

    # all possible (row, col) positions
    def get_state_space(self):
        return [(i, j) for i in range(self.size) for j in range(self.size)]

    # grid representation for plotting
    def render_map(self):
        grid = np.zeros((self.size, self.size), dtype=int)
        for (i, j) in self.obstacles:
            grid[i, j] = -1
        for (i, j) in self.parking_spots:
            grid[i, j] = 2
        sx, sy = self.start
        grid[sx, sy] = 3
        return grid

## environment builders
# simple layout, few obstacles
def env_builder_easy():
    obstacle_map = {
        "parking_slots": {
            (2,0),(3,0),(4,0),(5,0),(7,0),
            (2,9),(4,9),(5,9),(6,9),(7,9),(8,9),
            (2,4),(2,5),
            (3,3),(4,3),(5,3),(6,3),
            (3,6),(4,6),(5,6),(6,6),
            (9,9)
        },

        "storage": {(0,8), (0,9)},

        "tiang": {
            (6,0), (3,9), (9,0),
            (7,3), (7,6),
            (2,3), (2,6)
        },

        "bush": {
            (3,4),(4,4),(5,4),(6,4),
            (3,5),(4,5),(5,5),(6,5)
        },

        "guard": {(9,2), (9,3)},

        "parked_car": {
            (2,0),(3,0),(4,0),(7,0),
            (2,9),(4,9),(5,9),(6,9),(8,9),
            (2,5),
            (3,3),(4,3),(5,3),(6,3),
            (3,6),(4,6),(5,6),(6,6)
        },

        "female": {(7,4), (7,5)},
        "waiting": {(4,1)},
        "exiting": {(1,4),(7,8)},
        "empty_soon": {(2,4), (5,0),(7,9)}
    }

    obstacles = set().union(
        *[v for k, v in obstacle_map.items() if k != "parking_slots"]
    )

    env = ParkingGrid(
        size=10,
        start=(0,0),
        parking_spots=[(9,9)],
        obstacles=obstacles,
        move_penalty=-3,
        collision_penalty=-50,
        park_reward=200,
        boundary_penalty=-20,
        reward_shaping=True,
        shaping_coeff=0.1,
        slip_prob=0.1
    )

    env.visual_objects = obstacle_map
    return env

# more obstacles, slight randomness
def env_builder_medium():
    obstacle_map = {

        # ---------------- PARKING SLOTS (VISUAL ONLY) ----------------
        "parking_slots": {
            # left vertical slots
            (3,2),(4,2),(5,2),(6,2),(7,2),(8,2),(9,2),(10,2),(11,2),(12,2),(13,2),(14,2),
            (3,3),(4,3),(5,3),(6,3),(7,3),(8,3),(9,3),(10,3),(11,3),(12,3),(13,3),(14,3),

            # top horizontal slots
            (2,6),(2,7),(2,8),
            (2,11),(2,12),(2,13),(2,14),

            # second row slots
            (4,6),(4,7),(4,8),
            (4,11),(4,12),(4,13),(4,14),

            # mid vertical
            (8,7),(9,7),(10,7),
            (8,8),(9,8),(10,8),

            # lower mid
            (12,7),(13,7),
            (12,8),(13,8),

            # right clusters
            (9,12),(9,13),(9,14),
            (9,17),(9,18),(9,19),

            (10,12),(10,13),(10,14),
            (10,17),(10,18),(10,19),

            # female area
            (6,12),(6,13),(6,15),(6,16),(6,18),(6,19),
            (7,12),(7,13),(7,15),(7,16),(7,18),(7,19)
        },

        # ---------------- TICKET MACHINE AREA ----------------
        "ticket_machine": {
            (0,17),(0,18),(0,19),
            (1,17),(1,18),(1,19),
            (2,17),(2,18),(2,19),
            (3,17),(3,18),(3,19)
        },

        # ---------------- WATER LEAK / PIPE BOCOR ----------------
        "water_leak": {
            (13,14),(13,15),(13,16),(13,17),(13,18),
            (14,14),(14,15),(14,16),(14,17),(14,18),
            (15,17),(15,18)
        },

        # ---------------- BARRIER CONES ----------------
        "barrier_cone": {
            (15,13),(15,14),(15,15),(15,16),
            (16,17),(16,18),(16,19),
            (12,13),(12,14),(12,15),(12,16),(12,17),(12,18),(12,19),
            (13,13),(14,13),
            (13,19),(14,19),(15,19)
        },

        # ---------------- WALL ----------------
        "wall": {
            (15,2),(15,3),(15,4),(15,5),(15,6),(15,7),(15,8),(15,9),(15,10),(15,11),(15,12),
            (6,14),(6,17),
            (7,14),(7,17)
        },

        # ---------------- ENTRANCE ----------------
        "entrance": {
            (17,18),(18,18),(19,18),
            (17,19),(18,19),(19,19),
            (17,17)
        },

        # ---------------- BUSH ----------------
        "bush": {
            (16,2),(16,3),(16,4),(16,5),(16,6),(16,7),(16,8),(16,9),(16,10),(16,11),(16,12),
            (17,2),(17,3),(17,4),(17,5),(17,6),(17,7),(17,8),(17,9),(17,10),(17,11),(17,12),
            (7,4),(7,5),(7,6),(7,7),
            (3,6),(3,7),(3,8),
            (3,11),(3,12),(3,13),(3,14)
        },

        # ---------------- PARKED CARS ----------------
        "parked_car": {
            (3,2),(6,2),(7,2),(8,2),(9,2),(10,2),(11,2),(12,2),(13,2),(14,2),
            (3,3),(4,3),(5,2),(6,3),(7,3),(8,3),(9,3),(11,3),(12,3),(13,3),(14,3),

            (2,6),(2,7),(2,8),
            (2,11),(2,13),(2,14),
            (4,6),(4,7),(4,8),
            (4,11),(4,12),(4,13),(4,14),

            (9,12),(9,13),(9,14),
            (9,17),(9,18),(9,19),

            (10,12),(10,13),(10,14),
            (10,18),(10,19),

            (8,7),(10,7),(13,7),
            (8,8),(9,8),(10,8),
            (12,8),(13,8)
        },

        # ---------------- FEMALE PARKING (BLOCKED) ----------------
        "female": {
            (6,12),(6,13),(6,15),(6,16),(6,18),(6,19),
            (7,12),(7,13),(7,15),(7,16),(7,18),(7,19)
        },

        # ---------------- DYNAMIC OBJECTS ----------------
        "waiting": {(3,1)},
        "exiting": {(10,4),(5,4),(1,12),(5,8)},
        "empty_soon": {(10,17),(9,7),(12,7),(4,2),(10,3),(5,3),(2,12),(4,8)}
    }

    # exclude parking slots from obstacles
    obstacles = set().union(
        *[v for k, v in obstacle_map.items() if k != "parking_slots"]
    )

    env = ParkingGrid(
        size=20,
        start=(17,16),
        parking_spots=[(9,9)],
        obstacles=obstacles,
        move_penalty=-3,
        collision_penalty=-50,
        park_reward=200,
        boundary_penalty=-20,
        reward_shaping=True,
        shaping_coeff=0.1,
        slip_prob=0.1
    )

    env.visual_objects = obstacle_map
    return env

# hardest layout with moving obstacles
def env_builder_hard():
    return ParkingGrid(
        size=10,
        start=(0,0),
        parking_spots=[(9,9)],
        obstacles=[
            (2,2),(2,3),(2,4),
            (4,4),(4,5),(4,6),
            (6,2),(6,3),(6,4),
            (7,7)
        ],
        moving_obstacles=[(1,9),(3,9),(5,9),(7,9),(8,9)],
        move_penalty=-3,
        collision_penalty=-50,
        park_reward=200,
        boundary_penalty=-20,
        reward_shaping=True,
        shaping_coeff=0.1,
        slip_prob=0.1
    )

ENV_BUILDERS = {
    "easy": env_builder_easy,
    "medium": env_builder_medium,
    "hard": env_builder_hard
}

# --- 3. Visualization Helper ---
def draw_grid(env, title="Parking Grid"):
    fig, ax = plt.subplots(figsize=(6, 6))
    grid = env.render_map()
    
    # 1. Draw the Grid
    # origin='upper' puts (0,0) at top-left.
    ax.imshow(grid, cmap='tab20', origin='upper')
    
    # 2. FIX: Invert Y-axis to match your notebook's look
    ax.invert_yaxis() 

    # 3. Draw Agent (x=col, y=row)
    # Note: We plot (env.state[1], env.state[0]) which is (y, x)
    ax.scatter([env.state[1]], [env.state[0]], marker='s', s=150, color='blue', edgecolor='white', label='Agent')
    
    # 4. Draw Goal
    gx, gy = list(env.parking_spots)[0]
    ax.scatter([gy], [gx], marker='*', s=200, color='gold', edgecolor='black', label='Goal')

    # 5. Draw Objects
    if hasattr(env, "visual_objects"):
        for k, cells in env.visual_objects.items():
            for (x, y) in cells:
                color = 'gray'
                if 'cone' in k: color = 'red'
                if 'parked' in k: color = 'lightblue'
                if 'wall' in k: color = 'black'
                
                # Rectangle takes (x,y) -> (col, row)
                rect = Rectangle((y-0.5, x-0.5), 1, 1, facecolor=color, alpha=0.5, edgecolor='none')
                ax.add_patch(rect)

    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    return fig

# Helper to safely get Q-values
def qvals_for_state(qdict, state):
    if qdict is None or state not in qdict:
        return np.zeros(4)
    return qdict[state]

def load_models():
    # Ensure these files are in the same directory!
    with open("q_tables.pkl", "rb") as f:
        q_tables = pickle.load(f)
    with open("dq_combined_tables.pkl", "rb") as f:
        dq_tables = pickle.load(f)
    return q_tables, dq_tables

# --- 4. Streamlit App Main Loop ---
st.set_page_config(page_title="Auto Parking RL Demo", layout="wide")

st.title("🚗 Grid Auto-Parking Presentation")
st.caption("Visualizing trained policies for Q-Learning vs Double Q-Learning.")

# Sidebar
with st.sidebar:
    st.header("Settings")
    level = st.selectbox("Difficulty", ["easy", "medium", "hard"], index=1)
    algo = st.radio("Agent", ["Q-Learning", "Double-Q"], index=0)
    speed = st.slider("Animation Speed (seconds)", 0.0, 0.5, 0.05, 0.01)
    max_steps = st.slider("Max Steps", 50, 500, 200, 50)
    run_btn = st.button("▶ Start Animation", type="primary")
    reset_btn = st.button("↺ Reset")

# State Management
if "q_tables" not in st.session_state:
    st.session_state.q_tables = None
    st.session_state.dq_tables = None

# Load Data
if st.session_state.q_tables is None:
    try:
        st.session_state.q_tables, st.session_state.dq_tables = load_models()
        st.success("Models loaded successfully!")
    except Exception as e:
        st.error(f"Could not load model files (q_tables.pkl, dq_combined_tables.pkl). Error: {e}")
        st.stop()

# Select Policy
qdict = st.session_state.q_tables.get(level, {})
dqdict = st.session_state.dq_tables.get(level, {})
active_qdict = qdict if algo == "Q-Learning" else dqdict

# --- Reset Logic ---
if reset_btn or "env" not in st.session_state or st.session_state.get("stored_level") != level:
    st.session_state.env = ENV_BUILDERS[level]()
    st.session_state.stored_level = level
    st.session_state.step = 0
    st.session_state.done = False

# !!! CRITICAL FIX: Ensure 'env' is defined globally for this run !!!
env = st.session_state.env 

placeholder = st.empty()

# Run Animation
if run_btn:
    st.session_state.done = False
    env.reset()
    
    for _ in range(max_steps):
        # 1. Choose Greedy Action
        qvals = qvals_for_state(active_qdict, env.state)
        action = int(np.argmax(qvals))
        
        # 2. Step
        _, _, done, _ = env.step(action)
        st.session_state.step += 1
        
        # 3. Draw
        fig = draw_grid(env, title=f"{algo} - Step {st.session_state.step}")
        placeholder.pyplot(fig, clear_figure=True)
        time.sleep(speed)
        
        if done:
            st.success("Parked! 🎉")
            break
            
    if not done:
        st.warning("Max steps reached.")
else:
    # Show initial state
    fig = draw_grid(env, title=f"{algo} - Ready")
    placeholder.pyplot(fig, clear_figure=True)
