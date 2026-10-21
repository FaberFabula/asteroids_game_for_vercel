"""World units, timing, and bounded spawning knobs used by the game."""
import math

TURRET_HALF_ARC: float = math.pi / 3  # 120-degree total arc, relative to the hull.
WIDTH: int = 1000
HEIGHT: int = 700
PIXELS_PER_UNIT: float = 20.0
WORLD_WIDTH: float = WIDTH / PIXELS_PER_UNIT
WORLD_HEIGHT: float = HEIGHT / PIXELS_PER_UNIT
FIXED_DT: float = 1.0 / 60.0
MAX_FRAME_TIME: float = 0.15
WAVE_INTERVAL: float = 28.0
MAX_ASTEROIDS: int = 80
MAX_ENEMIES: int = 12
MAX_PROJECTILES: int = 160
MAX_PICKUPS: int = 24
POINTS_PER_AREA: float = 100.0
DROP_AREA: float = 0.7
LASER_DURATION: float = 6.0
