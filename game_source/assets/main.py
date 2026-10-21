"""Browser entry point. Game logic and physics remain Python."""
# /// script
# dependencies = []
# ///
import asyncio
from bootstrap import prepare_browser


async def main() -> None:
    """Yield once per frame so the browser can paint and process input."""
    await prepare_browser()
    # Delay native imports until their WebAssembly libraries are registered.
    pygame = __import__("pygame")
    config = __import__("asteroids.config", fromlist=["WIDTH"])
    game_module = __import__("asteroids.game", fromlist=["Game", "Controls"])
    Game, Controls = game_module.Game, game_module.Controls
    pygame.init()
    screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT))
    pygame.display.set_caption('Asteroids — carve, collect, upgrade')
    game = Game(seed=17)
    print("ASTEROIDS_READY")
    previous = pygame.time.get_ticks() / 1000
    try:
        while game.running:
            now = pygame.time.get_ticks() / 1000
            frame_dt = max(0, now - previous)
            previous = now
            for event in pygame.event.get():
                game.handle_event(event)
            keys = pygame.key.get_pressed()
            controls = Controls(
                turn=float(keys[pygame.K_a]) - float(keys[pygame.K_d]),
                thrust=float(keys[pygame.K_w] or keys[pygame.K_UP]) - float(keys[pygame.K_s] or keys[pygame.K_DOWN]),
                aim=game.screen_to_world(pygame.mouse.get_pos()),
                firing=bool(keys[pygame.K_SPACE] or pygame.mouse.get_pressed()[0]),
            )
            game.advance(frame_dt, controls)
            game.draw(screen)
            pygame.display.flip()
            await asyncio.sleep(0)
    finally:
        game.close()
        pygame.quit()


asyncio.run(main())
