import logging
from minigrid.manual_control import ManualControl
import pygame

logger = logging.getLogger(__name__)


def collect_demo_in_gridworld(env):
    """
    Collects a demonstration from a user.
    """
    traj = []
    manual_control = ManualControl(env)
    done_flag = False
    while not manual_control.closed and not done_flag:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                manual_control.env.close()
                done_flag = True
                break
            if event.type == pygame.KEYDOWN:
                event.key = pygame.key.name(int(event.key))
                print(event.key)
                traj.append(event.key)
                manual_control.key_handler(event)

                if event.key == "escape":
                    pygame.quit()
                    done_flag = True
                    break

    logger.info("Demo collected:")
    logger.info(f"{traj}")
    return traj


