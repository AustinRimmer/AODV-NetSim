from __future__ import annotations

import random
from typing import Optional, Tuple, List

import numpy as np
import pygame
import pygame.freetype

from manet_env import MANETEnv
from routing import hop_count_route, energy_aware_route

#colour helpers

def energy_colour(energy: float, max_energy: float) -> Tuple[int, int, int]:
    #maps energy fraction to a colour green -> yellow -> red
    #like a manual tween funct
    """
    node colour coding
    --------_-__________--------------___________________________---------
      green   -> high energy (>70 )
      Yellow  -> medium energy (30 – 70)
      red     -> low energy (<30)
      dark grey -> dead
    -_________---------_____________----------------____________--------_-
    """

    frac = max(0.0, min(1.0, energy / max(max_energy, 1e-9)))
    if frac > 0.7:
        r, g = int((1 - frac) / 0.3 * 255), 220
    elif frac > 0.3:
        t = (frac - 0.3) / 0.4
        r, g = int((1 - t) * 255), int(t * 220 + (1 - t) * 180)
    else:
        r, g = 220, int(frac / 0.3 * 180)
    return (r, g, 30)


#MANETRenderer

class MANETRenderer:

    SIM_W   = 700
    SIM_H   = 700
    PANEL_W = 320
    WIN_H   = 700
    NODE_R  = 8
    DEAD_R  = 5
    EDGE_ALPHA = 100

    def __init__(self, env: MANETEnv):
        self.env = env
        pygame.init()
        pygame.display.set_caption("MANET Simulation - COMP 4203")
        self.screen   = pygame.display.set_mode((self.SIM_W + self.PANEL_W, self.WIN_H))
        self.clock    = pygame.time.Clock()
        pygame.freetype.init()
        self._font_sm = pygame.freetype.SysFont("monospace", 12)
        self._font_md = pygame.freetype.SysFont("monospace", 14, bold=True)
        self._font_lg = pygame.freetype.SysFont("monospace", 18, bold=True)
        self.sim_surf = pygame.Surface((self.SIM_W, self.SIM_H))

        #rebuilt every step, never accumulates
        self._active_paths: List[Tuple[List[int], int, int]] = []

    #just for translating pixels to actual space, had some massive errors post implmntation
    def to_pixels(self, pos: np.ndarray) -> Tuple[int, int]:
        x = int(pos[0] / self.env.area_size * (self.SIM_W - 20) + 10)
        y = int(pos[1] / self.env.area_size * (self.SIM_H - 20) + 10)
        return x, y

    def draw(self, active_action: int = 0, paused: bool = False,
             step: int = 0, speed: int = 1,
             frame: int = 0, viz_flows: Optional[List] = None) -> None:
        env = self.env

        #bg + grid
        #f
        self.sim_surf.fill((18, 18, 28))
        grid_col = (35, 35, 45)
        for x in range(0, self.SIM_W, 70):
            pygame.draw.line(self.sim_surf, grid_col, (x, 0), (x, self.SIM_H))
        for y in range(0, self.SIM_H, 70):
            pygame.draw.line(self.sim_surf, grid_col, (0, y), (self.SIM_W, y))

        #edges between nodes
        edge_surf = pygame.Surface((self.SIM_W, self.SIM_H), pygame.SRCALPHA) #SRCALPHA alpha chnl support
        for u, v in env.graph.edges():
            pu = self.to_pixels(env.positions[u])
            pv = self.to_pixels(env.positions[v])
            pygame.draw.line(edge_surf, (80, 140, 200, self.EDGE_ALPHA), pu, pv, 1)
        self.sim_surf.blit(edge_surf, (0, 0))

        #active path flash, only have 5 flows
        FLOW_COLOURS = [
            (255, 220, 40,  255),   #yellow
            (40,  220, 255, 255),   #cyan
            (255, 100, 40,  255),   #orange
            (180, 40,  255, 255),   #purple
            (40,  255, 120, 255),   #green
        ]
        path_surf = pygame.Surface((self.SIM_W, self.SIM_H), pygame.SRCALPHA)
        for path, expiry, fi in self._active_paths:
            if frame > expiry: #this is for the old implementation
                continue
            #kill ghost lines ifthe frame source or dest dies
            if not env.alive[path[0]] or not env.alive[path[-1]]: #this is for if endpoint died
                continue

            col = FLOW_COLOURS[fi % len(FLOW_COLOURS)]
            #draws segment by seg, draws paths hop by hop
                        #counts num gaps btwn nodes
            for i in range(len(path) - 1):
                if env.alive[path[i]] and env.alive[path[i + 1]]:
                    pu = self.to_pixels(env.positions[path[i]])
                    pv = self.to_pixels(env.positions[path[i + 1]])
                    pygame.draw.line(path_surf, col, pu, pv, 5)
        self.sim_surf.blit(path_surf, (0, 0))

        #nodes
        max_e = env.energy_max
        for i in range(env.n_nodes):
            px, py = self.to_pixels(env.positions[i])
            if not env.alive[i]:
                pygame.draw.circle(self.sim_surf, (60, 60, 60), (px, py), self.DEAD_R)
                pygame.draw.circle(self.sim_surf, (40, 40, 40), (px, py), self.DEAD_R, 1)
            else:
                col = energy_colour(env.energies[i], max_e)
                pygame.draw.circle(self.sim_surf, col, (px, py), self.NODE_R)
                pygame.draw.circle(self.sim_surf, (255, 255, 255), (px, py), self.NODE_R, 1)
                self._font_sm.render_to(
                    self.sim_surf, (px + self.NODE_R + 1, py - 6), str(i), (200, 200, 200)
                )

        #S/D markers, draws the labels and rings

        flows_to_mark = viz_flows if viz_flows is not None else env.flows #flow colour selector
        for fi, (src, dst) in enumerate(flows_to_mark):
            col = FLOW_COLOURS[fi % len(FLOW_COLOURS)] #neat bit of code that allows us to cycle back thru colours if we have more
            ring_col = (col[0], col[1], col[2])   # same colour as the path
            #we do both src and dest draws in pairs
            for node, tag in [(src, "S"), (dst, "D")]:
                if env.alive[node]:
                    px, py = self.to_pixels(env.positions[node])
                    pygame.draw.circle(self.sim_surf, ring_col, (px, py), self.NODE_R + 3, 2)
                    self._font_sm.render_to(
                        self.sim_surf, (px - 4, py - 5), tag, (255, 255, 100)
                    )

        self.screen.fill((10, 10, 20))
        self.screen.blit(self.sim_surf, (0, 0))
        self._draw_panel(active_action, paused, step, speed, viz_flows)
        pygame.display.flip()

    #for drawing panel (duh)
    def _draw_panel(self, action: int, paused: bool, step: int, speed: int,
                    viz_flows: Optional[List] = None):
        env = self.env
        px0 = self.SIM_W + 12
        y   = 14
        lh  = 20
        #these are the helper functs i made for rendering
        #we use nonlocal y si that we avoid repedative pygame calls, that way we dont have to keep passing it
        def txt(text, color=(200, 200, 200), font=None, x=None):
            nonlocal y #modifies outer y var
            f = font or self._font_sm
            f.render_to(self.screen, (x or px0, y), text, color)
            y += lh #this moves cursor down a line
        #draws a sepperator
        def sep():
            nonlocal y
            pygame.draw.line(self.screen, (60, 60, 80), (self.SIM_W + 6, y), (self.SIM_W + self.PANEL_W - 6, y), 1) #horz divider
            y += 6

        #all of the panel rendering stuff

        self._font_lg.render_to(self.screen, (px0, y), "MANET SIM", (100, 180, 255))
        y += 26
        sep()

        alg_name = "Standard AODV" if action == 0 else "Energy Aware AODV"
        alg_col  = (100, 200, 255) if action == 0 else (255, 160, 60)
        txt("Algorithm:", (160, 160, 160))
        y -= lh
        self._font_md.render_to(self.screen, (px0 + 95, y), alg_name, alg_col)
        y += 12
        sep()

        txt(f"Step:          {step}", (200, 220, 200))
        alive = int(env.alive.sum())
        dead  = env.n_nodes - alive
        txt(f"Alive nodes:   {alive} / {env.n_nodes}", (100, 220, 100))
        txt(f"Dead nodes:    {dead}", (220, 80, 80) if dead > 0 else (100, 120, 100))
        txt(f"Speed:         {speed}x", (180, 180, 220))
        txt("PAUSED" if paused else "RUNNING",
            (255, 200, 60) if paused else (80, 220, 80), font=self._font_md)
        sep()
        #metrics render
        info = env._info()
        txt("<| Metrics |>", (160, 160, 200), font=self._font_md)
        y += 4
        txt(f"PDR:           {info['pdr']:.3f}")
        txt(f"Path len (avg):{info['mean_path_length']:.2f} hops")
        txt(f"Energy var:    {info['energy_variance']:.1f}")
        fd  = info["first_death_time"]
        txt(f"1st death at:  {'—' if fd  is None else str(fd)}")
        t30 = info["time_30pct_death"]
        txt(f"30% dead at:   {'—' if t30 is None else str(t30)}")
        td  = info["time_disconnection"]
        txt(f"Disconnect at: {'—' if td  is None else str(td)}")
        sep()

        #energy legend render
        txt("<| Energy Legend |>", (160, 160, 200), font=self._font_md)
        y += 4
        for label, col in [
            ("High (>70 %)",  (60, 220, 30)),
            ("Med (30-70 %)", (220, 180, 30)),
            ("Low (<30 %)",   (220, 60, 30)),
            ("Dead node",     (60, 60, 60)),
        ]:
            pygame.draw.circle(self.screen, col, (px0 + 6, y + 6), 6)
            self._font_sm.render_to(self.screen, (px0 + 18, y), label, (200, 200, 200))
            y += lh
        sep()

        #Flow list render
        FLOW_COLOURS = [
            (255, 220, 40),
            (40,  220, 255),
            (255, 100, 40),
            (180, 40,  255),
            (40,  255, 120),
        ]
        flows_to_show = viz_flows if viz_flows is not None else env.flows
        txt("<| Traffic Flows |>", (160, 160, 200), font=self._font_md)
        y += 2
        for fi, (src, dst) in enumerate(flows_to_show):
            s_ok = "node" #if env.alive[src] else "x" DEPRECIATED no longer keeping all src and dest dead
            d_ok = "node" #if env.alive[dst] else "x"
            dot_col = FLOW_COLOURS[fi % len(FLOW_COLOURS)]
            pygame.draw.circle(self.screen, dot_col, (px0 + 6, y + 6), 5)
            self._font_sm.render_to(
                self.screen, (px0 + 18, y),
                f"{fi + 1}: {src}({s_ok}) -> {dst}({d_ok})",
                (200, 200, 200)
            )
            y += lh
        sep()

        #control render
        txt("<| Controls |>", (140, 140, 180), font=self._font_md)
        y += 2
        for line in [
            "SPACE  pause/resume",
            "TAB    toggle algorithm",
            "R      reset",
            "+/-    speed",
            "Q/ESC  quit",
        ]:
            txt(line, (140, 140, 160))

    def close(self):
        pygame.quit()



#main game loop
#pretty standard pygame loop
def run_live(env_kwargs: Optional[dict] = None, seed: int = 0):

    if env_kwargs is None:
        env_kwargs = {}

    env = MANETEnv(render_mode="human", **env_kwargs)
    renderer = MANETRenderer(env)
    env.reset(seed=seed)

    #viz_flows is ONLY for drawing its never actually psased to env.step()
    #inits
    viz_flows = list(env.flows)

    action  = 0
    paused  = False
    speed   = 1
    step    = 0
    frame   = 0
    running = True

    #run loop
    while running:

        #event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

                elif event.key == pygame.K_SPACE:
                    paused = not paused

                elif event.key == pygame.K_TAB:
                    action = 1 - action
                    print(f"[viz] Switched to {'energy-aware' if action else 'hop-count'} routing")

                elif event.key == pygame.K_r:
                    seed += 1
                    env.reset(seed=seed)
                    viz_flows = list(env.flows)
                    step = 0
                    renderer._active_paths = []
                    print(f"[viz] Reset with seed {seed}")

                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    speed = min(speed + 1, 20)

                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed = max(speed - 1, 1)

        #sim step
        if not paused:
            for _ in range(speed): #note speed is steps per frame, chnage speed to speed it up (duh)
                _, _, terminated, truncated, info = env.step(action)
                step += 1

                #per step checkl viz_flows and reassign dead srcs and dests
                living = [i for i in range(env.n_nodes) if env.alive[i]]
                if len(living) >= 2:
                    new_viz_flows = []
                    #reassign dead srcs and dests
                    for src, dst in viz_flows:
                        if not env.alive[src] or not env.alive[dst]:
                            pair = random.sample(living, 2)
                            new_viz_flows.append((pair[0], pair[1]))
                        #if theyre not dead just draw them
                        else:
                            new_viz_flows.append((src, dst))
                    viz_flows = new_viz_flows

                #rebuild active paths from scratch for every step
                renderer._active_paths = []
                for fi, (src, dst) in enumerate(viz_flows):
                    if env.alive[src] and env.alive[dst]:
                        if action == 0:
                            path = hop_count_route(env.graph, src, dst)
                        else:
                            path = energy_aware_route(env.graph, src, dst, env.energies, env.alpha, env.beta )
                        if path and len(path) >= 2:
                            renderer._active_paths.append((path, frame + 30, fi))

                #an auto reset
                if terminated or truncated:
                    seed += 1
                    env.reset(seed=seed)
                    viz_flows = list(env.flows)
                    step = 0
                    renderer._active_paths = []
                    break

        #render
        renderer.draw(
            active_action=action, paused=paused,
            step=step, speed=speed, frame=frame,
            viz_flows=viz_flows,
        )
        #affects sim speed but makes it look choppy if below 30
        renderer.clock.tick(30)
        if not paused:
            frame += 1

    renderer.close()
    env.close()



#entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Live MANET visualisation")
    parser.add_argument("--seed",      type=int,   default=0)
    parser.add_argument("--n-nodes",   type=int,   default=30)
    parser.add_argument("--max-steps", type=int,   default=800)
    parser.add_argument("--alpha",     type=float, default=1.0)
    parser.add_argument("--beta",      type=float, default=50.0)
    args = parser.parse_args()

    run_live(
        env_kwargs=dict(
            n_nodes   = args.n_nodes,
            max_steps = args.max_steps,
            alpha     = args.alpha,
            beta      = args.beta,
        ),
        seed=args.seed,
    )