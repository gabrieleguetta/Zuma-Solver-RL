import os
import random

import zuma
import ex3
import time


def solve(game: zuma.Game, debug=False):
    # t1 = time.time()
    policy = ex3.Policy(game)
    # t2 = time.time()
    # print("Policy generation took: ", t2 - t1, " seconds.\n")
    # t3 = time.time()
    print("The average score of the policy is:",
          game.evaluate_policy(policy, 50000, visualize=debug))
    print("The policy file size is:", os.path.getsize("zuma_policy.pkl"))
    # t4 = time.time()
    # print("Policy evaluation took: ", t4 - t3, " seconds")


example = {
    'chosen_action_prob': {1: 0.6, 2: 0.7, 3: 0.5, 4: 0.9},
    # 'chosen_action_prob': {1: 1, 2: 1, 3: 1, 4: 1},
    'next_color_dist': {1: 0.1, 2: 0.6, 3: 0.15, 4: 0.15},
    # 'next_color_dist': {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25},
    'color_pop_prob': {1: 0.6, 2: 0.7, 3: 0.4, 4: 0.9},
    'color_pop_reward': {'3_pop': {1: 3, 2: 1, 3: 2, 4: 2},
                         'extra_pop': {1: 1, 2: 2, 3: 3, 4: 1}},
    'color_not_finished_punishment': {1: 2, 2: 3, 3: 5, 4: 1},
    'finished_reward': 150,
    'seed': 42}


def main():
    debug_mode = False
    game = zuma.create_zuma_game((50, 10, [random.randint(1, 4) for _ in range(random.randint(1, 10))], example, debug_mode))
    solve(game, debug_mode)


if __name__ == "__main__":
    main()
