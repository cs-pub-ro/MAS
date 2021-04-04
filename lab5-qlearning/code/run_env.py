from environment import *
from random import choice


if __name__ == "__main__":
    TEST_SUITE = "tests/0a/"

    EXT = ".txt"
    SI = "si"
    SF = "sf"

    start_stream = open(TEST_SUITE + SI + EXT)
    end_stream = open(TEST_SUITE + SF + EXT)

    env = DynamicBlocksWorldEnvironment(BlocksWorld(input_stream=start_stream), BlocksWorld(input_stream=end_stream),
                                        dynamics=0.5)

    available_actions = [NoAction(), Unstack(Block("A"), Block("B")), PutDown(Block("A")), PickUp(Block("B")),
                         Stack(Block("B"), Block("C"))]

    for i in range(100):
        print("=============== STEP %i ===============" % i)

        act = choice(available_actions)
        state, reward, done = env.step(act)

        print("## Chosen action: " + str(act))
        print("## State\n")
        print(state)

        print(env)

        print("\n")
        print("Reward: " + str(reward))

        print("\n")
        print("done: " + str(done))

        if done:
            break
