from puzzleMe import GameGrid
from RL_brain import DeepQNetwork


def run_maze():
    step = 0
    for episode in range(300):
        # initial observation
        observation = env.reset()
        print("obervation",observation,"type",type(observation))
        count =0;
        while True:
            count += 1
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)
            # if(count %50 == 0):
            #     print("action",action,"type",type(action))
            # RL take action and get next observation and reward
            observation_, reward, done, Score = env.step(action)

            maxSum = env.getMaxSum()

            emptySum = env.getEmpt()
            if (count % 5 == 0):
                print("observation_",observation_,"Score",Score,"Max",env.getMax(),"reward",reward,"action",action,"type",type(action),"maxSum",maxSum,"empty",emptySum)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 100) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = GameGrid()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      batch_size=128,

                      output_graph=False
                      )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()