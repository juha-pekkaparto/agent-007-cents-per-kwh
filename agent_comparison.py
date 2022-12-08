from environments import ThermostatEnvironment
from agents import BaselineAgent, MinimizerAgent, RandomAgent, SmartAgent


def main():
    env = ThermostatEnvironment(
        min_temperature=14,
        max_temperature=21,
        baseline_temperature=21,
        consumption_at_baseline_temperature=33.33,
        budget=125.0,
        days_to_survive=30,
        price_mu=0.125,
        price_sigma=0.125
    )

    EPISODES = 20
    AGENTS = {
        'baseline_agent': BaselineAgent(),
        'minimizer_agent': MinimizerAgent(),
        'random_agent': RandomAgent(),
        'smart_agent': SmartAgent(len(env.get_observation()), len(env.get_actions()), saved_model_path='smart_agent_500.h5')
    }

    agent_rewards = {
        'baseline_agent': 0.0,
        'minimizer_agent': 0.0,
        'random_agent': 0.0,
        'smart_agent': 0.0
    }

    agent_rewards_average = {
        'baseline_agent': 0.0,
        'minimizer_agent': 0.0,
        'random_agent': 0.0,
        'smart_agent': 0.0
    }

    for agent_name in AGENTS.keys():

        agent = AGENTS[agent_name]

        for episode in range(EPISODES):

            observation = env.get_observation()

            while True:
                action = agent.select_action(observation, env.get_actions())
                reward, next_observation, done, extra_info = env.step(action)
                observation = next_observation

                agent_rewards[agent_name] += reward
                agent_rewards_average[agent_name] += reward

                if done:
                    print('{}, episode {}, {}, {} days survived, {} money spent, reward {}'.format(
                        agent_name,
                        episode + 1,
                        extra_info['done_reason'],
                        extra_info['days_survived'],
                        round(extra_info['money_spent'], 2),
                        agent_rewards[agent_name]))
                    agent_rewards[agent_name] = 0.0
                    break

            env.reset()

    for agent_name in AGENTS.keys():
        print(agent_name, 'average reward {}'.format(agent_rewards_average[agent_name] / EPISODES))


if __name__ == '__main__':
    main()
