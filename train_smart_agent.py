import random

from environments import ThermostatEnvironment
from agents import SmartAgent
from models import Interaction


def main():
    EPISODES = 500
    MODEL_UPDATE_INTERVAL = 100

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

    agent = SmartAgent(len(env.get_observation()), len(env.get_actions()))

    for episode in range(EPISODES):

        selected_actions = []

        observation = env.get_observation()

        while True:
            action = agent.select_action(observation, env.get_actions())

            if random.random() < 0.2:
                action = random.choice(env.get_actions())

            selected_actions.append(action)

            reward, next_observation, done, extra_info = env.step(action)

            interaction = Interaction(observation, action, reward)
            agent.memory.append(interaction)

            observation = next_observation

            if done:
                print('Episode {}, {}, {} days survived, {} money spent, actions {}'.format(
                    episode + 1,
                    extra_info['done_reason'],
                    extra_info['days_survived'],
                    round(extra_info['money_spent'], 2),
                    selected_actions))
                break

        env.reset()

        if (episode + 1) % MODEL_UPDATE_INTERVAL == 0:
            agent.update_model()
            agent.save_model(episode + 1)


if __name__ == '__main__':
    main()
