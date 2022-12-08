import math

import numpy as np


class ThermostatEnvironment:
    def __init__(
        self,
        min_temperature: int,
        max_temperature: int,
        baseline_temperature: int,
        consumption_at_baseline_temperature: float,
        budget: float,
        days_to_survive: int,
        price_mu: float,
        price_sigma: float):

        self.baseline_temperature = baseline_temperature
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.temperature = baseline_temperature

        self.consumption_at_baseline_temperature = consumption_at_baseline_temperature
        self.consumed_electricity = 0.0

        self.budget = budget
        self.money_spent = 0.0

        self.days_to_survive = days_to_survive
        self.days_survived = 0

        self.price_mu = price_mu
        self.price_sigma = price_sigma
        self.price = self.generate_new_price()

    def get_observation(self):
        return np.array([
            self.temperature,
            abs(self.baseline_temperature - self.temperature),
            self.consumed_electricity,
            self.price, 2,
            self.money_spent,
            self.budget - self.money_spent,
            self.days_to_survive - self.days_survived
        ])

    def get_actions(self):
        return np.array((range(self.min_temperature, self.max_temperature + 1, 1)))

    def step(self, action: int):
        self.temperature = action

        self.consumed_electricity = self.calculate_consumed_electricity()
        self.money_spent += self.price * self.consumed_electricity

        self.price = self.generate_new_price()

        self.days_survived += 1

        # Each survived day is rewarded.
        # Temperatures lower than the baseline are penalized.
        reward = 1 + -abs(self.baseline_temperature - self.temperature)

        extra_info = {}
        extra_info['money_spent'] = self.money_spent
        extra_info['days_survived'] = self.days_survived

        done = False
        if self.is_survived():
            done = True
            extra_info['done_reason'] = 'Survived'
        elif self.is_budget_exceeded():
            done = True
            extra_info['done_reason'] = 'Budget exceeded'
            reward -= 300 # A blown budget is punished severely.

        return (reward, self.get_observation(), done, extra_info)

    def is_survived(self) -> bool:
        return self.days_survived == self.days_to_survive

    def is_budget_exceeded(self) -> bool:
        return self.money_spent > self.budget

    def reset(self):
        self.temperature = self.baseline_temperature
        self.consumed_electricity = 0.0
        self.money_spent = 0.0
        self.days_survived = 0
        self.price = self.generate_new_price()

    def calculate_consumed_electricity(self) -> float:
        return self.consumption_at_baseline_temperature * math.pow(0.95, self.baseline_temperature - self.temperature)

    def generate_new_price(self) -> float:
        new_price = np.random.normal(self.price_mu, self.price_mu, 1)[0]

        if new_price < 0:
            return 0.01

        return new_price
