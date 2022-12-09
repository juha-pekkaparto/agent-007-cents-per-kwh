# agent-007-cents-per-kwh
This is just a demo to showcase the benefits of learning optimal control. Different agent types are compared to give a better idea of why a smart thermostat is a good idea.

## Thermostat environment
A custom environment was created for this demo. The environment simulates an imaginary household. The household has electricity based heating. Only the inside temperature can be changed. Each temperature degree lower than a basedline temperature of 21 lowers the electricity consumption.

The environment operates in a daily cycle: the temperature can be set once per day, and the electricity price is selected randomly each day. The household has a budget that should not be exceeded. It should, however, be spent entirely if possible. This dynamic allows higher temperatures when electricity is cheap.

## Agent types and expected results
* Baseline agent:
    * Always chooses the baseline temperature.
    * Should exceed the budget roughly 50% of the time.
* Minimizer agent:
    * Always chooses the lowest temperature.
    * Should almost never exceed the budget.
    * Should receive a large penalty from always being very far from the baseline temperature.
* Random agent:
    * Always chooses random temperatures.
    * The reward it receives varies a lot based on chance.
* Smart agent:
    * Chooses temperatures based on previous experiences.
    * Should keep the temperature as high as possible without exceeding the budget.
    * Sets the temperature lower when the price of electricity is high, and sets the temperature higher when the price is low.
    * Should have the highest average reward after several episodes.
