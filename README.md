# agent-007-cents-per-kwh
This is just a demo to showcase the benefits of learning optimal control. Different agent types are compared to give a better idea of why a smart thermostat is a good idea.

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
