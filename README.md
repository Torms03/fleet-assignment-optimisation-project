# Aircraft Assignment Cost Minimization

This project involves optimizing the assignment of aircraft to flights for British Airways (BA) on 1st September 2019. The goal is to minimize the total cost of assigning aircraft to flights while ensuring that demand is met, each flight gets assigned exactly one aircraft, and the fleet size is respected.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dependencies](#dependencies)


## 1. Project Overview

The project focuses on optimizing aircraft assignments to minimize the total cost using the `PuLP` library for linear programming. It considers:
- Fleet availability based on the 2019 BA fleet
- Aircraft cost data as per Eurocontrol statistics
- Simulated demand based on aircraft capacity

The solution will aim to ensure that:
- Each flight is assigned exactly one aircraft.
- The aircraft meets the simulated demand.
- Aircraft type availability is not exceeded.
- Flights are able to fly the distance required
- The total assignment cost is minimized.

## 2. Dependencies

To run these code, you'll need the following Python libraries:
- `numpy`
- `pandas`
- `pulp`
- `deap`

You can install them using `pip`:

```bash
pip install numpy pandas pulp deap
