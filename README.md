# Aircraft Assignment Cost Minimization

This project involves optimizing the assignment of aircraft to flights for British Airways (BA) on 1st September 2019. The goal is to minimize the total cost of assigning aircraft to flights while ensuring that demand is met, each flight gets assigned exactly one aircraft, and the fleet size is respected.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dependencies](#dependencies)
3. [Data Input](#data-input)
4. [Optimization Problem Setup](#optimization-problem-setup)
5. [Solving the Problem](#solving-the-problem)
6. [Results](#results)

## Project Overview

The project focuses on optimizing aircraft assignments to minimize the total cost using the `PuLP` library for linear programming. It considers:
- Fleet availability based on the 2019 BA fleet
- Aircraft cost data as per Eurocontrol statistics
- Simulated demand based on aircraft capacity

The solution will ensure that:
- Each flight is assigned exactly one aircraft.
- The aircraft meets the simulated demand.
- Aircraft type availability is not exceeded.
- The total assignment cost is minimized.

## Dependencies

To run this code, you'll need the following Python libraries:
- `numpy`
- `pandas`
- `pulp`

You can install them using `pip`:

```bash
pip install numpy pandas pulp
