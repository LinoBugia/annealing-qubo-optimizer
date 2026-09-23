"""
Skript_Solve_Random_QUBO — demo: batched DA from several start vectors.

Shows the multi-start mode: given a list of start vectors, num_MC
Monte-Carlo trials are run for each of them. Visualisation and CSV output
follow the reference library (Runs/ folder).
"""

from Funcs_Qubo_ProblemGeneration import (create_random_qubo_signed,
                                          random_start_states)
from Funcs_Qubo_Optimizers import qubo_min_solver

if __name__ == "__main__":
    n = 500
    steps = 3000
    num_MC = 5
    seed = 42

    A, b, c = create_random_qubo_signed(n, density=0.5, seed=seed)

    # three different start vectors -> 3 groups x num_MC trials
    starts = random_start_states(n, 3, seed=seed)

    Min_varAss, Mins, Trajectories, Infos = qubo_min_solver(
        A, b, c,
        steps=steps, num_MC=num_MC,
        cooling_param=["logarithmic", 50, 0],
        seed_rand=seed,
        initial_varAssignements_pre=starts,
        offset_increase_rate="auto_gp",
        save_csv=True, save_addinfo=True, visual_inst=True)

    print("best minimum: %.6g (trial %s)"
          % (Infos["E_best"], Infos["labels"][Infos["best"]]))
    per_group = {g: min(m for m, gg in zip(Mins, Infos["group_of_trial"])
                        if gg == g)
                 for g in sorted(set(Infos["group_of_trial"]))}
    print("minimum per start group:", {k: round(v, 4) for k, v in per_group.items()})
