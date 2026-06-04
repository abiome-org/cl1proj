Goal: Measure how a trained neural task degrades over time, how quickly it returns to normal after retraining, and which noise or perturbation pattern produces the best forgetting/recovery profile.


Basic Experiment Outline

1. Define the task

Choose a simple closed-loop task with a measurable score.

Example task:

Stimulus A should produce response pattern A.
Stimulus B should produce response pattern B.

The score could be decoder accuracy, response separability, reward rate, latency, or similarity to a target firing pattern.


2. Record baseline state

  Open a CL1 session.

  Start a recording.

  Measure spontaneous activity before training.

  Run a short untrained evaluation block.

Save:

  Baseline firing statistics.
  Baseline task score.
  Baseline burst structure.
  Baseline responsiveness to simple probe stimuli.


3. Train the task

Run the closed-loop training loop.

Present task stimuli.

Read neural responses.

Deliver feedback or reinforcement when the response matches the target.

Periodically run an evaluation block without reinforcement.

Continue until the task score reaches the learned threshold.

Save the training curve.


4. Confirm learning

Run a clean evaluation block.

If the score is below the learned threshold, mark the run as failed-to-learn and skip the forgetting benchmark.

If the score is above the learned threshold, continue.


5. Begin forgetting phase

Stop normal task reinforcement.

Keep recording.

Choose one forgetting condition.

Examples:

No noise.
Low random noise.
High random noise.
Pink noise.
Sparse burst noise.
Dense burst noise.
Task-like stimulation without reward.
Anti-correlated stimulation.


6. Poll with exponential backoff

Evaluate the task after increasingly long delays.

Example delay sequence:

10 seconds.
20 seconds.
40 seconds.
80 seconds.
160 seconds.
320 seconds.
640 seconds.
1280 seconds.
2560 seconds.
5120 seconds.

Between evaluation points, apply the selected noise condition.

At each poll:

Pause or isolate the perturbation if needed.
Run a task evaluation block.
Measure task score.
Measure general culture health.
Log elapsed time, condition, score, firing statistics, and burst statistics.

Stop polling when:

The task score falls below the forgetting threshold.
The maximum forgetting window is reached.
The culture health metrics leave the acceptable range.


7. Measure recovery

Resume normal task training.

Train in short blocks.

After each block, run a clean evaluation.

Continue until:

The task score returns above the recovery threshold.
The maximum recovery time is reached.

Save the recovery curve.


8. Summarize one run

For each run, calculate:

Initial baseline score.
Post-training learned score.
Time to forgetting.
Forgetting half-life.
Area under the retention curve.
Minimum task score during forgetting.
Time to recovery.
Recovery slope.
Final recovered score.
Change in spontaneous firing rate.
Change in burst structure.
Change in responsiveness to probe stimuli.


9. Repeat across methods

For each noise or perturbation method:

Run multiple independent repeats.

Use the same task definition.

Use the same training threshold.

Use the same forgetting threshold.

Use the same recovery threshold.

Randomize condition order when possible.

Include no-noise controls.


10. Rank methods

If the goal is memory preservation:

Prefer the method with the largest area under the retention curve and smallest degradation.

If the goal is controlled forgetting:

Prefer the method with the shortest time to forgetting while preserving culture health.

If the goal is reversible forgetting:

Prefer the method with fast forgetting, low health disruption, and fast recovery.

If the goal is robust retraining:

Prefer the method with the highest final recovered score and steepest recovery slope.


Final Output

For each condition, produce:

A forgetting curve.
A recovery curve.
A health-metric curve.
A summary table.
A ranked list of perturbation methods.

The best method depends on the objective:

Best preservation.
Fastest forgetting.
Fastest reversible forgetting.
Best recovery after degradation.
