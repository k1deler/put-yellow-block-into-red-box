"""Select an offline candidate using arm AND gripper errors; never deploy it.

Errors retain their physical units in reports. Ratios are used only to rank
checkpoints, avoiding the addition of radians to dimensionless gripper errors.
"""
import math


OBJECTIVES = (
    'right_arm_rad', 'right_grip',
    'first_step_right_arm_rad', 'first_step_right_grip',
)
RULE = 'pareto_minimax_arm_gripper_v1'


def evaluation_reference(checkpoint, allow_training_overlap=False):
    """Require an explicit acknowledgement before evaluating full-data exports."""
    train = set(checkpoint['train_episodes'])
    names = set(checkpoint.get('val_episodes', []))
    if not names and allow_training_overlap:
        names = set(checkpoint.get('diagnostic_reference_episodes', []))
    overlap = sorted(train & names)
    if not names or (overlap and not allow_training_overlap):
        raise ValueError('No independent validation split; diagnostic overlap requires explicit acknowledgement')
    return names, dict(reference_is_held_out=not overlap,
                      reference_training_overlap=overlap,
                      evaluation_scope='held-out validation' if not overlap else 'training-overlap diagnostic')


def select_candidate(records):
    """Return the balanced Pareto candidate plus an auditable ranking.

    Minimize the worst ratio to each metric's best candidate, then the mean
    ratio. This is an offline ranking, not evidence of closed-loop success.
    The selected checkpoint must still be compared and validated separately.
    """
    if not records:
        raise ValueError('No evaluated candidates')
    values = []
    for record in records:
        row = []
        for key in OBJECTIVES:
            value = record['metrics'][key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError('Metric must be a finite nonnegative number: ' + key)
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError('Metric must be a finite nonnegative number: ' + key)
            row.append(value)
        values.append(row)
    references = [max(min(row[j] for row in values), 1e-12)
                  for j in range(len(OBJECTIVES))]
    ranking = []
    for i, (record, row) in enumerate(zip(records, values)):
        dominated = any(
            all(a <= b for a, b in zip(other, row))
            and any(a < b for a, b in zip(other, row))
            for j, other in enumerate(values) if j != i
        )
        ratios = [value / reference for value, reference in zip(row, references)]
        if not all(math.isfinite(value) for value in ratios):
            raise ValueError('Metric ratios overflowed')
        ranking.append(dict(index=i, checkpoint=str(record['checkpoint']),
                            epoch=int(record['epoch']), pareto=not dominated,
                            worst_ratio=max(ratios), mean_ratio=sum(ratios) / len(ratios)))
    eligible = [row for row in ranking if row['pareto']]
    winner = min(eligible, key=lambda row: (row['worst_ratio'], row['mean_ratio'],
                                          row['epoch'], row['checkpoint']))
    evidence = dict(rule=RULE, objectives=list(OBJECTIVES),
                    references=dict(zip(OBJECTIVES, references)), ranking=ranking,
                    selected_checkpoint=winner['checkpoint'],
                    deployment_authorized=False, robot_tested=False)
    return records[winner['index']], evidence
