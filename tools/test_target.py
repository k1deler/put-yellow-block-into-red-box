import json
from pathlib import Path
import pytest
from register_model import identify


def test_only_yellow_red_task_accepted():
    good=dict(format='yam_train/act_ref-v1',task='put_block_in_bowl',action_mode='delta')
    assert identify(good,'put_block_in_bowl')['format']=='act_ref'
    for task in ('put_blue_block_in_blue_cup','put_banana_in_bowl'):
        with pytest.raises(ValueError):identify(good,task)


@pytest.mark.parametrize('payload',[
    dict(format='yam_train/act_ref-v1',task='put_blue_block_in_blue_cup',action_mode='delta'),
    dict(format='yam_train/act_ref-v1',task='put_block_in_bowl',action_mode='abs'),
    dict(format='kideler-task1-act-v1',task='put_block_in_bowl'),
    dict(tasks=['put_block_in_bowl'],action_mode='delta'),
    dict(format='yam_train/act_ref-v1',task='put_block_in_bowl',action_mode='delta',reproduction_test_only=True)
])
def test_wrong_models_and_synthetic_models_rejected(payload):
    with pytest.raises(ValueError):identify(payload,'put_block_in_bowl')


def test_profile_distinguishes_prior_server_evidence_from_github_default():
    root=Path(__file__).resolve().parents[1]
    profile=json.loads((root/'target_task.json').read_text(encoding='utf-8'))
    assert profile['task']=='put_block_in_bowl'
    assert profile['expected_sha256'] is None
    assert 'registered_model.json' in profile['checksum_policy']
    assert profile['execution']['horizon']==60
    assert profile['episode']['max_seconds']==40
    assert 'GitHub' in profile['identity_evidence']
