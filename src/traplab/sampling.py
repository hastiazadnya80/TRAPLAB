"""Registered proposal space shared by adaptive and random arms."""
import random
from .core import DEFAULT_STORIES

PARAMETER_SPACE = {
    'simpson': {'n_a0': 'integer 20..2000', 'n_a1': 'integer 20..2000', 'n_b0': 'integer 20..2000', 'n_b1': 'integer 20..2000', 'p_a0': '0.05..0.95', 'p_a1': '0.05..0.95', 'p_b0': '0.05..0.95', 'p_b1': '0.05..0.95', 'z_role': ['confounder', 'mediator', 'unstated']},
    'collider': {'px': '0.1..0.9', 'p0': '0.1..0.9', 'beta': '-0.7..0.7; p0+beta must be 0.05..0.95', 'selection': ['random', 'on_X', 'on_X_and_Y'], 's00': '0.05..1', 's01': '0.05..1', 's10': '0.05..1', 's11': '0.05..1', 'scale': 'integer 1000..100000; every displayed cell >=20; random requires all s equal; on_X requires s00=s01 and s10=s11'},
    'rtm': {'rho': '0.1..0.95', 'threshold': '0.5..2.5', 'delta': '-1..1', 'design': ['no_control', 'randomized_control'], 'n': 'integer 20..2000'},
}


def random_proposal(family, rng: random.Random, stories=DEFAULT_STORIES):
    if family == 'simpson':
        p = {f'n_{k}': rng.randint(20,2000) for k in ('a0','a1','b0','b1')}
        p.update({f'p_{k}': round(rng.uniform(.05,.95), 4) for k in ('a0','a1','b0','b1')})
        p['z_role'] = rng.choice(['confounder','mediator','unstated'])
    elif family == 'collider':
        p = {'px':round(rng.uniform(.1,.9),4), 'p0':round(rng.uniform(.1,.9),4), 'beta':round(rng.uniform(-.7,.7),4), 'scale':rng.randint(1000,100000), 'selection':rng.choice(['random','on_X','on_X_and_Y'])}
        s = [round(rng.uniform(.05,1),4) for _ in range(4)]
        if p['selection'] == 'random': s = [s[0]]*4
        if p['selection'] == 'on_X': s = [s[0],s[0],s[2],s[2]]
        p.update(dict(zip(('s00','s01','s10','s11'), s)))
    elif family == 'rtm':
        p = {'rho':round(rng.uniform(.1,.95),4), 'threshold':round(rng.uniform(.5,2.5),4), 'delta':round(rng.uniform(-1,1),4), 'design':rng.choice(['no_control','randomized_control']), 'n':rng.randint(20,2000)}
    else:
        raise ValueError(family)
    return {'family':family, 'params':p, 'story_id':rng.choice(stories)['id']}


def smoke_proposal():
    return {'family':'simpson', 'story_id':'workshop', 'params':{'n_a0':80,'n_a1':800,'n_b0':800,'n_b1':80,'p_a0':.9,'p_a1':.3,'p_b0':.8,'p_b1':.2,'z_role':'confounder'}}
