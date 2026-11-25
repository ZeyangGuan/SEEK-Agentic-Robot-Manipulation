from copy import deepcopy
import math
import operator

OPS={'lt':operator.lt,'le':operator.le,'gt':operator.gt,'ge':operator.ge}
GUIDANCE='''
CANDIDATE CONTINUATION: optional candidate_guard applies to this teacher batch only.
It contains role_generations (role name -> current generation), requirements and stop_when.
Each predicate is {feature,op,value}, where op is lt/le/gt/ge on a measured numeric feature.
All requirements must hold. All nonempty stop_when predicates holding ends the batch.
Unknown referenced features, invalid/rebound roles or unmet requirements discard the suffix
and return to decide before any next action. An empty guard declares no additional predicates.
These are model-declared applicability/measurement conditions, not a verified task score
or executor limits. Choose only conditions needed by this candidate's intended action;
diagnostic variation need not become a requirement. A discarded suffix calls for a new
decision, not an automatic release or a declaration that manipulation failed.
Programs can handle richer conditions in decide; a local act replaces the teacher batch.
The guard belongs to the policy-owned candidate plan. It does not change the
fixed resolved targets, motion-query identity, receipt cursor or tracking feedback.
'''


def guard_schema():
    predicate=dict(type='object',additionalProperties=False,properties=dict(
        feature=dict(type='string'),op=dict(type='string',enum=list(OPS)),value=dict(type='number')),
        required=['feature','op','value'])
    return dict(anyOf=[dict(type='null'),dict(type='object',additionalProperties=False,
        properties=dict(role_generations=dict(type='object',additionalProperties=dict(type='integer',minimum=1)),
            requirements=dict(type='array',maxItems=16,items=predicate),
            stop_when=dict(type='array',maxItems=16,items=predicate)),
        required=['role_generations','requirements','stop_when'])])


def validate_guard(value):
    if value is None:return None
    if not isinstance(value,dict) or set(value)!={'role_generations','requirements','stop_when'}:
        raise ValueError('candidate_guard needs role_generations, requirements, stop_when')
    roles=value['role_generations']
    if not isinstance(roles,dict) or len(roles)>32 or any(not isinstance(k,str) or type(v) is not int or v<1 for k,v in roles.items()):
        raise ValueError('candidate_guard role generations must be positive integers')
    for field in ('requirements','stop_when'):
        predicates=value[field]
        if not isinstance(predicates,list) or len(predicates)>16:raise ValueError('At most 16 measured predicates per field')
        for p in predicates:
            if (not isinstance(p,dict) or set(p)!={'feature','op','value'} or not isinstance(p['feature'],str)
                    or p['op'] not in OPS or type(p['value']) not in (int,float) or not math.isfinite(p['value'])):
                raise ValueError('Measured predicate requires feature, op lt/le/gt/ge, finite value')
    return deepcopy(value)


def guard_reason(guard,ctx):
    if guard is None:return None
    for name,generation in guard['role_generations'].items():
        role=ctx['roles'].get(name,{})
        if not role.get('valid') or role.get('generation')!=generation:
            return dict(type='candidate_role_unavailable',role=name,expected_generation=generation)
    checks={}
    for field in ('requirements','stop_when'):
        checks[field]=[]
        for p in guard[field]:
            actual=ctx['features'].get(p['feature'])
            if type(actual) not in (int,float) or not math.isfinite(actual):
                return dict(type='candidate_feature_unknown',feature=p['feature'])
            checks[field].append(OPS[p['op']](actual,p['value']))
    if not all(checks['requirements']):return dict(type='candidate_requirement_changed')
    if checks['stop_when'] and all(checks['stop_when']):return dict(type='candidate_stop_condition_observed')
    return None
