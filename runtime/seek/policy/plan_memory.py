from copy import deepcopy
from pathlib import Path
import hashlib
import json

SCHEMA = 'seek-memory'
PROGRAM_ABI = 'seek-control'
COMPATIBLE_ABIS = ('seek-plan', 'seek-control-basic', 'seek-control-tracking', 'seek-control-observed', PROGRAM_ABI)


def migrate_memory(memory):
    original = memory
    catalog = Path(__file__).resolve().parents[2] / 'setup/installed/snapshots.json'
    if catalog.is_file():
        formats = json.loads(catalog.read_text())
        if memory.get('schema') in formats['schemas']:
            incoming = deepcopy(memory)
            previous = incoming['schema']
            incoming['schema'] = dict(zip(formats['schemas'], ('seek-memory-initial', 'seek-memory-procedural', SCHEMA)))[previous]
            interfaces = dict(zip(formats['interfaces'], COMPATIBLE_ABIS))
            for key, value in incoming.get('program_abis', {}).copy().items():
                incoming['program_abis'][key] = interfaces.get(value, value)
            for record in incoming.get('programs', []) + incoming.get('operation_knowledge', {}).get('versions', []):
                if record.get('abi') in interfaces:
                    record['abi'] = interfaces[record['abi']]
            if incoming['schema'] == SCHEMA:
                return incoming
            memory = incoming
    if memory.get('schema') not in ('seek-memory-initial','seek-memory-procedural'):
        raise ValueError('Unsupported snapshot format. Install the compatibility catalog and supply the original snapshot.')
    result = deepcopy(memory)
    if memory['schema']=='seek-memory-procedural':
        result['schema']=SCHEMA
        result['control_experience']=dict(methods={},invocations=[])
        result.setdefault('migrations',[]).append(dict(source_schema=original['schema'],
            source_sha256=hashlib.sha256(json.dumps(original,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
            program_status='Compatible code retained; no scene objective or learned method invented',
            evidence_status='Raw transitions remain available to goal-conditioned fitting'))
        return result
    result.update(schema=SCHEMA, active=None, program_abis={})
    result['control_experience']=dict(methods={},invocations=[])
    result['migration'] = dict(source_schema=original['schema'],
        source_sha256=hashlib.sha256(json.dumps(original,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
        previous_active=memory.get('active'),
        historical_action_semantics='Recorded commands were relative to each actual observation; do not reconstruct a fixed teacher trajectory from them',
        program_status='Source retained for explicit interface review or reproposal; legacy buffered_actions assumptions are not implicitly reactivated',
        physical_evidence='All original transitions, numerical evidence, effects and episode results retained unchanged')
    return result


def accept_program(memory, record):
    if not any(p['id']==record['id'] for p in memory['programs']):
        memory['programs'].append(record)
    memory.setdefault('program_abis',{})[record['id']] = PROGRAM_ABI
    memory['active'] = record['id']


def require_compatible(memory, identity):
    if memory.get('program_abis',{}).get(identity) not in COMPATIBLE_ABIS:
        raise ValueError('Stored program needs explicit interface review before activation')
