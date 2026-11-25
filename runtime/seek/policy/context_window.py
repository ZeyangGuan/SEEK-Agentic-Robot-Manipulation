from copy import deepcopy
from pathlib import Path
import hashlib
import json
import os
import re
import time

from .context_view import _dictionary_delta

INPUT_BUDGET = 300_000
IMAGE_ALLOWANCE = 16_384
GUIDANCE = '''
CONTEXT REPRESENTATION (applies to each JSON input text or tool result separately):
An object with only "$same" is an exact key/index path to an earlier value in
THAT JSON block. Resolve it before interpreting the value; it is not missing data.
Within a boundary case, actual_following.before/after with "$observation" index
that case's observation_sequence. Its first entry has a complete snapshot;
later entries have base and delta: set replaces/adds, update changes nested keys,
remove deletes keys. Null remains a value. Every action and distinct measurement
is retained by this encoding; it does not label task effects or success.
An object with "$read" instead denotes an exactly SAVED historical value. Inactive
operation source and historical receipts are indexed on every request, even below
the size trigger. Operation names, descriptions and version IDs remain visible;
read a source before editing it or when its parameter contract is needed. The
active operation source remains direct. Complete immutable operation sources read
or authored during this episode also remain direct across later decisions. This
working set does not retain old scene parameters or receipts as current facts.
preview is navigation, not complete evidence.
For a stored operation's source, its enclosing operation id is an evidence_id:
recall_experience(query="", evidence_ids=[<operation id>, ...], limit=1..12)
returns complete operation records. Use this when you need the full executable
parameter contract; a small context_ref string page may stop before decide().
Related operation and demonstration IDs can be requested together in one call.
Read it with recall_experience(query="", evidence_ids=[], limit=1..12,
context_ref=<the $read value>, path=[], offset=0). path navigates exact object
keys/list indexes; query can locate descriptive records inside a list. The reply
reports original indexes/keys, coverage and next_offset. A string page uses
character offsets. No match is no evidence. All archived values remain available.
To read related references/pages together, supply context_reads=[{context_ref,
path,offset,limit,query}, ...] instead of context_ref/path/offset. Only context_ref
is required in each item; the top-level query must be empty and evidence_ids=[].
Each item returns its own exact result, coverage and continuation offset. Batch
up to 12 related pages in one lookup response; do not fetch unrelated history.
The indexing threshold never shortens the pages you requested or hides lookup
results retained in this decision's tool history, including earlier related reads.
Use path/offset/limit to choose the evidence pages you need.
Prefer currently relevant evidence; do not read every page just to fill context.
Missing historical detail is unknown, never a negative outcome or absent skill.
These references are a model-input representation only: local decide/observe,
recall_experience, saved memory and physical execution still receive their full values.
'''


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def tokens(text):
    import tiktoken
    return len(tiktoken.get_encoding('o200k_base').encode(text, disallowed_special=()))


def encode(value):
    result = deepcopy(value)

    def sequences(node):
        if isinstance(node, dict):
            rows = node.get('actual_following')
            if isinstance(rows, list) and rows:
                observations, sequence, indices = [], [], {}
                for row in rows:
                    for side in ('before', 'after'):
                        snapshot = row.get(side)
                        if not isinstance(snapshot, dict):
                            continue
                        signature = json.dumps(snapshot, sort_keys=True, ensure_ascii=False)
                        if signature not in indices:
                            index = len(observations)
                            sequence.append(dict(snapshot=deepcopy(snapshot)) if not observations else
                                dict(base=index-1, delta=_dictionary_delta(observations[-1], snapshot)))
                            indices[signature] = index
                            observations.append(snapshot)
                        row[side] = {'$observation': indices[signature]}
                node['observation_sequence'] = sequence
            for child in node.values():
                sequences(child)
        elif isinstance(node, list):
            for child in node:
                sequences(child)
    sequences(result)
    shared = {}

    def share(node, path):
        if isinstance(node, (dict, list, str)):
            signature = dumps(node)
            if len(signature) >= 80:
                if signature in shared:
                    ref = {'$same': shared[signature]}
                    if len(dumps(ref)) < len(signature):
                        return ref
                else:
                    shared[signature] = path
        if isinstance(node, dict):
            return {k: share(v, path+[k]) for k,v in node.items()}
        if isinstance(node, list):
            return [share(v, path+[i]) for i,v in enumerate(node)]
        return node
    return share(result, [])


def _summary(value):
    if not isinstance(value, dict):
        return dict(type=type(value).__name__, length=len(value) if isinstance(value,(str,list)) else None)
    keys = ('id','name','episode','origin_episode','step','start','end','status','decision','description','source','when')
    return {k:value[k] for k in keys if k in value and
            (value[k] is None or isinstance(value[k],(bool,int,float)) or
             isinstance(value[k],str) and len(value[k]) <= 200)}


def decode(value, archive=None):
    def expand(node):
        if isinstance(node,dict):
            if set(node)=={'$same'}:
                target=value
                for key in node['$same']:
                    target=target[key]
                return expand(target)
            if '$read' in node and archive is not None:
                return archive.get(node['$read'])
            return {k:expand(v) for k,v in node.items()}
        if isinstance(node,list):
            return [expand(v) for v in node]
        return node
    def apply(base,delta):
        result=deepcopy(base)
        for key in delta.get('remove',[]):
            del result[key]
        result.update(deepcopy(delta.get('set',{})))
        for key,change in delta.get('update',{}).items():
            result[key]=apply(result[key],change)
        return result
    def sequences(node):
        if isinstance(node,dict):
            if 'observation_sequence' in node:
                observations=[]
                for entry in node.pop('observation_sequence'):
                    observations.append(deepcopy(entry['snapshot']) if 'snapshot' in entry else
                        apply(observations[entry['base']],entry['delta']))
                for row in node['actual_following']:
                    for side in ('before','after'):
                        if isinstance(row.get(side),dict) and set(row[side])=={'$observation'}:
                            row[side]=deepcopy(observations[row[side]['$observation']])
            for child in node.values():
                sequences(child)
        elif isinstance(node,list):
            for child in node:
                sequences(child)
    result=expand(value)
    sequences(result)
    return result


class ContextArchive:
    def __init__(self, directory):
        self.directory = Path(directory)

    def put(self, value):
        raw = dumps(value).encode()
        identity = hashlib.sha256(raw).hexdigest()
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory/(identity+'.json')
        if not path.exists():
            temporary = path.with_suffix('.tmp')
            temporary.write_bytes(raw)
            temporary.replace(path)
        marker = {'$read': identity, 'type':type(value).__name__, 'length':len(value),
                  'scope':'Exact value saved; preview is an index, not execution evidence'}
        if isinstance(value, list):
            indexes = sorted(set(range(min(3,len(value)))) | set(range(max(0,len(value)-3),len(value))))
            marker['preview'] = [dict(index=i, **_summary(value[i])) for i in indexes]
        elif isinstance(value, dict):
            marker['preview_keys'] = list(value)[:12]
            marker['preview'] = _summary(value)
        return marker

    def get(self, identity):
        if not isinstance(identity,str) or not re.fullmatch('[0-9a-f]{64}',identity):
            raise ValueError('context_ref must be a supplied SHA256 reference')
        return json.loads((self.directory/(identity+'.json')).read_text(encoding='utf-8'))

    def read_request(self, query='', evidence_ids=(), limit=12, filters=None,
                     context_ref=None, path=(), offset=0, context_reads=None):
        if context_reads is None:
            return self.read(context_ref,path,offset,limit,query,evidence_ids,filters)
        if context_ref is not None or path or offset or query or evidence_ids or filters:
            raise ValueError('context_reads cannot be mixed with other selectors; use empty query/evidence_ids')
        if not isinstance(context_reads,list) or not 1<=len(context_reads)<=12:
            raise ValueError('context_reads requires 1..12 page requests')
        if type(limit) is not int or not 1<=limit<=12:
            raise ValueError('limit must be 1..12')
        results=[]
        for request in context_reads:
            if (not isinstance(request,dict) or 'context_ref' not in request or
                    set(request)-{'context_ref','path','offset','limit','query'}):
                raise ValueError('Each context read requires context_ref and optional path/offset/limit/query')
            selected=dict(path=[],offset=0,limit=limit,query='')
            selected.update(deepcopy(request))
            try:
                result=self.read(**selected)
            except (ValueError,TypeError,KeyError,IndexError,FileNotFoundError) as error:
                result=dict(error=str(error),executed=False)
            results.append(dict(request=selected,result=result))
        return dict(context_results=results,executed=False,
                    scope='Exact requested context pages; each result has independent coverage')

    def read(self, context_ref, path=(), offset=0, limit=12, query='', evidence_ids=(), filters=None):
        if not isinstance(query,str) or len(query)>500:
            raise ValueError('query must be a string of at most 500 characters')
        if evidence_ids or filters:
            raise ValueError('Use context_ref/path or saved-record evidence_ids/filters, not both')
        if not isinstance(path,(list,tuple)) or any(type(k) not in (str,int) for k in path):
            raise ValueError('path must contain exact object keys/list indexes')
        if type(offset) is not int or offset<0 or type(limit) is not int or not 1<=limit<=12:
            raise ValueError('offset must be nonnegative and limit must be 1..12')
        value = self.get(context_ref)
        for key in path:
            if isinstance(value,list) and (type(key) is not int or key<0):
                raise ValueError('List paths require nonnegative integer indexes')
            value = value[key]
        result = dict(context_ref=context_ref,path=list(path),type=type(value).__name__,executed=False)
        if isinstance(value, (list,dict)):
            items = list(enumerate(value)) if isinstance(value,list) else list(value.items())
            terms = set(re.findall(r'\w+',query.lower()))
            if terms:
                scored = [(len(terms & set(re.findall(r'\w+',dumps(v).lower()))),k,v) for k,v in items]
                items = [(k,v) for score,k,v in sorted(scored,key=lambda r:-r[0]) if score]
            chosen = items[offset:offset+limit]
            result.update(total=len(value),matching=len(items),offset=offset,
                records=[dict(index=k,value=deepcopy(v)) for k,v in chosen],
                next_offset=offset+len(chosen) if offset+len(chosen)<len(items) else None,
                complete=offset==0 and len(chosen)==len(items))
        elif isinstance(value,str):
            end = min(len(value), offset+limit*2048)
            result.update(total=len(value),offset=offset,text=value[offset:end],
                          next_offset=end if end<len(value) else None,complete=offset==0 and end==len(value))
        else:
            result.update(value=value,complete=True)
        return result


def request_size(payload):
    total = tokens(payload.get('instructions','')) + tokens(dumps(payload.get('tools',[])))
    images = 0
    for message in payload.get('input',[]):
        for part in message.get('content',[]):
            if part.get('type') == 'input_text':
                total += tokens(part['text'])
            elif part.get('type') == 'input_image':
                images += 1
        for key in ('output','arguments'):
            if isinstance(message.get(key),str):
                total += tokens(message[key])
    return dict(text_tokens=total, image_count=images,
        image_allowance_tokens=images*IMAGE_ALLOWANCE,
        estimated_input_tokens=total+images*IMAGE_ALLOWANCE+1024,
        estimator='o200k_base text estimate + 16384 allowance/image + 1024 envelope; not billed usage')


_PINNED = {'robot','features','roles','last_execution','last_error',
           'source','official_task_prompt','task','native_result','validation_feedback',
           'mechanism_state','effects','commanded_grippers','experience_access','task_lessons',
           'operations'}
_ARCHIVE = {'flows','prior_episode_results','previous_results','program_versions','operation_versions',
            'program_abis','improvement_history'}


def _candidates(value, path=()):
    if isinstance(value,dict):
        if '$read' in value:
            return
        for key,child in value.items():
            if key in _PINNED:
                continue
            child_path = path+(key,)
            if isinstance(child,(dict,list,str)) and len(dumps(child))>4096:
                priority = 0 if key in _ARCHIVE else 1 if key=='actual_following' else 2
                if key in _ARCHIVE or isinstance(child,(list,str)):
                    yield priority,-len(dumps(child)),child_path,child
            yield from _candidates(child,child_path)
    elif isinstance(value,list):
        for i,child in enumerate(value):
            yield from _candidates(child,path+(i,))


def operation_source_ids(value, versions):
    by_source = {}
    for record in versions:
        by_source.setdefault(record['source'], set()).add(record['id'])
    found = set()
    def visit(node):
        if isinstance(node, dict):
            for key, child in node.items():
                if key == 'source' and isinstance(child, str):
                    found.update(by_source.get(child, ()))
                elif key == 'text' and node.get('complete') is True and node.get('offset', 0) == 0 and isinstance(child, str):
                    found.update(by_source.get(child, ()))
                elif isinstance(child, (dict, list)):
                    visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)
    visit(value)
    return found


def _index_history(raw, archive, block, deferred, source_working_set=()):
    if not isinstance(raw, dict):
        return

    def save(parent, key, path):
        value = parent.get(key) if isinstance(parent, dict) else parent[key]
        if not isinstance(value, (dict, list, str)) or not value or (
                isinstance(value, dict) and ('$read' in value or '$same' in value)):
            return
        marker = archive.put(value)
        parent[key] = marker
        deferred.append(dict(block=block, path=list(path), reference=marker['$read'],
                             length=marker['length'], reason='persistent_history_index'))

    ctx = raw.get('context')
    if isinstance(ctx, dict):
        save(ctx, 'experience', ('context', 'experience'))
        history = ctx.get('history')
        if isinstance(history, list):
            for i in range(max(0, len(history)-1)):
                save(history, i, ('context', 'history', i))
        operations = ctx.get('operation_knowledge')
        if isinstance(operations, dict):
            active = raw.get('active_program') or operations.get('active') or {}
            active_ids = ({active.get('id'), active.get('version')} - {None}) | set(source_working_set)
            for i, record in enumerate(operations.get('operations', [])):
                if isinstance(record, dict) and record.get('id') not in active_ids:
                    save(record, 'source', ('context', 'operation_knowledge', 'operations', i, 'source'))
            save(operations, 'flows', ('context', 'operation_knowledge', 'flows'))
    for name in ('prior_episode_results', 'previous_results', 'improvement_history'):
        save(raw, name, (name,))
    agenda = raw.get('learning_agenda')
    if isinstance(agenda, dict):
        current = agenda.get('current_episode') or {}
        evidence = current.get('boundary_evidence') or {}
        for i, case in enumerate(evidence.get('cases', [])):
            if isinstance(case, dict):
                save(case, 'actual_following', ('learning_agenda', 'current_episode',
                     'boundary_evidence', 'cases', i, 'actual_following'))


def bound_payload(payload, archive, budget=None, *, source_working_set=()):
    started = time.perf_counter()
    budget = int(os.environ.get('SEEK_CONTEXT_INPUT_BUDGET',INPUT_BUDGET)) if budget is None else budget
    result = deepcopy(payload)
    if GUIDANCE not in result.get('instructions',''):
        result['instructions'] = result.get('instructions','')+'\n'+GUIDANCE
    original = request_size(result)
    blocks = []
    protected_blocks = set()
    for message in result.get('input',[]):
        texts = [(p,'text') for p in message.get('content',[]) if p.get('type')=='input_text']
        if message.get('type')=='function_call_output':
            texts.append((message,'output'))
        for parent,key in texts:
            try:
                raw = json.loads(parent[key])
            except (ValueError,TypeError):
                continue
            if isinstance(raw,(dict,list)):
                if message.get('type')=='function_call_output':
                    protected_blocks.add(len(blocks))
                blocks.append((parent,key,raw))
    def render():
        for parent,key,raw in blocks:
            parent[key] = dumps(encode(raw))
        return request_size(result)
    current = render()
    deduplicated = deepcopy(current)
    deferred = []
    for i, (_, _, raw) in enumerate(blocks):
        if i not in protected_blocks:
            _index_history(raw, archive, i, deferred, source_working_set)
    current = render()
    while current['estimated_input_tokens']>budget:
        options = [(priority,size,i,path,value) for i,(_,_,raw) in enumerate(blocks) if i not in protected_blocks
                   for priority,size,path,value in _candidates(raw)]
        if not options:
            break
        _,_,index,path,value = min(options,key=lambda r:(r[0],r[1],r[2],str(r[3])))
        marker = archive.put(value)
        parent = blocks[index][2]
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = marker
        deferred.append(dict(block=index,path=list(path),reference=marker['$read'],length=marker['length']))
        current = render()
    return result,dict(before=original,after_deduplication=deduplicated,after=current,
        index_trigger_tokens=budget,deferred=deferred,elapsed_s=time.perf_counter()-started,
        above_index_trigger=current['estimated_input_tokens']>budget,
        raw_memory_changed=False,
        persistent_history_index=True,
        source_working_set=sorted(source_working_set),
        scope='Always index inactive sources and historical receipts; preserve current evidence, active source and all results requested in this decision. Additional historical indexing above the size trigger.')
