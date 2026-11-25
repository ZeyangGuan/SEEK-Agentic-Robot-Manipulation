from copy import deepcopy
import json
import re

LOOKUP_LIMIT = 2
CONTEXT_LOOKUP_LIMIT = 2
RECORD_TYPES = (('transition','transitions'),('program','programs'),('proposal','teacher_updates'),
                ('error','errors'),('motion_evidence','motion_evidence'),('effect','effects'),('plan','plans'),
                ('boundary','learning_boundaries'))
RECORD_KINDS = tuple(kind for kind,_ in RECORD_TYPES) + ('control_method','control_invocation','operation','operation_invocation','operation_flow','review','lesson')
GUIDANCE = '''
EXPERIENCE ACCESS AND LEARNING:
recall_experience(query,evidence_ids,limit,filters) reads stored evidence without physical action.
filters is optional for existing local programs. When the request specifies source episodes or
record types, put them in filters.episodes and filters.record_types, not in the query text.
Each supplied list is an allowlist; both filters and exact evidence_ids are intersected BEFORE
text ranking. No matches means an empty result, never a fallback to proposals or other episodes.
Within the requested limit, exact-ID lookup first covers distinct eligible requested IDs in
request order, then returns additional ranked records/revisions. unreturned_requested_ids
reports requested IDs not represented in this page; unknown/excluded IDs are reported separately.
Use record_types=['transition'] for actual execution receipts, ['proposal'] for model proposals,
or ['effect'] for recorded effect assessments. Query is only positive descriptive search terms;
words such as 'episode 1', 'not proposals', or 'successful' are NOT structural constraints.
For a successful grasp, inspect the relevant effect assessment, its revisions and cited evidence,
then retrieve its exact transition/observation IDs with record_types=['transition'] and the
source episode. A feasible command or a text match is not a confirmed grasp. Effect assessments
remain revisable model judgments (native_truth=false), not native success labels. If evidence
is missing or refuted, report that rather than substituting a proposal or inventing success.
At one physical observation/review, the teacher/reviewer may use at most two ordinary
experience lookup/probe responses and, separately, two deferred-context read responses.
context_ref or context_reads use the deferred-context allowance. Group related pages
with context_reads to avoid repeated requests. Each model response is counted and paid;
these are read-response allowances, not limits on transport retries.
The same function is available inside decide/observe with no model call.
Use exact transition/observation/program/proposal/error IDs or a short query about a method,
condition or failure. Returned raw state is a historical hypothesis, not a current-scene fact.
Saved review and lesson IDs are also retrievable and valid citations. A boundary ID
refers to its recorded observation, teacher proposal and actual receipts; it is not
a success label. Referencing a prior lesson preserves its hypothesis and evidence,
not an independent physical confirmation.
Complete raw transitions remain available through recall_experience. A recorded trajectory is not a learned
controller. When evidence supports a repeatable local response, author a small decide program
that generates an action from current features/parameters, checks applicability and observed
stopping conditions, and asks on missing evidence. Recorded responses do not automatically
generate actions or establish task success. Cached teacher rows do not count as autonomous control.
Do not invent code or unsupported goals merely to increase the autonomous-action count.
Use current state parameters instead of permanent layout coordinates. Explain the evidence
or missing capability when a reusable controller cannot yet be justified. These improvements
can be supplied in an ordinary act; no separate training or preparation request is required.
'''


def recall_tool():
    tool = dict(type='function', name='recall_experience', strict=False,
        description='Read exact saved SEEK evidence; no motion or automatic model request.',
        parameters=dict(type='object', additionalProperties=False,
            properties=dict(query=dict(type='string', maxLength=500),
                evidence_ids=dict(type='array', maxItems=8, items=dict(type='string')),
                limit=dict(type='integer', minimum=1, maximum=12),
                filters=dict(type='object', additionalProperties=False,
                    description='Hard allowlists applied before ranking and intersected with evidence_ids. Omit a field to leave that dimension unrestricted. No fallback on an empty result.',
                    properties=dict(
                        episodes=dict(type='array', minItems=1, items=dict(type='integer',minimum=0),
                            description='Source episode numbers, not the current query episode.'),
                        record_types=dict(type='array', minItems=1, items=dict(type='string',enum=list(RECORD_KINDS)),
                            description='Use transition for execution receipts; proposal is not executed evidence.')))),
            required=['query','evidence_ids','limit']))
    tool['parameters']['properties'].update(
        context_ref=dict(type='string',description='Optional $read reference supplied in this request. Reads exact deferred context instead of searching memory.'),
        path=dict(type='array',items=dict(type=['string','integer']),description='Exact keys/indexes within context_ref; default [].'),
        offset=dict(type='integer',minimum=0,description='Page offset within context_ref/path; default 0.'),
        context_reads=dict(type='array',minItems=1,maxItems=12,
            description='Read related deferred context pages together, instead of context_ref/path/offset. Use query="" and evidence_ids=[].',
            items=dict(type='object',additionalProperties=False,required=['context_ref'],properties=dict(
                context_ref=dict(type='string'),
                path=dict(type='array',items=dict(type=['string','integer'])),
                offset=dict(type='integer',minimum=0),
                limit=dict(type='integer',minimum=1,maximum=12),
                query=dict(type='string',maxLength=500)))))
    return tool


def validate_lookup_arguments(args):
    required={'query','evidence_ids','limit'}
    if not isinstance(args,dict) or not required<=set(args) or set(args)-required-{'filters','context_ref','path','offset','context_reads'}:
        raise ValueError('Lookup requires query, evidence_ids, limit and optional filters')
    if 'filters' in args and args['filters'] is None:
        raise ValueError('filters must be an object; omit it to leave records unrestricted')
    if ('path' in args or 'offset' in args) and 'context_ref' not in args:
        raise ValueError('path/offset requires context_ref')
    if 'context_reads' in args and (set(args)&{'context_ref','path','offset','filters'} or args['query'] or args['evidence_ids']):
        raise ValueError('context_reads requires empty query/evidence_ids and no other selectors')
    return args


def is_context_lookup(args):
    return isinstance(args,dict) and ('context_ref' in args or 'context_reads' in args)


def validate_filters(filters):
    if filters is None:return {}
    if not isinstance(filters,dict) or set(filters)-{'episodes','record_types'}:
        raise ValueError('filters supports only episodes and record_types')
    for field,values in filters.items():
        if not isinstance(values,list) or not values:
            raise ValueError('filters.'+field+' must be a nonempty allowlist; omit the field for no restriction')
        if field=='episodes' and any(type(v) is not int or v<0 for v in values):
            raise ValueError('filters.episodes requires nonnegative integer episode numbers')
        if field=='record_types' and any(not isinstance(v,str) or v not in RECORD_KINDS for v in values):
            raise ValueError('Unknown record type in filters.record_types')
    return deepcopy(filters)


def record_episode(kind,row):
    field='origin_episode' if kind in ('program','operation') else 'episode'
    if field in row:
        value=row[field]
        return value if type(value) is int and value>=0 else None
    for field in ('id','observation_id','anchor_observation_id'):
        value=row.get(field)
        match=re.match(r'^e([0-9]+):',value) if isinstance(value,str) else None
        if match:return int(match.group(1))
    return None


def available_ids(memory):
    known={r['id'] for _,key in RECORD_TYPES for r in memory.get(key,[])}
    known.update(t[side]['id'] for t in memory.get('transitions',[]) for side in ('before','after'))
    known.update(r['observation_id'] for r in memory.get('motion_evidence',[]) if 'observation_id' in r)
    control=memory.get('control_experience',{})
    known.update(control.get('methods',{}))
    known.update(r['id'] for r in control.get('invocations',[]))
    operations=memory.get('operation_knowledge',{})
    known.update(r['id'] for key in ('versions','invocations','flows') for r in operations.get(key,[]))
    known.update(row['id'] for _,row in learning_records(memory))
    return known


def learning_records(memory):
    for review in memory.get('improvement_history',[]):
        yield 'review',review
        for lesson in review.get('lessons',[]):
            yield 'lesson',lesson


def lookup(memory, query, evidence_ids, limit, errors=(), *, filters=None):
    if (not isinstance(query,str) or len(query)>500 or not isinstance(evidence_ids,list)
            or len(evidence_ids)>8 or any(not isinstance(x,str) for x in evidence_ids)
            or type(limit) is not int or not 1<=limit<=12):
        raise ValueError('Lookup requires query <=500 characters, <=8 evidence IDs and integer limit 1..12')
    filters=validate_filters(filters)
    records=[]
    for kind,key in RECORD_TYPES:
        for row in memory.get(key,[]):
            aliases={row['id']}
            if kind=='transition': aliases.update(row[s]['id'] for s in ('before','after'))
            if kind=='motion_evidence' and 'observation_id' in row:aliases.add(row['observation_id'])
            records.append((kind,row,aliases))
    for kind,row in learning_records(memory):
        records.append((kind,row,{row['id']}))
    stored_errors={r['id'] for r in memory.get('errors',[])}
    for row in errors:
        if row['id'] not in stored_errors:records.append(('error',row,{row['id']}))
    control=memory.get('control_experience',{})
    for row in control.get('methods',{}).values():records.append(('control_method',row,{row['id']}))
    for row in control.get('invocations',[]):records.append(('control_invocation',row,{row['id']}))
    operations=memory.get('operation_knowledge',{})
    for kind,key in (('operation','versions'),('operation_invocation','invocations'),('operation_flow','flows')):
        for row in operations.get(key,[]):records.append((kind,row,{row['id']}))
    requested=set(evidence_ids)
    available=set().union(*(r[2] for r in records)) if records else set()
    eligible=[]
    for i,(kind,row,aliases) in enumerate(records):
        if 'record_types' in filters and kind not in filters['record_types']:continue
        if 'episodes' in filters and record_episode(kind,row) not in filters['episodes']:continue
        eligible.append((i,kind,row,aliases))
    eligible_ids=set().union(*(r[3] for r in eligible)) if eligible else set()
    tokens=set(re.findall(r'\w+',query.lower()))
    ranked=[]
    for i,kind,row,aliases in eligible:
        if requested and not requested.intersection(aliases): continue

        words=set(re.findall(r'\w+',json.dumps(row,ensure_ascii=False).lower()))
        score=len(tokens.intersection(words))
        if tokens and not requested and not score: continue
        ranked.append((score,i,kind,row,aliases))
    ranked.sort(key=lambda x:(-x[0],-x[1]))
    selected=[];covered=set()

    for identity in dict.fromkeys(evidence_ids):
        if len(selected)>=limit:break
        if identity in covered:continue
        match=next((item for item in ranked if identity in item[4]),None)
        if match is not None:
            selected.append(match);covered.update(match[4]&requested)
    selected_indices={item[1] for item in selected}
    selected.extend(item for item in ranked if item[1] not in selected_indices)
    selected=selected[:limit]
    for item in selected:covered.update(item[4]&requested)
    return dict(records=[dict(kind=kind,evidence=deepcopy(row)) for _,_,kind,row,_ in selected],
        matching_records=len(ranked),omitted_records=max(0,len(ranked)-limit),
        unknown_ids=sorted(requested-available),executed=False,
        returned_requested_ids=sorted(covered),unreturned_requested_ids=sorted(requested-covered),
        applied_filters=filters,eligible_records=len(eligible),
        filter_excluded_ids=sorted((requested&available)-eligible_ids),
        scope='Exact saved evidence; historical beliefs and commanded actions are not verified task effects')


def error_records(rows,episode):
    return [dict(row,id=row.get('id',f'e{episode}:error{i+1}')) for i,row in enumerate(rows)]


def attach_teacher_evidence(transitions,teachers,episode):
    result=[];index=0;current=None
    for row in transitions:
        while index<len(teachers) and teachers[index]['step']<=row['start']:
            current=teachers[index];index+=1
        item=deepcopy(row)
        if current is not None and row.get('source') in ('model','astra','teacher_buffer'):
            item.setdefault('teacher_proposal_id',f'e{episode}:teacher{current["turn"]}')
            item.setdefault('execution_note',current.get('execution_note',''))
        result.append(item)
    return result
