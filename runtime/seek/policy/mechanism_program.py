import ast
from copy import deepcopy
import hashlib
import json
import math
import operator
import time
from .motion_query import CapabilityRequest
from .rigid_geometry import quat_matrix, quat_rotate, matvec, transform_points, box_corners, box_relation


MAX_SOURCE_CHARS = 60000


DEFAULT_CPU_BUDGET_SECONDS = 10.0


def source_schema():
    return dict(type='string', minLength=1, maxLength=MAX_SOURCE_CHARS,
        description=f'Complete source for one observe/decide program, at most {MAX_SOURCE_CHARS} characters. Reuse stored operations or split meaningful stages into named operations with kind=continue and operation; preserve useful observation and control logic.')


class ProgramError(ValueError): pass
class Returned(Exception):
    def __init__(self,value):self.value=value
class BreakLoop(Exception): pass
class ContinueLoop(Exception): pass

BIN={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,
     ast.FloorDiv:operator.floordiv,ast.Mod:operator.mod,ast.Pow:operator.pow}
CMP={ast.Eq:operator.eq,ast.NotEq:operator.ne,ast.Lt:operator.lt,ast.LtE:operator.le,
     ast.Gt:operator.gt,ast.GtE:operator.ge,ast.Is:operator.is_,ast.IsNot:operator.is_not,
     ast.In:lambda a,b:a in b,ast.NotIn:lambda a,b:a not in b}
SAFE={'abs':abs,'min':min,'max':max,'sum':sum,'len':len,'round':round,'float':float,
      'int':int,'bool':bool,'str':str,'list':list,'dict':dict,'sorted':sorted,
      'range':lambda *a:list(range(*a)) if len(range(*a))<=2048 else (_ for _ in ()).throw(ProgramError('range exceeds 2048')),
      'enumerate':lambda x:list(enumerate(x)),'zip':lambda *x:list(zip(*x)),
      'all':all,'any':any,'sqrt':math.sqrt,'sin':math.sin,'cos':math.cos,
      'atan2':math.atan2,'exp':math.exp,'log':math.log,
      'clip':lambda v,lo,hi:min(hi,max(lo,v)),
      'norm':lambda xs:math.sqrt(sum(x*x for x in xs))}
GEOMETRY_SAFE=dict(quat_matrix=quat_matrix, quat_rotate=quat_rotate, matvec=matvec,
                   transform_points=transform_points, box_corners=box_corners, box_relation=box_relation)
METHODS={'get','items','keys','values','append','extend','pop','copy','update','count','index'}
ALLOWED=(ast.Module,ast.FunctionDef,ast.arguments,ast.arg,ast.Return,ast.Assign,ast.AugAssign,
         ast.Expr,ast.If,ast.For,ast.While,ast.Break,ast.Continue,ast.Pass,ast.Name,ast.Load,ast.Store,
         ast.Constant,ast.List,ast.Tuple,ast.Dict,ast.Subscript,ast.Slice,ast.BinOp,ast.UnaryOp,
         ast.BoolOp,ast.Compare,ast.IfExp,ast.Call,ast.keyword,ast.Attribute,ast.ListComp,
         ast.DictComp,ast.comprehension,ast.Add,ast.Sub,ast.Mult,ast.Div,ast.FloorDiv,ast.Mod,
         ast.Pow,ast.USub,ast.UAdd,ast.Not,ast.And,ast.Or,*CMP)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def validate_decision(value):
    if not isinstance(value,dict):raise ProgramError('decide must return an object')
    if value.get('kind') not in ('act','cached','continue','delegate','ask','blocked'):
        raise ProgramError('decision.kind must be act, cached, continue, delegate, ask or blocked')
    if 'operation' in value and value['kind']!='continue':
        raise ProgramError('Operation transitions use continue before any execution')
    if 'state' in value:raise ProgramError('ABI 4: use state_update for current evidence and on_execution for receipt-dependent state')
    for field in ('state_update','on_execution'):
        if value.get(field) is not None and not isinstance(value[field],dict):
            raise ProgramError('decision.'+field+' must be an object/null')
    if value['kind'] in ('ask','blocked','continue','delegate') and value.get('on_execution') is not None:
        raise ProgramError('ask/blocked/continue/delegate has no owned action: on_execution must be null or absent')
    if value.get('intent') is not None:
        intent=value['intent']
        objective=intent.get('control') if isinstance(intent,dict) and intent.get('kind')=='probe' else intent
        if value['kind'] not in ('continue','delegate') or not isinstance(objective,dict) or objective.get('kind')!='control':
            raise ProgramError('Local intent requires continue/delegate and a measured control objective')
    if not isinstance(value.get('reason',''),str):raise ProgramError('decision.reason must be text')
    if value['kind']=='act':
        action=value.get('action')
        if not isinstance(action,list) or len(action)!=14 or any(type(x) not in (int,float) or not math.isfinite(x) for x in action):
            raise ProgramError('act requires one finite numeric 14-vector')
    if 'query_id' in value and (value['kind'] != 'act' or not isinstance(value['query_id'], str)):
        raise ProgramError('query_id selects a numerical solution for an act decision')

    try:
        json.dumps(value,allow_nan=False)
    except (TypeError,ValueError) as e:raise ProgramError('decision must be finite JSON: '+str(e)) from None
    return value


class Program:
    def __init__(self,source,fuel=None,cpu_budget_s=DEFAULT_CPU_BUDGET_SECONDS):
        if not isinstance(source,str) or not source.strip() or len(source)>MAX_SOURCE_CHARS:
            raise ProgramError(f'Program source must contain 1..{MAX_SOURCE_CHARS} characters')
        try:tree=ast.parse(source)
        except SyntaxError as e:raise ProgramError(f'Syntax line {e.lineno}: {e.msg}') from None
        if not math.isfinite(cpu_budget_s) or cpu_budget_s<=0:
            raise ProgramError('CPU budget must be positive and finite')
        self.functions={};self.limit=fuel;self.source=source;self.sha=digest(source)
        self.cpu_budget_s=cpu_budget_s
        for node in ast.walk(tree):
            if not isinstance(node,ALLOWED):raise ProgramError(f'Unsupported syntax {type(node).__name__} at line {getattr(node,"lineno",0)}')
            if isinstance(node,ast.Name) and node.id.startswith('_'):raise ProgramError('Private names are unavailable')
            if isinstance(node,ast.Attribute) and node.attr not in METHODS:
                raise ProgramError('Only data-container methods are available: '+node.attr)
            if isinstance(node,ast.comprehension) and node.is_async:raise ProgramError('Async is unavailable')
        for f in tree.body:
            if not isinstance(f,ast.FunctionDef):raise ProgramError('Top level must contain function definitions only')
            if f.name.startswith('_') or f.name in SAFE or f.name in ('control_status','realize_motion','realize_joint_motion','measure_patch','measure_region','measure_rgb','segment_rgb','locate_rgb_point','recall_experience'):raise ProgramError('Reserved function name')
            if f.decorator_list or f.returns or f.args.defaults or f.args.kw_defaults or f.args.kwonlyargs or f.args.vararg or f.args.kwarg or f.args.posonlyargs:
                raise ProgramError('Use plain positional function definitions without annotations/defaults')
            if any(a.annotation for a in f.args.args):raise ProgramError('Annotations unavailable')
            if f.name in self.functions:raise ProgramError('Duplicate function: '+f.name)
            self.functions[f.name]=f
        if 'decide' not in self.functions or [a.arg for a in self.functions['decide'].args.args]!=['ctx','state']:
            raise ProgramError('Required entry: def decide(ctx, state)')

    def tick(self):
        self.operations_used+=1
        if self.fuel is not None:
            self.fuel-=1
            if self.fuel<0:raise ProgramError('Mechanism computation budget exhausted')
        if self.operations_used%1024==0:self._check_cpu_budget()

    def _check_cpu_budget(self):
        if time.thread_time()>=self._cpu_deadline:
            raise ProgramError('Mechanism CPU time budget exhausted')

    def _start(self,capabilities):
        self.fuel=self.limit;self.operations_used=0;self.depth=0
        self.capabilities=capabilities or {}
        self._cpu_deadline=time.thread_time()+self.cpu_budget_s

    def run(self,ctx,state=None,capabilities=None):
        self._start(capabilities)
        try:
            value=validate_decision(self.call('decide',[deepcopy(ctx),deepcopy(state or {})]))
            self._check_cpu_budget()
            return value
        except (ProgramError, CapabilityRequest):raise
        except Exception as e:raise ProgramError(type(e).__name__+': '+str(e)[:400]) from None

    def observe(self,ctx,state=None,capabilities=None):
        if 'observe' not in self.functions:return None
        self._start(capabilities)
        try:
            value=self.call('observe',[deepcopy(ctx),deepcopy(state or {})])
            if not isinstance(value,dict) or set(value)-{'state_update','effects','reason'}:
                raise ProgramError('observe returns state_update/effects/reason, never a motion decision')
            if value.get('state_update') is not None and not isinstance(value['state_update'],dict):
                raise ProgramError('observe.state_update must be object/null')
            json.dumps(value,allow_nan=False)
            self._check_cpu_budget()
            return value
        except ProgramError:raise
        except Exception as e:raise ProgramError(type(e).__name__+': '+str(e)[:400]) from None

    def call(self,name,args):
        self.tick()
        if name in SAFE:return self.bound(SAFE[name](*args))

        if name in GEOMETRY_SAFE and name not in self.functions:
            return self.bound(GEOMETRY_SAFE[name](*args))
        if name in self.capabilities:return self.bound(self.capabilities[name](*deepcopy(args)))
        if name not in self.functions:raise ProgramError('Unknown function: '+name)
        f=self.functions[name]
        if len(args)!=len(f.args.args):raise ProgramError('Wrong argument count: '+name)
        self.depth+=1
        if self.depth>16:raise ProgramError('Call depth exceeded')
        try:
            self.block(f.body,dict(zip([a.arg for a in f.args.args],args)))
        except Returned as result:return result.value
        finally:self.depth-=1
        return None

    def bound(self,value):
        if isinstance(value,(list,tuple,dict,str)) and len(value)>10000:raise ProgramError('Container limit exceeded')
        if isinstance(value,int) and value.bit_length()>256:raise ProgramError('Integer magnitude exceeded')
        if isinstance(value,float) and not math.isfinite(value):raise ProgramError('Nonfinite computation')
        return value

    def set(self,target,value,env):
        if isinstance(target,ast.Name):env[target.id]=value
        elif isinstance(target,(ast.Tuple,ast.List)):
            if len(target.elts)!=len(value):raise ProgramError('Unpack size mismatch')
            for t,v in zip(target.elts,value):self.set(t,v,env)
        elif isinstance(target,ast.Subscript):self.expr(target.value,env)[self.expr(target.slice,env)]=value
        else:raise ProgramError('Invalid assignment target')

    def block(self,nodes,env):
        for n in nodes:
            self.tick()
            if isinstance(n,ast.Return):raise Returned(self.expr(n.value,env) if n.value else None)
            elif isinstance(n,ast.Assign):
                value=self.expr(n.value,env)
                for t in n.targets:self.set(t,value,env)
            elif isinstance(n,ast.AugAssign):self.set(n.target,self.binary(n.op,self.expr(n.target,env),self.expr(n.value,env)),env)
            elif isinstance(n,ast.Expr):self.expr(n.value,env)
            elif isinstance(n,ast.If):self.block(n.body if self.expr(n.test,env) else n.orelse,env)
            elif isinstance(n,(ast.For,ast.While)):
                iterator=iter(self.expr(n.iter,env)) if isinstance(n,ast.For) else None
                completed=True
                while True:
                    self.tick()
                    if iterator is not None:
                        try:item=next(iterator)
                        except StopIteration:break
                        self.set(n.target,item,env)
                    elif not self.expr(n.test,env):break
                    try:self.block(n.body,env)
                    except ContinueLoop:continue
                    except BreakLoop:completed=False;break
                if completed:self.block(n.orelse,env)
            elif isinstance(n,ast.Break):raise BreakLoop()
            elif isinstance(n,ast.Continue):raise ContinueLoop()
            elif not isinstance(n,ast.Pass):raise ProgramError('Unsupported statement '+type(n).__name__)

    def binary(self,op,a,b):
        if isinstance(op,ast.Pow) and (abs(b)>16 or abs(a)>1e12):raise ProgramError('Power operand limit')
        if isinstance(op,ast.Mult) and ((isinstance(a,(str,list,tuple)) and len(a)*b>10000) or
                                      (isinstance(b,(str,list,tuple)) and len(b)*a>10000)):
            raise ProgramError('Repeated container limit')
        return self.bound(BIN[type(op)](a,b))

    def expr(self,n,env):
        self.tick()
        if isinstance(n,ast.Constant):return self.bound(n.value)
        if isinstance(n,ast.Name):
            if n.id not in env:raise ProgramError('Undefined name: '+n.id)
            return env[n.id]
        if isinstance(n,(ast.List,ast.Tuple)):return [self.expr(x,env) for x in n.elts]
        if isinstance(n,ast.Dict):return {self.expr(k,env):self.expr(v,env) for k,v in zip(n.keys,n.values)}
        if isinstance(n,ast.Slice):return slice(*(self.expr(v,env) if v else None for v in (n.lower,n.upper,n.step)))
        if isinstance(n,ast.Subscript):return self.expr(n.value,env)[self.expr(n.slice,env)]
        if isinstance(n,ast.BinOp):return self.binary(n.op,self.expr(n.left,env),self.expr(n.right,env))
        if isinstance(n,ast.UnaryOp):return {ast.Not:operator.not_,ast.USub:operator.neg,ast.UAdd:operator.pos}[type(n.op)](self.expr(n.operand,env))
        if isinstance(n,ast.BoolOp):
            value=None
            for item in n.values:
                value=self.expr(item,env)
                if isinstance(n.op,ast.And) and not value:break
                if isinstance(n.op,ast.Or) and value:break
            return value
        if isinstance(n,ast.Compare):
            left=self.expr(n.left,env)
            for op,right_node in zip(n.ops,n.comparators):
                right=self.expr(right_node,env)
                if not CMP[type(op)](left,right):return False
                left=right
            return True
        if isinstance(n,ast.IfExp):return self.expr(n.body if self.expr(n.test,env) else n.orelse,env)
        if isinstance(n,ast.Call):
            args=[self.expr(v,env) for v in n.args]
            if any(k.arg is None for k in n.keywords):raise ProgramError('Keyword expansion unavailable')
            kwargs={k.arg:self.expr(k.value,env) for k in n.keywords}
            if isinstance(n.func,ast.Name):
                if kwargs:
                    if n.func.id!='dict':raise ProgramError('Use positional arguments for '+n.func.id)
                    return dict(*args,**kwargs)
                return self.call(n.func.id,args)
            if isinstance(n.func,ast.Attribute):
                obj=self.expr(n.func.value,env);name=n.func.attr
                permitted=({'get','items','keys','values','pop','copy','update'} if isinstance(obj,dict) else
                           {'append','extend','pop','copy','count','index'} if isinstance(obj,list) else set())
                if name not in permitted or kwargs:raise ProgramError('Unsupported data method: '+name)
                result=getattr(obj,name)(*args);self.bound(obj)
                return list(result) if name in ('items','keys','values') else self.bound(result)
            raise ProgramError('Only named functions can be called')
        if isinstance(n,(ast.ListComp,ast.DictComp)):
            output=[] if isinstance(n,ast.ListComp) else {}
            def visit(index,scope):
                self.tick()
                if index==len(n.generators):
                    if isinstance(n,ast.ListComp):output.append(self.expr(n.elt,scope))
                    else:output[self.expr(n.key,scope)]=self.expr(n.value,scope)
                    self.bound(output);return
                gen=n.generators[index]
                for value in self.expr(gen.iter,scope):
                    self.tick();inner=dict(scope);self.set(gen.target,value,inner)
                    if all(self.expr(c,inner) for c in gen.ifs):visit(index+1,inner)
            visit(0,env);return output
        raise ProgramError('Unsupported expression '+type(n).__name__)


def validate_candidate(source,contexts,capabilities=None,state=None,decision_validator=None,observer_validator=None):
    program=Program(source);checks=[]
    for original in contexts:
        ctx=deepcopy(original);current_state=deepcopy(state or {})
        if observer_validator is not None:
            observation_capabilities={k:v for k,v in (capabilities or {}).items()
                if k in ('measure_patch','measure_region','measure_rgb','segment_rgb','locate_rgb_point','recall_experience')}
            update=program.observe(ctx,current_state,observation_capabilities)
            if update is not None:
                ctx=observer_validator(update,ctx)
                if update.get('state_update') is not None:current_state=deepcopy(update['state_update'])
        result=program.run(ctx,current_state,capabilities)
        if decision_validator is not None:decision_validator(result,ctx)
        checks.append(dict(step=ctx.get('step'),kind=result['kind']))
    return dict(compiles=True,checks=checks,sha256=program.sha,status='provisional',
                scope='Executable on supplied inputs; no counterfactual physics or performance claim')
