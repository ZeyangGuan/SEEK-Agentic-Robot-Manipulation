import re


def normalize_instruction(value):
    return ' '.join(value.split()) if isinstance(value, str) and value.strip() else None


def task_binding(instruction, task='general_pickup'):
    text = normalize_instruction(instruction)
    result = dict(task=task, instruction=text, parsed=False, operation=None,
                  target=None, target_key=None, target_description=None, height_m=None)
    if task != 'general_pickup' or not text:
        return result
    match = re.fullmatch(r'pick\s+up\s+(?:the\s+|an?\s+)?(.+?)\s+by\s+'
                         r'(\d+(?:\.\d+)?)\s*(cm|centimeters?|centimetres?|m|meters?|metres?)\.?',
                         text, re.IGNORECASE)
    if not match:
        return result
    target = ' '.join(match.group(1).lower().split())
    height = float(match.group(2)) * (.01 if match.group(3).lower().startswith('c') else 1.)
    if not target or height <= 0:
        return result
    result.update(parsed=True, operation='pickup', target=target, target_key=target,
                  target_description=match.group(1), height_m=height)
    return result


def _binding(value, task):
    return value if isinstance(value, dict) else task_binding(value, task)


def same_target(first, second, task='general_pickup'):
    a, b = _binding(first, task), _binding(second, task)
    if a.get('task') != b.get('task'):
        return False
    if a.get('parsed') and b.get('parsed'):
        return a['operation'] == b['operation'] and a['target_key'] == b['target_key']
    return bool(a.get('instruction') and a.get('instruction') == b.get('instruction'))
