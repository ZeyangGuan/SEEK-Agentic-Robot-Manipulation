from scipy.spatial.transform import Rotation


class FeatureFrame:
    def __init__(self, reference, names):
        self.rotations = {}
        for arm in ('left', 'right'):
            keys = tuple(arm+'.r'+axis for axis in 'xyz')
            if set(keys) <= set(names):
                self.rotations[keys] = Rotation.from_rotvec([reference[k] for k in keys])

    def project(self, features):
        result = dict(features)
        for keys, reference in self.rotations.items():
            values = [features.get(k) for k in keys]
            if any(v is None for v in values):
                continue
            relative = Rotation.from_rotvec(values)*reference.inv()
            result.update(zip(keys, relative.as_rotvec().tolist()))
        return result

    def restore(self, features):
        result = dict(features)
        for keys, reference in self.rotations.items():
            absolute = Rotation.from_rotvec([features[k] for k in keys])*reference
            result.update(zip(keys, absolute.as_rotvec().tolist()))
        return result


def goal_errors(features, goals, scales):
    frame = FeatureFrame(features, goals)
    current, target = frame.project(features), frame.project(goals)
    return {name:(current[name]-target[name])/scales[name] for name in goals}
