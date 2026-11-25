from bisect import insort
from copy import deepcopy
import math

import numpy as np


def relevant_motion(rows, robot):
    def joints(value):
        return np.concatenate([np.asarray(value[arm+'_arm_joint_state'],float) for arm in ('left','right')])
    current=joints(robot)
    distances={i:float(np.max(np.abs(joints(r['robot'])-current))) for i,r in enumerate(rows) if 'robot' in r}
    chosen=list(range(max(0,len(rows)-2),len(rows)))
    for i in sorted(distances,key=lambda i:(distances[i],-i)):
        if i not in chosen:chosen.append(i)
        if len(chosen)>=8:break
    result=[]
    for i in sorted(chosen):
        item=deepcopy(rows[i]);item['configuration_distance_rad']=distances.get(i)
        result.append(item)
    return result


class MotionEvidenceIndex:
    def __init__(self):
        self.invalidate()

    def invalidate(self):
        self._rows = None
        self._key = None
        self._count = 0
        self._distances = {}
        self._nearest = []
        self._nonfinite = False

    @staticmethod
    def _joints(robot):
        return np.concatenate([
            np.asarray(robot[arm + '_arm_joint_state'], float)
            for arm in ('left', 'right')
        ])

    def select(self, rows, robot):
        current = self._joints(robot)
        if not np.isfinite(current).all():
            return relevant_motion(rows, robot)
        key = (current.shape, current.tobytes())
        if rows is not self._rows or key != self._key or len(rows) < self._count:
            self.invalidate()
            self._rows, self._key = rows, key
        if self._nonfinite:
            return relevant_motion(rows, robot)

        for i in range(self._count, len(rows)):
            row = rows[i]
            if 'robot' in row:
                distance = float(np.max(np.abs(self._joints(row['robot']) - current)))
                if not math.isfinite(distance):
                    self._nonfinite = True
                    return relevant_motion(rows, robot)
                insort(self._nearest, (distance, -i, i))
                if len(self._nearest) > 8:
                    self._nearest.pop()
                self._distances[i] = distance
            self._count = i + 1

        chosen = list(range(max(0, len(rows) - 2), len(rows)))
        for _, _, i in self._nearest:
            if i not in chosen:
                chosen.append(i)
            if len(chosen) >= 8:
                break
        result = []
        for i in sorted(chosen):
            item = deepcopy(rows[i])
            item['configuration_distance_rad'] = self._distances.get(i)
            result.append(item)
        return result
