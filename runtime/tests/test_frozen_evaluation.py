from copy import deepcopy
import json
import unittest
from unittest.mock import patch
import policy_fixture as fixture
from seek.policy.execution_policy import Model, atomic_json
from seek.policy.operation_knowledge import prepare_updates
from seek.policy.frozen_evaluation import knowledge


class FrozenTests(unittest.TestCase):
    observe = fixture.PolicyFixture.observe
    action = fixture.PolicyFixture.action

    def setUp(self):
        self.freeze = patch.dict('os.environ', SEEK_FREEZE_LEARNING='1')
        self.freeze.start()
        fixture.PolicyFixture.setUp(self)

    def tearDown(self):
        fixture.PolicyFixture.tearDown(self)
        self.freeze.stop()

    def response(self, **changes):
        args = dict(actions=[self.action()], execution_note='offline', bindings=[],
                    program=None, state_update=None, intent=None, operations=[])
        args.update(changes)
        return dict(output=[dict(type='function_call', name='act', arguments=json.dumps(args))])

    def record_steps(self):
        m = self.model
        for step in range(1, 4):
            packet = m._dispatch(self.action(), m._context(), {}, 'teacher_buffer', 'offline')
            self.robot['left_ee_pose'] = list(packet['action']['left_ee_pose'])
            self.observe(step, packet['action'])
            m._context()

    def test_new_program_and_operation_are_rejected_before_state_changes(self):
        m = self.model
        for proposal in [dict(program=dict(source="def decide(ctx,state):\n return {'kind':'cached'}",description='new',evidence_ids=[])),
                         dict(operations=[dict(name='new')])]:
            with patch.object(m, '_request', side_effect=None, return_value=self.response(**proposal)):
                self.assertIsNone(m._ask())
            self.assertEqual(m.seek_error['kind'], 'proposal_error')
            self.assertIn('Frozen evaluation', m.seek_error['detail'])
            self.assertEqual(knowledge(m.seek_bank), knowledge(m.seek_base))
            self.assertIsNone(m.seek_pending)

    def test_direct_actions_remain_available_and_output_memory_is_exact(self):
        m = self.model
        with patch.object(m, '_request', side_effect=None, return_value=self.response()):
            self.assertEqual(m._ask(), [self.action()])
        self.record_steps()
        m.finalize()
        self.assertEqual(m.seek_output.read_bytes(), m.seek_input.read_bytes())
        trace = json.loads((self.root/'evaluation_trace_memory.json').read_text())
        self.assertEqual(len(trace['transitions']), 3)
        self.assertEqual(knowledge(trace), knowledge(m.seek_base))

    def test_current_receipts_remain_trace_evidence(self):
        m = self.model
        self.record_steps()
        m._checkpoint_memory('offline')
        self.assertEqual(len(m.seek_bank['transitions']), 3)
        self.assertEqual(m.seek_output.read_bytes(), m.seek_input.read_bytes())
        self.assertEqual(m.seek_base['transitions'], [])
        self.assertFalse(hasattr(m, '_fit_request'))

    def test_existing_operation_can_execute_without_authoring(self):
        m = self.model
        memory = deepcopy(m.seek_base)
        code = "def decide(ctx,state):\n return {'kind':'act','action':[0,0,.005,0,0,0,1,0,0,0,0,0,0,1]}"
        records = prepare_updates([dict(name='retained',source=code,description='prior operation',evidence_ids=[])],set(),1,0)
        memory['operation_knowledge']['versions'] = records
        memory['operation_knowledge']['selected'] = {'retained':records[0]['id']}
        memory['operation_knowledge']['entry'] = 'retained'
        memory['episodes'] = [dict(episode=1)]
        atomic_json(m.seek_input, memory)
        with patch.dict('os.environ', SEEK_EPISODE='2'):
            self.model = Model(dict(model='gpt-6-astra',task_name='general_pickup'))
        self.observe(0)
        with patch.object(self.model, '_request', side_effect=AssertionError('No teacher needed')):
            packet = self.model.get_action()
        self.assertAlmostEqual(packet['action']['left_ee_pose'][2], .405, places=6)
        self.model._checkpoint_memory('offline')
        self.assertEqual(self.model.seek_output.read_bytes(), self.model.seek_input.read_bytes())

    def test_current_control_keeps_feedback_without_learning_new_method(self):
        m = self.model
        spec = dict(kind='control',name='current target',goals={'left.z':.45},scales={'left.z':.001},requires=[])
        m.seek_experience.activate(spec,m._context())
        self.record_steps()
        self.assertEqual(m.seek_experience.bank['methods'], {})
        self.assertEqual(len(m.seek_experience.active['assessments']), 3)
        self.assertEqual(m.seek_experience.assess(m._context())['kind'], 'fallback')


if __name__ == '__main__':
    unittest.main()
