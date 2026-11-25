from copy import deepcopy
import json, os, sys, tempfile, types, unittest
from pathlib import Path
from unittest.mock import patch
import policy_fixture as fixture
import test_frozen_evaluation as frozen
from seek.policy.execution_policy import Model, atomic_json, empty_memory
from seek.policy.operation_knowledge import OperationKnowledge, prepare_updates


def code(dz):
    return "def decide(ctx,state):\n return {'kind':'act','action':[0,0,"+str(dz)+",0,0,0,1,0,0,0,0,0,0,1]}"


class EpisodeAdaptationTests(unittest.TestCase):
    observe = fixture.PolicyFixture.observe
    action = fixture.PolicyFixture.action
    response = frozen.FrozenTests.response
    record_steps = frozen.FrozenTests.record_steps

    def setUp(self):
        self.mode = patch.dict(os.environ, SEEK_FREEZE_LEARNING='1', SEEK_EPISODE_ADAPTATION='1')
        self.mode.start()
        fixture.PolicyFixture.setUp(self)

    def tearDown(self):
        fixture.PolicyFixture.tearDown(self)
        self.mode.stop()

    def receipt(self, packet):
        self.robot['left_ee_pose'] = list(packet['action']['left_ee_pose'])
        self.observe(self.model.step+1, packet['action'])
        self.model._context()

    def test_program_can_be_authored_executed_and_is_absent_next_episode(self):
        m = self.model
        incoming = m.seek_input.read_bytes()
        program = dict(source=code(.006),description='Current-scene executable revision',evidence_ids=['e1:s0'])
        with patch.object(m,'_request',side_effect=None,return_value=self.response(actions=[],program=program)):
            packet = m.get_action()
        self.assertAlmostEqual(packet['action']['left_ee_pose'][2],.406,places=6)
        identity = m.seek_bank['active']
        self.receipt(packet)
        self.assertEqual(m.seek_bank['transitions'][-1]['program'],identity)
        m.finalize()
        self.assertEqual(m.seek_input.read_bytes(),incoming)
        self.assertEqual(m.seek_output.read_bytes(),incoming)
        trace = json.loads((self.root/'evaluation_trace_memory.json').read_text())
        self.assertEqual(trace['active'],identity)
        self.assertEqual(len(trace['transitions']),1)
        audit = json.loads((self.root/'episode_adaptation_audit.json').read_text())
        self.assertIn(identity,audit['new_program_ids'])
        with patch.dict(os.environ,SEEK_MEMORY=str(m.seek_output),SEEK_OUTPUT=str(self.root/'next.json'),SEEK_EPISODE='2'):
            self.model = Model(dict(model='gpt-6-astra',task_name='general_pickup'))
        self.observe(0)
        self.assertIsNone(self.model.seek_bank['active'])
        self.assertNotIn(identity,{p['id'] for p in self.model.seek_bank['programs']})
        self.assertEqual(self.model.seek_bank['transitions'],[])

    def test_operation_revision_executes_then_original_version_is_restored(self):
        m = self.model
        memory = deepcopy(m.seek_base)
        bank = OperationKnowledge(memory,0)
        original = prepare_updates([dict(name='lift',source=code(.002),description='Incoming operation',evidence_ids=[])],set(),0,0)
        bank.install(original)
        atomic_json(m.seek_input,memory)
        self.model = Model(dict(model='gpt-6-astra',task_name='general_pickup'))
        self.observe(0);m = self.model
        incoming = m.seek_input.read_bytes()
        updated = dict(name='lift',source=code(.007),description='Revised from current response',evidence_ids=['e1:s0'])
        reply = self.response(actions=[],operations=[updated],operation=dict(name='lift',parameters={}))
        with patch.object(m,'_request',side_effect=None,return_value=reply):packet=m.get_action()
        revised = m.seek_operations.bank['selected']['lift']
        self.assertNotEqual(revised,original[0]['id'])
        self.assertAlmostEqual(packet['action']['left_ee_pose'][2],.407,places=6)
        self.receipt(packet);m.finalize()
        self.assertEqual(m.seek_output.read_bytes(),incoming)
        trace=json.loads((self.root/'evaluation_trace_memory.json').read_text())
        self.assertEqual(trace['operation_knowledge']['selected']['lift'],revised)
        with patch.dict(os.environ,SEEK_MEMORY=str(m.seek_output),SEEK_OUTPUT=str(self.root/'next.json'),SEEK_EPISODE='2'):
            self.model=Model(dict(model='gpt-6-astra',task_name='general_pickup'))
        self.observe(0)
        self.assertEqual(self.model.seek_operations.bank['selected']['lift'],original[0]['id'])
        reply=self.response(actions=[],operation=dict(name='lift',parameters={}))
        with patch.object(self.model,'_request',side_effect=None,return_value=reply):packet=self.model.get_action()
        self.assertAlmostEqual(packet['action']['left_ee_pose'][2],.409,places=6)

    def test_schema_prompt_and_validation_agree(self):
        m=self.model;schema=m._tools()[0]['parameters']['properties']
        self.assertIn('anyOf',schema['program'])
        self.assertNotEqual(schema['operations'].get('maxItems'),0)
        self.assertIn('WITHIN-EPISODE PROGRAM ADAPTATION',m.prompt)
        self.assertNotIn('act.program must be null',m.prompt)
        bad=dict(source=code(.005),description='Unknown physical evidence',evidence_ids=['fabricated'])
        with patch.object(m,'_request',side_effect=None,return_value=self.response(actions=[],program=bad)):
            self.assertIsNone(m._ask())
        self.assertIn('Unavailable evidence IDs',m.seek_error['detail'])
        self.assertIsNone(m.seek_pending)

    def test_current_receipts_support_local_feedback(self):
        self.record_steps()
        self.assertEqual(len(self.model.seek_bank['transitions']), 3)
        self.assertEqual(self.model.seek_base['transitions'], [])
        self.assertEqual(self.model.seek_experience.bank['methods'], {})
        self.assertFalse(hasattr(self.model, '_fit_request'))

    def test_mode_requires_fixed_carryover(self):
        with patch.dict(os.environ,SEEK_FREEZE_LEARNING='0'):
            with self.assertRaisesRegex(ValueError,'fixed incoming/output memory'):
                Model(dict(model='gpt-6-astra',task_name='general_pickup'))


if __name__=='__main__':unittest.main()
