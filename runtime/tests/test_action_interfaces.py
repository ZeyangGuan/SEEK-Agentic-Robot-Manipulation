from copy import deepcopy
import json,unittest
from unittest.mock import patch
import policy_fixture as fixture
from seek.policy.mechanism_program import Program,ProgramError
from seek.policy.review_replay import replay


class ActionInterfaceTests(unittest.TestCase):
    setUp=fixture.PolicyFixture.setUp
    tearDown=fixture.PolicyFixture.tearDown
    observe=fixture.PolicyFixture.observe
    receipt=fixture.PolicyFixture.receipt
    action=fixture.PolicyFixture.action

    def objective(self,z):
        return dict(kind='control',name='measured waypoint',goals={'left.z':z},scales={'left.z':.001},requires=[])

    def test_actual_receipts_remain_readable_but_do_not_generate_actions(self):
        m=self.model
        for _ in range(3):self.receipt(m._dispatch(self.action(),m._context(),{},'teacher_buffer','actual fixture receipt')['action'])
        ctx=m._context();m.seek_experience.activate(self.objective(.45),ctx)
        before=deepcopy(m.seek_bank)
        status=m._capabilities(ctx)['control_status']()
        self.assertEqual(status['kind'],'fallback');self.assertNotIn('action',status)
        self.assertEqual(m.seek_bank,before)
        self.assertFalse(hasattr(m,'_fit_request'));self.assertFalse(hasattr(m.seek_experience,'propose'))
        self.assertEqual(len(m.seek_bank['transitions']),3)

    def test_retired_capabilities_are_absent_and_cannot_execute(self):
        caps=self.model._capabilities(self.model._context())
        for name in ('fit_action','reuse_experience'):
            self.assertNotIn(name,caps)
            with self.assertRaises(ProgramError):Program('def decide(ctx,state):\n return '+name+'()').run({},capabilities=caps)

    def test_goal_completion_and_applicability_checks_remain(self):
        m=self.model;ctx=m._context();spec=self.objective(ctx['features']['left.z'])
        m.seek_experience.activate(spec,ctx)
        self.assertEqual(m.seek_experience.assess(ctx)['reason'],'measured_objective_complete')
        spec=self.objective(.5);spec['requires']=[dict(feature='left.z',lower=.8,upper=.9)]
        m.seek_experience.activate(spec,ctx)
        self.assertEqual(m.seek_experience.assess(ctx)['reason'],'objective_condition_changed')

    def test_no_implicit_action_after_teacher_plan_exhaustion(self):
        m=self.model
        for _ in range(3):self.receipt(m._dispatch(self.action(),m._context(),{},'teacher_buffer','actual fixture receipt')['action'])
        m.seek_experience.activate(self.objective(.45),m._context())
        m.seek_program=Program("def decide(ctx,state):\n return {'kind':'delegate'}")
        before=len(m.seek_bank['transitions'])
        with patch.object(m,'_request',side_effect=AssertionError('new teaching needed')):
            with self.assertRaisesRegex(AssertionError,'new teaching needed'):m.get_action()
        self.assertEqual(len(m.seek_bank['transitions']),before);self.assertIsNone(m.seek_pending)

    def test_review_replay_supports_read_only_control_status(self):
        ctx=self.model._context();path=self.root/'empty_observations.jsonl';path.write_text('')
        source="def decide(ctx,state):\n status=control_status()\n return {'kind':'ask','reason':status['reason']}"
        result=replay(source,['fresh_entry'],{'cases':[]},self.model.seek_bank,path,initial_context=ctx)
        self.assertEqual(result['cases'][0]['status'],'observed_decision')
        self.assertEqual(result['cases'][0]['decisions'][0]['reason'],'current_control_objective_missing')
