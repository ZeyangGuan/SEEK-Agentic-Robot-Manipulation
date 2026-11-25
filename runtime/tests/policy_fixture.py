import copy,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from seek.policy.execution_policy import Model,empty_memory,atomic_json,executed_delta
from seek.policy.mechanism_program import Program,ProgramError
from seek.policy.unified_vision import UnifiedVision
from tools.review import validate_review
from seek.policy.prompt_layout import restore_action_packet
from seek.policy.execution_receipt import execution_receipt

class PolicyFixture:
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        atomic_json(self.root/'input.json',empty_memory('general_pickup'))
        atomic_json(self.root/'config.json',dict(predict_horizon=5,execute_horizon=5,max_model_requests=None))
        self.env=patch.dict(os.environ,dict(ASTRA_API_KEY='offline-test',ASTRA_LOG_DIR=str(self.root/'trace'),
            ASTRA_HARNESS_CONFIG=str(self.root/'config.json'),SEEK_MEMORY=str(self.root/'input.json'),
            SEEK_OUTPUT=str(self.root/'output.json'),SEEK_EPISODE='1'))
        self.env.start();self.model=Model(dict(model='gpt-6-astra',task_name='general_pickup'))
        self.network=patch.object(self.model,'_request',side_effect=AssertionError('No live API in tests'));self.network.start()
        self.robot={}
        for arm in ('left','right'):
            self.robot.update({arm+'_ee_pose':[.3,.2,.4,1.,0.,0.,0.],arm+'_ee_joint_state':[1.],arm+'_arm_joint_state':[0.]*6})
        self.rgb=np.random.default_rng(0).integers(0,255,(80,80,3),dtype=np.uint8)
        self.observe(0)

    def tearDown(self):
        for tracker in self.model.seek_trackers.values():tracker.close()
        self.network.stop();self.env.stop();self.temp.cleanup()

    def observe(self,step,command=None):
        meta=dict(step=step,max_episode_steps=200)
        if command is not None:
            pending=self.model.seek_pending
            meta['executed_action']=command
            meta['execution_receipt']=execution_receipt(pending['plan_reference'],step,command,pending['before']['robot'],self.robot)
        self.model.update_obs(dict(model_meta=meta,state=copy.deepcopy(self.robot),instruction='Pick up the block.',
            vision={name:dict(color=self.rgb) for name in ('cam_head','cam_left_wrist','cam_right_wrist')}))

    def action(self):return [0.,0.,.005,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,1.]

    def receipt(self,command):
        for arm in ('left','right'):
            self.robot[arm+'_ee_pose']=list(command[arm+'_ee_pose'])
            self.robot[arm+'_ee_joint_state']=list(command[arm+'_ee_joint_state'])
        self.observe(self.model.step+1,command);self.model._context()
