"""Entry-independent checks of ego-change display acceleration and isolation."""
from types import SimpleNamespace as N

import pytest

from test_radar_display import Correction, Tracker, point

def model(frame,starting=False):
  return N(timestampEof=frame+1,meta=N(laneChangeState='laneChangeStarting' if starting else 'off'),
          laneLines=[N(x=[0.,2.5,5.],y=[y,y,y]) for y in (-5.25,-1.75,1.75,5.25)],laneLineProbs=[.9]*4)

def step(c,t,frame,y,slot=0,starting=False,blinker=False):
  c.update_ego_follow(model(frame,starting),20.,3 if starting else 0,blinker,False,frame*.01)
  c.lane_change=starting or blinker
  p=point(y=y);t.observe(N(points=[p]),frame)
  c.update_centers(3.5,3.5,frame,t.tracks)
  _,f=t.filter_position(p,y,('lane','center'),frame)
  out=c.apply(p,y,f,slot,('lane','center'),((1.75,-1.75),(5.25,1.75),(-1.75,-5.25))[slot],t.tracks[1],frame,lateral_filter=t.positions[1][-1])
  values={k:0 for fields in c.SLOT_FIELDS for k in fields}
  distance,lateral,detect=c.SLOT_FIELDS[slot]
  values.update({distance:16.,lateral:-out if slot in (0,2) else out,detect:1})
  selected=[None]*3;selected[slot]=1
  c.stabilize_slots(values,tuple(selected),t.tracks,frame)
  shown=c.display_selected.index(1)
  return values,values[c.SLOT_FIELDS[shown][1]]*(-1 if shown in (0,2) else 1)

def pair():
  c,t=Correction(),Tracker();t.position_correction=c
  return c,t

def test_slot_crossing_converges_within_point_two_seconds():
  for old,new,y in ((0,1,3.5),(0,2,-3.5),(1,0,0.),(2,0,0.)):
    c,t=pair()
    for f in range(0,100,5):step(c,t,f,(0.,3.5,-3.5)[old],old)
    outputs=[]
    for f in range(100,181,5):
      _,shown=step(c,t,f,y,new,starting=True)
      outputs.append((f,abs(shown-y)))
    first=next(f for f,e in outputs if e<=.3)
    assert first-100<=20
    assert all(e<=.3 for f,e in outputs if f>=first)

def test_blinker_without_lateral_motion_is_unchanged():
  c,t=pair();plain,pt=pair()
  for frame in range(0,401,5):
    y=.25 if frame%10 else .3
    actual,_=step(c,t,frame,y,blinker=True)
    expected,_=step(plain,pt,frame,y)
    assert not c.ego_follow_active
    assert actual==expected

def test_post_interval_resumes_ordinary_rates_from_visible_value():
  c,t=pair()
  previous=None
  for frame in range(0,401,5):
    active=100<=frame<=200;y=3.3 if frame>=100 else .1;slot=1 if frame>=100 else 0
    _,output=step(c,t,frame,y,slot,starting=active)
    if 100<=frame<=300:assert c.ego_follow_active
    if frame>300:
      assert not c.ego_follow_active
      assert abs(output-previous)<=.075+1e-9
    previous=output


def test_post_handoff_preserves_last_visible_output_and_lateral_filter():
  c,t=pair()
  for frame in range(0,351,5):
    _,visible=step(c,t,frame,1.2 if frame==350 else .1,starting=200<=frame<=250)
  assert c.ego_follow_active and visible>=.95-1e-9
  _,resumed=step(c,t,355,1.2)
  assert not c.ego_follow_active
  assert abs(resumed-visible)<=.015+1e-9
  assert t.positions[1][-1].value==visible
  assert c.tracks[1]['locked']


def gap_model(left=1.8,starting=False):
  return N(timestampEof=1,meta=N(laneChangeState='laneChangeStarting' if starting else 'off'),
           laneLines=[N(x=[0.,2.5,5.,30.],y=[-y]*4) for y in (left+3.6,left,left-3.6,left-7.2)],
           laneLineProbs=[.95]*4,orientationRate=N(z=[0.]),leadsV3=[])

def gap_values(slot):
  v={k:0 for fields in Correction.SLOT_FIELDS for k in fields}
  for k,n in zip(Correction.SLOT_FIELDS[slot],(10.,0. if slot==0 else 3.,1)):v[k]=n
  return v

@pytest.mark.parametrize('old',[1,2])
def test_actual_change_side_to_ff_skips_only_its_debounce(old):
  c=Correction();selected=[None]*3;selected[old]=1
  c.stabilize_slots(gap_values(old),tuple(selected),{1:(0.,0)},0)
  c.update_ego_follow(gap_model(starting=True),20.,3,True,False,.05)
  c.lane_change=True
  c.apply(N(trackId=1,dRel=10.,yRel=0.),0.,0.,0,('lane','center'),(1.75,-1.75),(0.,5),5)
  v=gap_values(0);c.stabilize_slots(v,(1,None,None),{1:(0.,5)},5)
  assert c.ego_change_active and c.display_selected==(1,None,None)
  assert v['FF_DETECT']==1 and v[Correction.SLOT_FIELDS[old][2]]==0

def test_signal_only_keeps_debounce():
  c=Correction();c.stabilize_slots(gap_values(1),(None,1,None),{1:(0.,0)},0)
  for frame in (5,10,15,20,25):
    md=gap_model();md.timestampEof=frame+1
    c.update_ego_follow(md,20.,0,True,False,frame*.01);c.lane_change=True
    c.apply(N(trackId=1,dRel=10.,yRel=0.),0.,0.,0,('lane','center'),(1.75,-1.75),(0.,frame),frame)
    v=gap_values(0);c.stabilize_slots(v,(1,None,None),{1:(0.,frame)},frame)
    assert not c.ego_change_active
    assert c.display_selected==((None,1,None) if frame<25 else (1,None,None))

def test_one_valid_inner_line_confirms_motion_without_changing_old_draw_activation():
  c=Correction()
  for i in range(6):
    md=gap_model();md.timestampEof=i+1;md.laneLineProbs[2]=0.
    for line in md.laneLines:
      line.y=[y+.025*x for x,y in zip(line.x,line.y)]
    c.update_ego_follow(md,20.,0,True,False,i*.05)
  assert c.ego_change_active
  assert not c.ego_follow_active

def test_pending_crossing_keeps_ordinary_debounce_after_ff_becomes_empty():
  c=Correction();v=gap_values(0)
  for k,n in zip(Correction.SLOT_FIELDS[1],(12.,3.,1)):v[k]=n
  c.stabilize_slots(v,(2,1,None),{1:(0.,0),2:(0.,0)},0)
  for frame in (5,10,15,20,25):
    md=gap_model(starting=True);md.timestampEof=frame+1
    c.update_ego_follow(md,20.,3,True,False,frame*.01);c.lane_change=True
    c.apply(N(trackId=1,dRel=10.,yRel=0.),0.,0.,0,('lane','center'),(1.75,-1.75),(0.,frame),frame)
    v=gap_values(0)
    for k,n in zip(Correction.SLOT_FIELDS[2],(10.,3.,1)):v[k]=n
    c.stabilize_slots(v,(1,None,2),{1:(0.,frame),2:(0.,frame)},frame)
    assert c.display_selected==((None,1,2) if frame<25 else (1,None,2))

@pytest.mark.parametrize('bounds',[None,(.1,-3.5),(.25,-3.5),(1.75,float('nan'))])
def test_missing_or_boundary_geometry_keeps_the_existing_debounce(bounds):
  c=Correction();c.stabilize_slots(gap_values(1),(None,1,None),{1:(0.,0)},0)
  c.update_ego_follow(gap_model(starting=True),20.,3,True,False,.05);c.lane_change=True
  c.apply(N(trackId=1,dRel=10.,yRel=0.),0.,0.,0,('lane','center'),bounds,(0.,5),5)
  v=gap_values(0);c.stabilize_slots(v,(1,None,None),{1:(0.,5)},5)
  assert c.display_selected==(None,1,None) and not v['FF_DETECT']
