"""Entry-independent checks of ego-change display acceleration and isolation."""
from types import SimpleNamespace as N

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
  out=c.apply(p,y,f,slot,('lane','center'),((1.75,-1.75),(5.25,1.75),(-1.75,-5.25))[slot],t.tracks[1],frame)
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

def test_post_interval_restores_original_rules_and_history():
  c,t=pair();base,bt=pair()
  base.update_ego_follow=lambda *args:None
  for frame in range(0,401,5):
    active=100<=frame<=200;y=3.3 if frame>=100 else .1;slot=1 if frame>=100 else 0
    actual,_=step(c,t,frame,y,slot,starting=active)
    # The reference runs ordinary rules without the new drawing activation.
    expected,_=step(base,bt,frame,y,slot,starting=active)
    if frame<=300 and frame>=100:assert c.ego_follow_active
    if frame>300:
      assert not c.ego_follow_active
      assert actual==expected
      assert c.tracks==base.tracks
      assert c.display_slots==base.display_slots
