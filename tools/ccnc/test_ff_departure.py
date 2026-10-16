"""FF self-baseline departure and vision confirmation regressions."""
import math
from types import SimpleNamespace as N
from test_radar_display import Correction,Tracker,corrected_sample as original_sample

def corrected_sample(c,t,frame,raw,*args,vision=None,**kwargs):
  visual=raw if vision is None else vision
  c.vision_model=N(timestampEof=frame+1,leadsV3=[N(prob=.95,x=[21.52],y=[-visual])])
  return original_sample(c,t,frame,raw,*args,**kwargs)

def test_offset_vehicle_small_oscillation_keeps_lock():
  c,t=Correction(),Tracker()
  for frame in range(0,201,5):corrected_sample(c,t,frame,.3,reference=('lane','center'))
  for frame in range(205,1001,5):
    output=corrected_sample(c,t,frame,.3+.35*math.sin((frame-200)*.07),reference=('lane','center'))
    assert not c.tracks[1].get('ff_departure',{}).get('released',False)
    assert output==0.

def test_stepped_departure_releases_within_point_three_seconds():
  for sign in (-1,1):
    c,t=Correction(),Tracker()
    for frame in range(0,201,5):corrected_sample(c,t,frame,.3,reference=('lane','center'))
    crossing=release=None
    for frame in range(205,341,5):
      # 0.175m every 0.25s = 0.7m/s; the pause stays within the requested 0.25s.
      dev=sign*.175*((frame-200)//25)
      output=corrected_sample(c,t,frame,.3+dev,reference=('lane','center'))
      ff=c.tracks[1].get('ff_departure',{})
      if abs(dev)>=.5 and crossing is None:crossing=frame
      if ff.get('released') and release is None:release=frame
    assert release is not None and release-crossing<=30
    # Re-locking must re-arm the same track for a later maneuver.
    final=.3+sign*.875
    for frame in range(345,801,5):corrected_sample(c,t,frame,final,reference=('lane','center'))
    assert c.tracks[1]['ff_departure']['locked']
    for frame in range(805,936,5):
      corrected_sample(c,t,frame,final-sign*.175*((frame-800)//25),reference=('lane','center'))
    assert 875<=c.tracks[1]['ff_departure']['release_frame']<=905

def test_ff_state_is_cleared_on_side_slot_and_ego_lane_change():
  for transition in ('side','ego'):
    c,t=Correction(),Tracker()
    for frame in range(0,201,5):corrected_sample(c,t,frame,0.,reference=('lane','center'))
    for frame in range(205,326,5):corrected_sample(c,t,frame,.175*((frame-200)//25),reference=('lane','center'))
    assert c.tracks[1].get('ff_departure',{}).get('released')
    if transition=='ego':
      c.lane_change=True
      corrected_sample(c,t,330,1.,reference=('lane','center'))
    else:corrected_sample(c,t,330,2.,slot=1,bounds=(5.4,1.8),reference=('lane','center'))
    assert 'ff_departure' not in c.tracks[1]
    assert 'ff_vision' not in c.tracks[1]


def test_radar_departure_without_camera_motion_is_vetoed():
  for sign in (-1,1):
    c,t=Correction(),Tracker()
    for frame in range(0,201,5):corrected_sample(c,t,frame,0.,vision=0.,reference=('lane','center'))
    ready=None
    for frame in range(205,401,5):
      raw=sign*(.5+(frame-205)*.0001)
      corrected_sample(c,t,frame,raw,vision=0.,reference=('lane','center'))
      ff=c.tracks[1]['ff_departure']
      if ff['radar_ready'] is not None:ready=ff['radar_ready']
      assert not ff['released']
    assert ready is not None and c.tracks[1]['ff_departure']['vision_rejected']
    assert c.tracks[1]['ff_departure']['vision_reason']=='timeout'


def test_camera_confirmation_point_three_seconds_after_radar_releases_then():
  for sign in (-1,1):
    c,t=Correction(),Tracker()
    for frame in range(0,201,5):corrected_sample(c,t,frame,0.,vision=0.,reference=('lane','center'))
    ready=release=None
    for frame in range(205,321,5):
      raw=sign*(.5+(frame-205)*.0001)
      visual=sign*.3 if ready is not None and frame>=ready+30 else 0.
      corrected_sample(c,t,frame,raw,vision=visual,reference=('lane','center'))
      ff=c.tracks[1]['ff_departure']
      if ready is None and ff['radar_ready'] is not None:ready=ff['radar_ready']
      if ff['released'] and release is None:release=frame
    assert ready is not None and release==ready+30


def display_ff(c,t,frame,key,raw):
  output=original_sample(c,t,frame,raw,track_id=key,reference=('lane','center'))
  values={k:0 for fields in c.SLOT_FIELDS for k in fields}
  values.update(FF_DETECT=1,FF_LATERAL=-output,FF_DISTANCE=16.)
  c.stabilize_slots(values,(key,None,None),t.tracks,frame)
  return -values['FF_LATERAL']


def test_locked_ff_replacement_inherits_visible_lateral_through_short_report_gap():
  c,t=Correction(),Tracker()
  for frame in range(0,101,5):display_ff(c,t,frame,1,0.)
  # A remembered, still-locked lead remains valid during the existing geometry hold.
  c.tracks[1]['valid_frame']=70
  assert display_ff(c,t,105,2,.9)==0.
  assert c.tracks[2]['output']==0.
  assert c.tracks[2]['locked'] and not c.tracks[2]['following']


def test_ff_replacement_keeps_boundary_motion_and_ego_change_outputs():
  for moving,ego in ((True,False),(False,True)):
    c,t=Correction(),Tracker()
    for frame in range(0,101,5):display_ff(c,t,frame,1,0.)
    c.lane_change=ego
    output=display_ff(c,t,105,2,2. if moving else .9)
    assert output==(2. if moving else .9)
