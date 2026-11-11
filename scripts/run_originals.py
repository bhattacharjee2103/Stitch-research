"""Execute untouched upstream scripts using a bounded, file-mode GUI adapter."""
import argparse
import os
import json
from pathlib import Path
import runpy
import time
import cv2
import numpy as np

class EndOfInput(Exception):pass

def run(script,source,limit):
    original={k:getattr(cv2,k) for k in ['VideoCapture','imshow','waitKey','destroyAllWindows',
                                       'findHomography','warpPerspective']}
    stats={'script':script,'source':source,'limit':limit,'decoded_frames':0,
           'displayed_frames':0,'homography_calls':0,'homography_nonnull':0,'warp_calls':0,
           'headless_file_adapter':True,'vision_code_modified':False}
    captures=[];writer=None
    class Capture:
        def __init__(self,*args,**kwargs):
            self.inner=original['VideoCapture'](source);captures.append(self)
        def read(self):
            if stats['decoded_frames']>=limit:raise EndOfInput()
            ok,frame=self.inner.read()
            if not ok:raise EndOfInput()
            stats['decoded_frames']+=1
            return ok,frame
        def __getattr__(self,name):return getattr(self.inner,name)
    def show(name,frame):
        nonlocal writer
        stats['displayed_frames']+=1
        if writer is None:
            raw=Path(os.environ.get('STITCH_RESULTS','results'))/'raw';raw.mkdir(parents=True,exist_ok=True)
            writer=cv2.VideoWriter(str(raw/(script[:-3]+'.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),8,
                                    (frame.shape[1],frame.shape[0]))
        writer.write(frame)
    def find(*args,**kwargs):
        stats['homography_calls']+=1
        H,mask=original['findHomography'](*args,**kwargs)
        stats['homography_nonnull']+=int(H is not None)
        return H,mask
    def warp(*args,**kwargs):
        stats['warp_calls']+=1
        return original['warpPerspective'](*args,**kwargs)
    cv2.VideoCapture=Capture;cv2.imshow=show;cv2.waitKey=lambda _: -1
    cv2.destroyAllWindows=lambda:None;cv2.findHomography=find;cv2.warpPerspective=warp
    cv2.setNumThreads(2);cv2.setRNGSeed(17)
    start=time.perf_counter()
    try:
        runpy.run_path(script,init_globals={'URL':source},run_name='__main__')
        stats['status']='complete'
    except EndOfInput:
        stats['status']='bounded_file_input_complete'
    except Exception as e:
        stats['status']='error';stats['error']=repr(e)
    finally:
        stats['elapsed_s']=time.perf_counter()-start
        for cap in captures:cap.release()
        if writer:writer.release()
        for name,value in original.items():setattr(cv2,name,value)
    stats['ms_per_decoded_frame']=stats['elapsed_s']*1000/max(1,stats['decoded_frames'])
    return stats

def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=90)
    p.add_argument('--source',default='video.mp4')
    p.add_argument('--scripts',nargs='+',default=['tep.py','tep2.py','tep3.py']);args=p.parse_args()
    out=[]
    for script in args.scripts:
        result=run(script,args.source,args.limit);out.append(result);print(json.dumps(result),flush=True)
    (Path(os.environ.get('STITCH_RESULTS','results'))/'original_baselines.json').write_text(json.dumps(out,indent=2))

if __name__=='__main__':main()
