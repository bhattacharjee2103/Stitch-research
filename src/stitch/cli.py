"""Command line entry point; all commands operate without a display server."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import yaml
from .database import PlaceDatabase
from .io import frames, read_image
from .multiview import coverage, place_views
from .pipeline import Pipeline, live_config, resize_for_scan
from .retrieval import GlobalEngine

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/default.yaml')
    sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('index');b.add_argument('database')
    reg=sub.add_parser('register',help='Register a captured reference, overlay and GPS location')
    reg.add_argument('database');reg.add_argument('place_id');reg.add_argument('reference')
    reg.add_argument('overlay');reg.add_argument('--name',default='')
    reg.add_argument('--latitude',type=float,required=True);reg.add_argument('--longitude',type=float,required=True)
    reg.add_argument('--opacity',type=float,default=.8)
    reg.add_argument('--anchor',type=float,nargs=8,
                     help='Normalized overlay corner coordinates x0 y0 x1 y1 x2 y2 x3 y3')
    reg.add_argument('--depth-map',help='Metric reference depth map saved as a NumPy .npy file')
    reg.add_argument('--anchor-3d',type=float,nargs=12,
                     help='Reference-camera 3D overlay corners x0 y0 z0 ... x3 y3 z3, in metres')
    for name in ['query','video']:
        a=sub.add_parser(name);a.add_argument('database');a.add_argument('input');a.add_argument('output')
        a.add_argument('--method',choices=['classical','local','full','depth3d'],default='full')
        a.add_argument('--latitude',type=float);a.add_argument('--longitude',type=float)
        if name=='video':a.add_argument('--limit',type=int,default=120)
    plan=sub.add_parser('plan',help='Print the yaw angles to capture for a 3D placement')
    plan.add_argument('--sweep',type=float,help='Total rescan range in degrees (default: config)')
    plan.add_argument('--tolerance',type=float,help='Per-view matching tolerance in degrees (default: config)')
    plan.add_argument('--database',help='Also report coverage of the views already registered')
    plan.add_argument('--place',help='Place ID used with --database')
    scan=sub.add_parser('scan',help='Show live recognition and overlays from a webcam or stream')
    scan.add_argument('database');scan.add_argument('input',help='Webcam index, video path or stream URL')
    scan.add_argument('--method',choices=['classical','local','full','depth3d'],default='full')
    scan.add_argument('--latitude',type=float);scan.add_argument('--longitude',type=float)
    scan.add_argument('--limit',type=int,default=0)
    args=p.parse_args()
    config=yaml.safe_load(Path(args.config).read_text())
    torch.set_num_threads(config['threads']);cv2.setNumThreads(config['threads'])
    torch.manual_seed(config['seed']);np.random.seed(config['seed']);cv2.setRNGSeed(config['seed'])
    if args.command=='plan':
        sweep=args.sweep if args.sweep is not None else config.get('capture_angle_sweep_deg',30.0)
        tolerance=args.tolerance if args.tolerance is not None else config.get('view_tolerance_deg',15.0)
        captured=[]
        if args.database:
            if not args.place:raise ValueError('--place is required with --database')
            captured=[v['yaw_deg'] for v in place_views(PlaceDatabase(args.database),args.place)]
        report=coverage(captured,sweep,tolerance)
        print(json.dumps(dict(sweep_deg=sweep,tolerance_deg=tolerance,**report),indent=2))
        return
    if args.command=='register':
        path=Path(args.database)
        if path.exists():
            db=PlaceDatabase(args.database)
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps({'schema_version':1,'places':[]},indent=2)+'\n')
            db=PlaceDatabase(args.database,allow_empty=True)
        place=db.register(args.place_id,args.name,args.reference,args.overlay,
                          args.latitude,args.longitude,args.opacity,
                          None if args.anchor is None else np.asarray(args.anchor).reshape(4,2),
                          depth_map=(np.load(args.depth_map,allow_pickle=False) if args.depth_map else None),
                          anchor_3d_camera=(None if args.anchor_3d is None else
                                            np.asarray(args.anchor_3d).reshape(4,3)))
        print(json.dumps(place,indent=2))
        return
    db=PlaceDatabase(args.database)
    if args.command=='index':
        db.build(GlobalEngine(config['cosplace_weights'],config['device'],size=config['cosplace_size']),
                 config['max_side'])
        print('Indexed',len(db.rows),'reference images')
        return
    pipeline=Pipeline(db,live_config(config,args.method),args.method)
    if args.command=='scan':
        shown=0
        try:
            for frame in frames(args.input,args.limit):
                frame=resize_for_scan(frame,config)
                prediction=pipeline.stabilise(pipeline.query(frame,args.latitude,args.longitude))
                output=pipeline.render(frame,prediction)
                title=(f"Stitch: {prediction.place_id} ({prediction.elapsed_ms:.0f} ms)"
                       if prediction.place_id else f"Stitch: searching ({prediction.elapsed_ms:.0f} ms)")
                cv2.imshow(title,output)
                shown+=1
                if cv2.waitKey(1) & 0xFF in (ord('q'),27):break
        finally:
            cv2.destroyAllWindows()
        if not shown:raise OSError('Video input contains no decodable frames')
        return
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    if args.command=='query':
        image,_=read_image(args.input,int(config.get('scan_query_side',config['max_side'])))
        prediction=pipeline.query(image,args.latitude,args.longitude)
        if not cv2.imwrite(args.output,pipeline.render(image,prediction)):
            raise OSError('Cannot write image output: '+args.output)
        print(json.dumps({'place_id':prediction.place_id,'reference':prediction.reference,
                          'latency_ms':prediction.elapsed_ms,'candidates':prediction.candidates,
                          'content':prediction.content,'location':next((r for r in prediction.candidates
                          if r['place_id']==prediction.place_id and r['reference']==prediction.reference),None)},indent=2))
        return
    writer=None
    written=0
    try:
        for frame in frames(args.input,args.limit):
            frame=resize_for_scan(frame,config)
            prediction=pipeline.stabilise(pipeline.query(frame,args.latitude,args.longitude))
            output=pipeline.render(frame,prediction)
            if writer is None:
                writer=cv2.VideoWriter(args.output,cv2.VideoWriter_fourcc(*'mp4v'),8,
                                       (output.shape[1],output.shape[0]))
                if not writer.isOpened():raise OSError('Cannot create video output')
            writer.write(output)
            written+=1
    finally:
        if writer:writer.release()
    if not written:
        raise OSError('Video input contains no decodable frames')

if __name__=='__main__':main()
