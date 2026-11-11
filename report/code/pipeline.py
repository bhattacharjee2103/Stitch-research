"""Coarse retrieval, local verification and content rendering."""
from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional
import time
import cv2
import numpy as np
from .database import PlaceDatabase
from .features import Features, LocalEngine
from .geometry import Verification, verify
from .io import read_image
from .rendering import render, render_projected_quad
from .retrieval import GlobalEngine, rank
from .pose3d import camera_matrix, estimate_pose, project_points

@dataclass
class Prediction:
    place_id: Optional[str]
    reference: Optional[str]
    verification: Optional[Verification]
    candidates: List[Dict[str,Any]]
    elapsed_ms: float
    content: List[Dict[str,Any]]
    pose: Any = None
    query_matrix: Optional[np.ndarray] = None

class Pipeline:
    def __init__(self, database: PlaceDatabase, config: Dict[str,Any], method: str='full'):
        if method not in ('classical','local','full','depth3d'):
            raise ValueError('Unknown pipeline method: '+method)
        if config['top_k'] < 1:
            raise ValueError('top_k must be positive')
        self.database,self.config,self.method=database,config,method
        self.local=LocalEngine('sift' if method in ('classical','depth3d') else 'superpoint',
                               config['max_keypoints'],config['sift_ratio'],config['device'],
                               depth_confidence=config.get('lightglue_depth_confidence',.95),
                               width_confidence=config.get('lightglue_width_confidence',.99))
        self.global_engine=None if method in ('classical','local','depth3d') else GlobalEngine(
            config['cosplace_weights'],config['device'],size=config['cosplace_size'])
        if self.global_engine:
            self.descriptors=database.load_descriptors(self.global_engine,config['max_side'])
        self.images=[]
        self.features=[]
        self.depth_maps=[]
        self.reference_matrices=[]
        has_3d_placement=False
        fov = float(config.get('camera_fov_x_deg', 70.0))
        for row in database.rows:
            image,_=read_image(database.resolve(row['reference']),config['max_side'])
            self.images.append(image)
            self.features.append(self.local.extract(image))
            contents = row.get('content', [])
            needs_3d = any(content.get('anchor_3d_camera_m') is not None for content in contents)
            has_3d_placement = has_3d_placement or needs_3d
            depth_path = next((content.get('depth_map') for content in contents
                               if content.get('depth_map')), None)
            depth_map = None
            if depth_path:
                depth_map = np.load(database.resolve(depth_path), allow_pickle=False).astype(np.float32)
                if depth_map.ndim != 2 or not np.isfinite(depth_map).all() or np.any(depth_map <= 0):
                    raise ValueError(f"Invalid saved depth map for '{row['reference']}'")
            if needs_3d and depth_map is None:
                raise ValueError(f"3D overlay for '{row['reference']}' has no saved depth map")
            if method != 'depth3d' and needs_3d:
                raise ValueError("This database has 3D overlay anchors; choose the depth3d scan mode")
            metadata = next((content.get('capture_metadata', {}) for content in contents
                             if content.get('capture_metadata')), {})
            reference_fov = metadata.get('camera_fov_x_deg')
            if (needs_3d and reference_fov is not None
                    and not math.isclose(float(reference_fov), fov, rel_tol=0.0, abs_tol=1e-3)):
                raise ValueError(
                    f"3D reference '{row['reference']}' was captured with a {reference_fov:g}° "
                    f"horizontal FOV; set camera_fov_x_deg to the same calibrated value for scanning")
            matrix = metadata.get('reference_camera_matrix')
            if matrix is not None:
                matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
            elif depth_map is not None:
                matrix = camera_matrix(depth_map.shape, fov)
            else:
                matrix = camera_matrix(image.shape, fov)
            self.depth_maps.append(depth_map)
            self.reference_matrices.append(matrix)
        if method == 'depth3d' and not has_3d_placement:
            raise ValueError('No depth-backed 3D placements are registered; create one in the 3D placement tab')

    def query(self,image: np.ndarray, latitude: Optional[float]=None,
              longitude: Optional[float]=None) -> Prediction:
        if (latitude is None) != (longitude is None):
            raise ValueError('Query location needs both latitude and longitude')
        if latitude is not None and not (math.isfinite(latitude) and math.isfinite(longitude)
                                         and -90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError('Query coordinates are outside valid latitude/longitude ranges')
        start=time.perf_counter()
        feats=self.local.extract(image)
        candidates=np.arange(len(self.features))
        if self.global_engine:
            candidates=rank(self.global_engine.extract(image),self.descriptors)[:self.config['top_k']]
            if latitude is not None:
                nearby=sorted((i for i,row in enumerate(self.database.rows)
                               if row.get('latitude') is not None),
                              key=lambda i:self.database.distance_m(latitude,longitude,
                                  self.database.rows[i]['latitude'],self.database.rows[i]['longitude']))
                candidates=np.asarray(list(dict.fromkeys([*candidates.tolist(),
                                                           *nearby[:self.config['top_k']]])))
        query_matrix = camera_matrix(image.shape, float(self.config.get('camera_fov_x_deg', 70.0)))
        rows=[]
        best=None
        for i in candidates:
            a,b,indices=self.local.match(self.features[i],feats)
            planar=verify(a,b,self.config['ransac_px'],self.config['min_inliers'],
                          self.config['min_inlier_ratio'],self.images[i].shape)
            row=self.database.rows[i]
            needs_3d=any(content.get('anchor_3d_camera_m') is not None
                         for content in row.get('content', []))
            pose=None
            if self.method == 'depth3d' and needs_3d and self.depth_maps[i] is not None:
                raw_h,raw_w=self.depth_maps[i].shape[:2]
                feature_h,feature_w=self.images[i].shape[:2]
                reference_scale=np.asarray([raw_w/feature_w,raw_h/feature_h],np.float32)
                raw_reference_points=a*reference_scale
                pose=estimate_pose(
                    raw_reference_points,b,self.depth_maps[i],self.reference_matrices[i],
                    query_matrix,self.config['min_inliers'],self.config['min_inlier_ratio'],
                    self.config['ransac_px'],float(self.config.get('depth_max_m',80.0)))
                v=Verification(planar.H,pose.inlier_mask,pose.inliers,pose.ratio,pose.accepted)
            else:
                v=planar
            distance=(self.database.distance_m(latitude,longitude,row['latitude'],row['longitude'])
                      if latitude is not None and row.get('latitude') is not None else None)
            rows.append({'index':int(i),'place_id':row['place_id'],'reference':row['reference'],
                         'name':row.get('name'),
                         'latitude':row.get('latitude'),'longitude':row.get('longitude'),
                         'distance_m':distance,
                         'inliers':v.inliers,'ratio':v.ratio,'accepted':v.accepted,
                         'geometry':'pnp' if pose is not None else 'homography'})
            old_distance=(self.database.distance_m(latitude,longitude,
                self.database.rows[best[0]]['latitude'],self.database.rows[best[0]]['longitude'])
                if best is not None and latitude is not None and
                self.database.rows[best[0]].get('latitude') is not None else float('inf'))
            new_distance=distance if distance is not None else float('inf')
            if v.accepted and (best is None or (v.inliers,v.ratio,-new_distance)>
                               (best[1].inliers,best[1].ratio,-old_distance)):
                best=(int(i),v,pose)
        elapsed=(time.perf_counter()-start)*1000
        if best is None:
            return Prediction(None,None,None,rows,elapsed,[])
        row=self.database.rows[best[0]]
        return Prediction(row['place_id'],row['reference'],best[1],rows,elapsed,row['content'],
                          best[2],query_matrix if best[2] is not None else None)

    def render(self,frame: np.ndarray,prediction: Prediction) -> np.ndarray:
        if prediction.place_id is None:
            return frame.copy()
        index=next(i for i,r in enumerate(self.database.rows)
                   if r['place_id']==prediction.place_id and r['reference']==prediction.reference)
        output=frame.copy()
        for content in prediction.content:
            asset=cv2.imread(str(self.database.resolve(content['asset'])),cv2.IMREAD_UNCHANGED)
            if asset is None:
                continue
            anchor_3d=content.get('anchor_3d_camera_m')
            if anchor_3d is not None:
                if prediction.pose is None or prediction.query_matrix is None:
                    continue
                quad=project_points(anchor_3d,prediction.pose.rvec,prediction.pose.tvec,
                                    prediction.query_matrix)
                if quad is not None:
                    output=render_projected_quad(output,asset,quad,content.get('opacity',.8))
                continue
            anchor=np.array(content['anchor_normalized'],float)
            h,w=self.images[index].shape[:2]
            anchor=anchor*np.array([w-1,h-1])
            output=render(output,asset,prediction.verification.H,self.images[index].shape,anchor,
                          content.get('opacity',.8))
        return output
