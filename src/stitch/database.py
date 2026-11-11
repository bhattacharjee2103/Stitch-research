"""Portable JSON place records plus a versioned NumPy descriptor index."""
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import cv2
from .io import read_image
from .geometry import valid_anchor

class PlaceDatabase:
    def __init__(self, path: str, allow_empty: bool = False):
        self.path=Path(path)
        self.document=json.loads(self.path.read_text())
        if self.document.get('schema_version') != 1:
            raise ValueError('Unsupported database schema')
        self.places=self.document['places']
        if not self.places and not allow_empty:
            raise ValueError('Database needs at least one place')
        ids=[p['id'] for p in self.places]
        if len(ids)!=len(set(ids)):
            raise ValueError('Place IDs must be unique')
        for p in self.places:
            if ('latitude' in p) != ('longitude' in p):
                raise ValueError('Place coordinates need both latitude and longitude')
            if 'latitude' in p:
                lat, lon = float(p['latitude']), float(p['longitude'])
                if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
                    raise ValueError('Place coordinates are outside valid latitude/longitude ranges')
                p['latitude'], p['longitude'] = lat, lon
            if not p.get('references'):
                raise ValueError('Each place needs at least one reference')
            if len(p['references']) != len(set(p['references'])):
                raise ValueError('Reference paths within a place must be unique')
            for content in p.get('content',[]):
                if content.get('reference',p['references'][0]) not in p['references']:
                    raise ValueError('Content reference must belong to its place')
                if 'anchor_normalized' in content:
                    anchor=np.asarray(content['anchor_normalized'],dtype=float)
                    if not valid_anchor(anchor) or np.any(anchor<0) or np.any(anchor>1):
                        raise ValueError('Normalized anchor must be a convex quadrilateral within [0,1]')
                if 'anchor_3d_camera_m' in content:
                    anchor_3d=np.asarray(content['anchor_3d_camera_m'],dtype=float)
                    if (anchor_3d.shape!=(4,3) or not np.isfinite(anchor_3d).all()
                            or np.any(anchor_3d[:,2]<=0) or not content.get('depth_map')):
                        raise ValueError('3D anchors need four finite camera-frame corners and a depth map')
        # A polygon belongs to one photograph; it cannot be reused on another view.
        self.rows=[dict(place_id=p['id'],reference=r,latitude=p.get('latitude'),
                   longitude=p.get('longitude'),name=p.get('name',p['id']),
                   content=[c for c in p.get('content',[])
                   if c.get('reference',p['references'][0]) == r])
                   for p in self.places for r in p['references']]
        self.index_path=self.path.with_suffix('.npz')

    def resolve(self, path: str) -> Path:
        return (self.path.parent/path).resolve()

    def register(self, place_id: str, name: str, reference: str, asset: str,
                 latitude: float, longitude: float, opacity: float = 0.8,
                 anchor: Any = None, capture_metadata: Optional[Dict[str, Any]] = None,
                 depth_map: Optional[np.ndarray] = None,
                 anchor_3d_camera: Optional[Any] = None) -> Dict[str, Any]:
        """Copy a reference and overlay into the database and persist its place record."""
        if not place_id:
            raise ValueError('Place ID must be non-empty')
        latitude, longitude, opacity = float(latitude), float(longitude), float(opacity)
        if not (math.isfinite(latitude) and -90 <= latitude <= 90 and
                math.isfinite(longitude) and -180 <= longitude <= 180):
            raise ValueError('Coordinates must be valid latitude and longitude values')
        if not math.isfinite(opacity) or not 0 <= opacity <= 1:
            raise ValueError('Opacity must be in [0,1]')
        anchor = [[0,0],[1,0],[1,1],[0,1]] if anchor is None else np.asarray(anchor,dtype=float).reshape(4,2).tolist()
        anchor_array=np.asarray(anchor,dtype=float)
        if not valid_anchor(anchor_array) or np.any(anchor_array<0) or np.any(anchor_array>1):
            raise ValueError('Anchor must be a convex quadrilateral of normalized coordinates in [0,1]')
        ref_src, asset_src = Path(reference), Path(asset)
        if not ref_src.is_file() or not asset_src.is_file():
            raise FileNotFoundError('Reference image and overlay must both exist')
        reference_image = cv2.imread(str(ref_src), cv2.IMREAD_COLOR)
        if reference_image is None:
            raise ValueError('Reference image cannot be decoded')
        overlay = cv2.imread(str(asset_src), cv2.IMREAD_UNCHANGED)
        if overlay is None or min(overlay.shape[:2]) < 2:
            raise ValueError('Overlay image cannot be decoded')
        if depth_map is not None:
            depth_map = np.asarray(depth_map, dtype=np.float32)
            if depth_map.shape != reference_image.shape[:2]:
                raise ValueError('Depth map dimensions must match the reference image')
            if not np.isfinite(depth_map).all() or np.any(depth_map <= 0):
                raise ValueError('Depth map must contain finite positive metre values')
        if anchor_3d_camera is not None:
            anchor_3d_camera = np.asarray(anchor_3d_camera, dtype=np.float64).reshape(4, 3)
            if (not np.isfinite(anchor_3d_camera).all()
                    or np.any(anchor_3d_camera[:, 2] <= 0)):
                raise ValueError('3D overlay corners must be finite and in front of the reference camera')
            if depth_map is None:
                raise ValueError('A depth map is required for a 3D overlay anchor')
        slug = ''.join(c if c.isalnum() or c in '-_' else '_' for c in place_id).strip('_')
        if not slug:
            raise ValueError('Place ID must contain a filename-safe character')
        place = next((p for p in self.places if p['id'] == place_id), None)
        if place is not None:
            for existing_ref in place['references']:
                stored = self.resolve(existing_ref)
                if stored.is_file() and hashlib.sha256(stored.read_bytes()).digest() == hashlib.sha256(ref_src.read_bytes()).digest():
                    raise ValueError('This reference image is already registered for the place')
            if place.get('latitude') is not None and self.distance_m(
                    latitude, longitude, float(place['latitude']), float(place['longitude'])) > 100:
                raise ValueError('Additional reference coordinates must be within 100 m of the place')
            reference_number = len(place['references']) + 1
        else:
            reference_number = 1
        ref_rel = Path('assets/references') / slug / f'{reference_number}_{ref_src.name}'
        asset_rel = Path('assets/content') / slug / f'{reference_number}_{asset_src.name}'
        ref_dst, asset_dst = self.path.parent / ref_rel, self.path.parent / asset_rel
        depth_rel = ref_rel.with_suffix(ref_rel.suffix + '.depth.npy') if depth_map is not None else None
        depth_dst = self.path.parent / depth_rel if depth_rel is not None else None
        if ref_dst.exists() or asset_dst.exists() or (depth_dst is not None and depth_dst.exists()):
            raise FileExistsError('Registered asset path already exists')
        ref_dst.parent.mkdir(parents=True, exist_ok=True)
        asset_dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(ref_src, ref_dst)
            shutil.copy2(asset_src, asset_dst)
            content = dict(id=f'{slug}-overlay-{reference_number}', reference=ref_rel.as_posix(),
                           asset=asset_rel.as_posix(), anchor_normalized=anchor, opacity=opacity)
            if depth_map is not None:
                np.save(depth_dst, depth_map)
                content['depth_map'] = depth_rel.as_posix()
            if anchor_3d_camera is not None:
                content['anchor_3d_camera_m'] = anchor_3d_camera.tolist()
            if capture_metadata:
                content['capture_metadata'] = capture_metadata
            if place is None:
                place = dict(id=place_id, name=name or place_id, latitude=latitude,
                             longitude=longitude, references=[], content=[])
                self.places.append(place)
            elif place.get('latitude') is None:
                place['latitude'], place['longitude'] = latitude, longitude
            place['references'].append(ref_rel.as_posix())
            place.setdefault('content', []).append(content)
            self.document['places'] = self.places
            temp = self.path.with_suffix(self.path.suffix + '.tmp')
            temp.write_text(json.dumps(self.document, indent=2) + '\n')
            temp.replace(self.path)
            self.rows.append(dict(place_id=place_id, reference=ref_rel.as_posix(),
                                  latitude=latitude, longitude=longitude, name=name or place_id,
                                  content=[content]))
            return place
        except Exception:
            ref_dst.unlink(missing_ok=True)
            asset_dst.unlink(missing_ok=True)
            if depth_dst is not None:
                depth_dst.unlink(missing_ok=True)
            raise

    @staticmethod
    def distance_m(latitude: float, longitude: float, other_latitude: float,
                   other_longitude: float) -> float:
        """Haversine distance; location supports candidate discovery, image matching verifies."""
        rad = math.pi / 180
        lat1, lat2 = latitude * rad, other_latitude * rad
        dlat, dlon = (other_latitude-latitude)*rad, (other_longitude-longitude)*rad
        a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        return 6371000 * 2 * math.atan2(math.sqrt(a), math.sqrt(max(0, 1-a)))

    def build(self, engine: Any, max_side: int = 640) -> None:
        descriptors=[]
        fingerprints=[]
        for row in self.rows:
            path=self.resolve(row['reference'])
            fingerprints.append(hashlib.sha256(path.read_bytes()).hexdigest())
            descriptors.append(engine.extract(read_image(path,max_side)[0]))
        np.savez_compressed(self.index_path,descriptors=np.stack(descriptors),
                            fingerprints=np.asarray(fingerprints),
                            references=np.asarray([r['reference'] for r in self.rows]),
                            schema_version=np.array(2),
                            config=np.asarray(json.dumps(self.descriptor_config(engine,max_side))))

    @staticmethod
    def descriptor_config(engine: Any, max_side: int) -> Dict[str,Any]:
        return dict(size=engine.size,enhance=engine.enhance,max_side=max_side,
                    encoder=type(engine).__name__,weights_sha256=getattr(engine,'weights_sha256',None),
                    preprocessing_version=1)

    def load_descriptors(self, engine: Any, max_side: int = 640) -> np.ndarray:
        with np.load(self.index_path,allow_pickle=False) as index:
            if int(index['schema_version']) != 2:
                raise ValueError('Descriptor index version changed; rebuild database')
            if list(index['references']) != [r['reference'] for r in self.rows]:
                raise ValueError('Reference list changed; rebuild database')
            if len(index['fingerprints']) != len(self.rows):
                raise ValueError('Incomplete reference fingerprints; rebuild database')
            for row,expected in zip(self.rows,index['fingerprints']):
                if hashlib.sha256(self.resolve(row['reference']).read_bytes()).hexdigest()!=expected:
                    raise ValueError('Reference image changed; rebuild database')
            config=json.loads(str(index['config']))
            if config!=self.descriptor_config(engine,max_side):
                raise ValueError('Descriptor configuration changed; rebuild database')
            descriptors=index['descriptors']
            if descriptors.ndim!=2 or len(descriptors)!=len(self.rows) or not np.isfinite(descriptors).all():
                raise ValueError('Invalid descriptor matrix; rebuild database')
            return descriptors.copy()
