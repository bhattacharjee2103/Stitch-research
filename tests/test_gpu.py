"""Real-model checks on CUDA: every learned component runs on the GPU with real data.

Skipped automatically where no CUDA device exists (e.g. the Mac), so the CPU
suite is unchanged there. Run on a GPU host with, for example:
    CUDA_VISIBLE_DEVICES=1 STITCH_CONFIG=configs/gpu.yaml .venv-linux/bin/python -m pytest -q
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from stitch import runenv
from stitch.database import PlaceDatabase
from stitch.features import LocalEngine
from stitch.geometry import corner_error, verify
from stitch.io import read_image
from stitch.pipeline import Pipeline
from stitch.retrieval import GlobalEngine, rank

ROOT = Path(__file__).resolve().parents[1]
HP = ROOT / 'data' / 'hpatches-sequences-release'
BENCHMARK = ROOT / 'data' / 'benchmark.json'

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA device'),
    pytest.mark.skipif(not BENCHMARK.exists(), reason='datasets not downloaded (make data)'),
]
DEVICE = 'cuda'


@pytest.fixture(scope='module')
def config():
    result = runenv.load_config()
    result['device'] = DEVICE
    return result


@pytest.fixture(scope='module')
def superpoint(config):
    return LocalEngine('superpoint', config['max_keypoints'], device=DEVICE)


@pytest.fixture(scope='module')
def cosplace(config):
    return GlobalEngine(config['cosplace_weights'], DEVICE, size=config['cosplace_size'])


def test_configured_device_is_cuda():
    """With the GPU config selected, the configured device must be a live CUDA device."""
    configured = runenv.device(runenv.load_config())
    if not configured.startswith('cuda'):
        pytest.skip('STITCH_CONFIG selects CPU; GPU models are still tested below')
    assert torch.cuda.is_available() and torch.zeros(1, device=configured).is_cuda


def test_models_live_on_gpu(superpoint, cosplace):
    for model in (superpoint.extractor, superpoint.matcher, cosplace.model):
        assert all(p.is_cuda for p in model.parameters())


def scaled_truth(sequence, target, max_side):
    _, s1 = read_image(HP / sequence / '1.ppm', max_side)
    _, s2 = read_image(HP / sequence / f'{target}.ppm', max_side)
    H = np.loadtxt(HP / sequence / f'H_1_{target}')
    return s2 @ H @ np.linalg.inv(s1)


@pytest.mark.parametrize('prefix', ['i_', 'v_'])
def test_superpoint_lightglue_gpu_recovers_hpatches_homography(superpoint, config, prefix):
    """SuperPoint+LightGlue on CUDA recover the ground-truth homography on real HPatches pairs."""
    sequence = sorted(p.name for p in HP.iterdir() if p.name.startswith(prefix))[0]
    reference, _ = read_image(HP / sequence / '1.ppm', config['max_side'])
    query, _ = read_image(HP / sequence / '2.ppm', config['max_side'])
    a, b = superpoint.extract(reference), superpoint.extract(query)
    assert a.raw['keypoints'].is_cuda and a.raw['descriptors'].is_cuda
    src, dst, _ = superpoint.match(a, b)
    v = verify(src, dst, config['ransac_px'], config['min_inliers'], config['min_inlier_ratio'],
               reference.shape)
    assert v.accepted, (sequence, v.inliers, v.ratio)
    truth = scaled_truth(sequence, 2, config['max_side'])
    assert corner_error(v.H, truth, reference.shape) < 5.0


def test_superpoint_gpu_agrees_with_cpu(superpoint, config):
    """GPU features match the CPU reference implementation (same model, different device)."""
    image, _ = read_image(HP / sorted(p.name for p in HP.iterdir() if p.is_dir())[0] / '1.ppm',
                          config['max_side'])
    cpu = LocalEngine('superpoint', config['max_keypoints'], device='cpu').extract(image)
    gpu = superpoint.extract(image)
    shared = min(len(cpu.points), len(gpu.points))
    assert abs(len(cpu.points) - len(gpu.points)) <= 0.02 * max(len(cpu.points), 1)
    close = np.linalg.norm(cpu.points[:, None] - gpu.points[None], axis=2).min(1) < 1.0
    assert close.mean() > 0.97 and shared > 100


def test_cosplace_gpu_matches_cpu_descriptor(cosplace, config):
    image, _ = read_image(ROOT / 'data' / 'tokyo' / '00271.jpg', config['max_side'])
    cpu = GlobalEngine(config['cosplace_weights'], 'cpu', size=config['cosplace_size']).extract(image)
    gpu = cosplace.extract(image)
    assert np.isfinite(gpu).all() and float(cpu @ gpu) > 0.9999


@pytest.mark.parametrize('protocol', ['tokyo_day_to_day', 'synthetic_night', 'hpatches_illumination'])
def test_cosplace_gpu_retrieval_on_dataset(cosplace, config, protocol):
    """GPU retrieval over a full benchmark split stays well above chance (recall@10)."""
    proto = next(p for p in json.loads(BENCHMARK.read_text())['protocols'] if p['name'] == protocol)
    index = np.stack([cosplace.extract(read_image(ROOT / g['path'], config['max_side'])[0])
                      for g in proto['gallery']])
    hits = []
    for q in proto['queries']:
        order = rank(cosplace.extract(read_image(ROOT / q['query'], config['max_side'])[0]), index)
        hits.append(any(proto['gallery'][i]['place_id'] == q['place_id'] for i in order[:10]))
    chance = min(1.0, 10 / len({g['place_id'] for g in proto['gallery']}))
    assert np.mean(hits) > max(0.5, chance)


def test_full_pipeline_on_gpu_verifies_tokyo_place(tmp_path, config):
    """Retrieval -> SuperPoint/LightGlue -> RANSAC -> rendering, end to end on CUDA."""
    proto = next(p for p in json.loads(BENCHMARK.read_text())['protocols']
                 if p['name'] == 'tokyo_day_to_day')
    places = [dict(id=g['place_id'], references=[str(ROOT / g['path'])],
                   content=[dict(id='post', kind='image', asset=str(ROOT / 'overlay.png'), opacity=.8,
                                 anchor_normalized=[[.25, .25], [.75, .25], [.75, .6], [.25, .6]])])
              for g in proto['gallery']]
    document = tmp_path / 'places.json'
    document.write_text(json.dumps(dict(schema_version=1, places=places)))
    db = PlaceDatabase(str(document))
    db.build(GlobalEngine(config['cosplace_weights'], DEVICE, size=config['cosplace_size']),
             config['max_side'])
    pipeline = Pipeline(db, config, 'full')
    correct = 0
    for q in proto['queries'][:6]:
        image, _ = read_image(ROOT / q['query'], config['max_side'])
        prediction = pipeline.query(image)
        rendered = pipeline.render(image, prediction)
        assert rendered.shape == image.shape
        correct += prediction.place_id == q['place_id']
    assert correct >= 3
    assert torch.cuda.max_memory_allocated() > 0


def test_depth_model_runs_on_gpu(config):
    pytest.importorskip('transformers')
    from stitch.depth import MonocularDepthEstimator
    estimator = MonocularDepthEstimator(config.get('depth_model_id'), DEVICE)
    assert next(estimator.model.parameters()).is_cuda
    image, _ = read_image(ROOT / 'data' / 'tokyo' / '00271.jpg', 640)
    depth = estimator.estimate(image)
    assert depth.shape == image.shape[:2] and np.isfinite(depth).all()
    assert 0.05 <= depth.min() < depth.max() <= config.get('depth_max_m', 80.0)
