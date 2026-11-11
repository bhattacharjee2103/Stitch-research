"""Execute ONNX export, float parity, QDQ INT8 quantization and retrieval validation."""
import json
from pathlib import Path
import time
import traceback
import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType,quantize_static
import torch
from stitch.io import read_image
from stitch import runenv
from stitch.retrieval import GlobalEngine,image_tensor

ROOT=Path(__file__).resolve().parents[1]

def main():
    config=runenv.load_config();device=runenv.device(config);out=runenv.results_dir()
    torch.set_num_threads(config['threads']);cv2.setNumThreads(config['threads'])
    engine=GlobalEngine(config['cosplace_weights'],'cpu')
    base=runenv.artifact_dir('models/exports')
    providers=(['CUDAExecutionProvider','CPUExecutionProvider'] if device.startswith('cuda') and
               'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider'])
    fp=base/'cosplace_fp32.onnx';qp=base/'cosplace_int8.onnx'
    record={'scope':'CosPlace global descriptor only; SuperPoint/LightGlue and full pipeline not exported',
            'input_shape':[1,3,224,224],'runtime':ort.__version__,'device':providers[0]}
    torch.onnx.export(engine.model,torch.zeros(1,3,224,224),str(fp),opset_version=17,
                      input_names=['image'],output_names=['descriptor'],dynamo=False)
    onnx.checker.check_model(onnx.load(str(fp)));record['onnx_export']='success'
    manifest=json.loads((ROOT/'data/benchmark.json').read_text())
    train=json.loads((out/'training.json').read_text())['train_places']
    class Reader(CalibrationDataReader):
        def __init__(self):
            self.paths=iter([ROOT/f'data/hpatches-sequences-release/{s}/{n}.ppm' for s in train for n in [1,2]])
        def get_next(self):
            path=next(self.paths,None)
            return None if path is None else {'image':image_tensor(read_image(path)[0]).numpy()}
    try:
        quantize_static(str(fp),str(qp),Reader(),quant_format=QuantFormat.QDQ,
                         activation_type=QuantType.QUInt8,weight_type=QuantType.QInt8,
                         op_types_to_quantize=['Conv','Gemm','MatMul'],per_channel=False)
        onnx.checker.check_model(onnx.load(str(qp)));record['int8_quantization']='success'
    except Exception as e:
        record['int8_quantization']='failed';record['quantization_error']=repr(e)
        traceback.print_exc()
    opts=ort.SessionOptions();opts.intra_op_num_threads=config['threads'];opts.inter_op_num_threads=1
    models=[('fp32',fp)]+([('int8',qp)] if record['int8_quantization']=='success' else [])
    protocols=[p for p in manifest['protocols'] if p['name'].startswith('tokyo_')]
    paths=sorted({g['path'] for p in protocols for g in p['gallery']}|{q['query'] for p in protocols for q in p['queries']})
    tensors={p:image_tensor(read_image(ROOT/p)[0]).numpy() for p in paths}
    native={p:engine.model(torch.from_numpy(x)).detach().numpy()[0] for p,x in tensors.items()}
    record['models']=[]
    for name,path in models:
        try:
            session=ort.InferenceSession(str(path),sess_options=opts,providers=providers)
            used=session.get_providers()[0]
            for _ in range(3):session.run(None,{'image':next(iter(tensors.values()))})
            values={};latencies=[]
            for p,x in tensors.items():
                start=time.perf_counter();v=session.run(None,{'image':x})[0][0]
                latencies.append((time.perf_counter()-start)*1000);values[p]=v
            parity=[float(np.max(np.abs(values[p]-native[p]))) for p in paths]
            recalls=[]
            for protocol in protocols:
                index=np.stack([values[g['path']] for g in protocol['gallery']]);hits=[]
                for q in protocol['queries']:
                    order=np.argsort(-(index@values[q['query']]),kind='stable')
                    hits.append([any(protocol['gallery'][i]['place_id']==q['place_id'] for i in order[:k]) for k in [1,5,10]])
                r=np.asarray(hits).mean(0)
                recalls.append(dict(protocol=protocol['name'],n=len(hits),recall1=float(r[0]),recall5=float(r[1]),recall10=float(r[2])))
            record['models'].append(dict(variant=name,path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,
                median_inference_ms=float(np.median(latencies)),p95_inference_ms=float(np.percentile(latencies,95)),
                provider=used,max_abs_parity_error=max(parity),mean_abs_max_error=float(np.mean(parity)),retrieval=recalls,status='success'))
        except Exception as e:
            record['models'].append(dict(variant=name,bytes=path.stat().st_size,status='failed',error=repr(e)))
            traceback.print_exc()
    (out/'export.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))

if __name__=='__main__':main()
