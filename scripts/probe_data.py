"""Record reachability of official manual-download sources without huge downloads."""
import json
from pathlib import Path
import subprocess

URLS={
 'aachen_v1_1':'https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/aachen_v1_1.zip',
 'robotcar_images':'https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/RobotCar-Seasons/images/',
 'cmu_slice2':'https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Extended-CMU-Seasons/slice2.tar',
 'msls_download':'https://www.mapillary.com/dataset/places',
 'tokyo_streetview_info':'https://www.di.ens.fr/willow/research/netvlad/'
}

def main():
    rows=[]
    for name,url in URLS.items():
        command=['curl','-sSIL','--connect-timeout','10','--max-time','25',url]
        process=subprocess.run(command,capture_output=True,text=True)
        rows.append(dict(name=name,url=url,argv=command,returncode=process.returncode,
                         headers=process.stdout,error=process.stderr,
                         image_archives_downloaded=False))
        print(name,'curl exit',process.returncode,flush=True)
    Path('results/dataset_probes.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':main()
