import os
import pandas as pd
import slideflow as sf

def main():
    project_dir = 'project'
    extraction_params = (
        (256, mag) for mag in ('40x', '20x', '10x')
    )
    if os.path.exists(project_dir):
        project = sf.load_project(project_dir)
    else:
        project = sf.create_project(
            root=project_dir,
            annotations='/exports/path-cutane-lymfomen-hpc/jan/rl/clam_train_annotations.csv',
            slides='/exports/path-cutane-lymfomen-hpc/Thom_Doeleman/CLAM_train_cropped'
        )
    for tile_px, tile_um in extraction_params:
        dataset = project.dataset(tile_px=tile_px, tile_um=tile_um)
        dataset.extract_tiles(mpp_override=0.25, skip_extracted=True)
    

if __name__ == '__main__':
    main()