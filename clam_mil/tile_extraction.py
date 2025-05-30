import os
import pandas as pd
import slideflow as sf

def main():
    project_dir = 'project'
    extraction_params = (
        (256, mag) for mag in ('10x', '20x', '40x')
    )
    if os.path.exists(project_dir):
        project = sf.load_project(project_dir)
    else:
        project = sf.create_project(
            root=project_dir,
            annotations='/exports/path-cutane-lymfomen-hpc/jan/rl/annotations_clam_full.csv',
            slides='/exports/path-cutane-lymfomen-hpc/Thom_Doeleman/CLAM_train_cropped'
        )
        project.add_source(
            name='validate',
            slides='/exports/path-cutane-lymfomen-hpc/Thom_Doeleman/CLAM_validate_cropped'
        )
    for tile_px, tile_um in extraction_params:
        dataset = project.dataset(tile_px=tile_px, tile_um=tile_um)
        dataset.extract_tiles(
            mpp_override=0.25,
            skip_extracted=True,
            report=False
        )
    

if __name__ == '__main__':
    main()