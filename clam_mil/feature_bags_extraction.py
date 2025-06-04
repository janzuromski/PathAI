import os
import slideflow as sf
from slideflow.mil import mil_config, train_mil

def main():
    project = sf.load_project('project')
    for mag in ('40x', '20x', '10x'):
        dataset = project.dataset(
            tile_px=256, tile_um=mag,
            filters={'dataset': 'train', 'category': ['MF', 'BID']}
        )
        extractor = sf.build_feature_extractor('resnet50_imagenet', tile_px=256)
        project.generate_feature_bags(
            extractor, dataset, outdir=f'project/bags/256px_{mag}'
        )



if __name__ == '__main__':
    main()