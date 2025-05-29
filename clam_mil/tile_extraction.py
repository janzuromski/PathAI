import os
import pandas as pd
import slideflow as sf

def main():
    if os.path.exists('project'):
        project = sf.load_project('project')
    else:
        project = sf.create_project(
            root='project',
            annotations='../clam_train_annotations.csv',
            slides='/exports/path-cutane-lymfomen-hpc/Thom_Doeleman/CLAM_train_cropped'
        )
    dataset = project.dataset(tile_px=256, tile_um='40x')
    dataset.extract_tiles(mpp_override=0.25, skip_extracted=True)

    # labels, _ = train_dataset.labels('category')

    # train_dataset = train_dataset.balance('category')

    # hp = sf.ModelParams(
    #     tile_px=256, tile_um=128, model='xception', batch_size=32, epochs=[3]
    # )

    # trainer = sf.model.build_trainer(
    #     hp=hp, outdir='./project/out/', labels=labels
    # )

    # results = trainer.train(train_dataset, val_dataset)

if __name__ == '__main__':
    main()