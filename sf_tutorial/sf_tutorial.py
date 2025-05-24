import pandas as pd
import slideflow as sf

def main():
    P = sf.create_project(
        root='project',
        annotations='../clam_train_annotations.csv',
        slides='/exports/path-cutane-lymfomen-hpc/Thom_Doeleman/CLAM_train_cropped'
    )
    P.extract_tiles(tile_px=256, tile_um='40x', mpp_override=0.25)

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