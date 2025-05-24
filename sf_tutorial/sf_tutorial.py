import pandas as pd
import slideflow as sf

def main():
    df = pd.read_csv('../annotations_clam_new.csv')

    train_dataset = sf.Dataset(
        slides='/exports/path-cutane-lymfomen-hpc/Thom_Doeleman/CLAM_train_cropped',
        tfrecords='project/tfrecords',
        annotations=df,
        tile_px=256,
        tile_um='40x',
    )
    # val_dataset = sf.Dataset(
    #     slides='/exports/path-cutane-lymfomen-hpc/Thom_Doeleman/CLAM_validate_cropped',
    #     tfrecords='project/tfrecords',
    #     annotations=df,
    #     tile_px=256,
    #     tile_um='10x',
    # )
    P = sf.create_project(
        root='project',
        tf
    )
    train_dataset.extract_tiles(override_mpp=0.25)
    # val_dataset.extract_tiles(override_mpp=1)

    print(f'Extracted {train_dataset.num_tiles} tiles.')
    labels, _ = train_dataset.labels('category')

    train_dataset = train_dataset.balance('category')

    hp = sf.ModelParams(
        tile_px=256, tile_um=128, model='xception', batch_size=32, epochs=[3]
    )

    trainer = sf.model.build_trainer(
        hp=hp, outdir='./project/out/', labels=labels
    )

    # results = trainer.train(train_dataset, val_dataset)

if __name__ == '__main__':
    main()