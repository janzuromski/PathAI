import os
import slideflow as sf

def main():

    project = sf.load_project('project')
    hp = sf.ModelParams(
        tile_px=256,
        tile_um='40x',
        model='xception',
        batch_size=32,
        epochs=[3]
    )

    project.train(
        'category',
        params=hp,
        val_k_fold=5,
        filters={
            'dataset': ['train'],
            'category': ['MF', 'BID']
        }
    )

    project.train(
        'category',
        params=hp,
        val_strategy='none',
        filters={
            'dataset': ['train'],
            'category': ['MF', 'BID']
        }
    )


if __name__ == '__main__':
    main()