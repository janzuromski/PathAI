import os
import slideflow as sf
from slideflow.mil import mil_config, train_mil

def main():

    project = sf.load_project('project')
    dataset = project.dataset(
        tile_px=256, 
        tile_um='40x',
        filters={'dataset': 'train', 'category': ['MF', 'BID']}
    )
    train, val = dataset.split(labels='category', val_fraction=0.2)
    config = mil_config(
        model='attention_mil',
        lr=1e-4,
        batch_size=32,
        epochs=10,
        fit_one_cycle=True
    )
    train_mil(
        config, 
        train_dataset=train, 
        val_dataset=val,
        outcomes='category', 
        outdir='project/mil_outcomes', 
        bags='project/bags'
    )


if __name__ == '__main__':
    main()