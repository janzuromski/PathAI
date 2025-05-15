import sys
sys.path.append('/exports/path-cutane-lymfomen-hpc/jan/gymhisto')

from gymhisto import HistoEnv
import numpy as np
import os
from PIL import Image

def main():
    imgs = [
        'LUMC_H04_01233_1A.tiff', 
        'LUMC_H14_15391_1A.tiff', 
        'LUMC_H15_00896_1A.tiff'
    ]
    img_dir ='/exports/path-cutane-lymfomen-hpc/Thom_Doeleman/CLAM_train_cropped' 
    xml_path = '/exports/path-cutane-lymfomen-hpc/jan/gymhisto/example_annot.xml'
    tile_size = 256
    result_path = './results'
    
    for img in imgs:
        img_path = os.path.join(img_dir, img)
        env = HistoEnv(img_path, xml_path, tile_size, result_path) 
        obs = env.reset()
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print('action=', action, 'info=', info, 'reward=', reward, 'done=', done)
        env.render(mode="human")
        state = env.state
        image = Image.fromarray(state.transpose((1, 2, 0)))
        image.save(f'data/{img}', format="TIFF")

if __name__ == '__main__':
    main()
