#!/usr/bin/env python3

import json

def get_image_id(name, images):
    for image in images:
        if image['file_name'] == f'{name}.jpg':
            return image['id']

    raise ValueError(name)

if __name__ == '__main__':
    
    with open('predictions_fixed.json', 'r') as in_f:
        data = json.load(in_f)

    for ann in data['annotations']:
        ann['image_id'] = get_image_id(ann['image_id'], data['images'])

    with open('predictions_fixed.json', 'w') as out_f:
        json.dump(data, out_f)
