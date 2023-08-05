

import os
def create_folder_structure(base_folder):
    if not os.path.exists(base_folder):
        os.mkdir(base_folder)

    subfolders = ['train', 'test', 'val']

    for folder in subfolders:
        folder_path = os.path.join(base_folder,folder)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            os.mkdir(os.path.join(folder_path,images))
            os.mkdir(os.path.join(folder_path,labels))


import albumentations as alb 

import albumentations as alb
augmentor=alb.Compose([alb.RandomCrop(width=300, height= 300),
                       alb.HorizontalFlip(p = 0.5),
                       alb.RandomBrightnessContrast( p = 0.2 ),
                       alb.RandomGamma( p = 0.2 ),
                       alb.RGBShift( p = 0.2 ),
                       alb.VerticalFlip( p = 0.5 ),
                       ],
                      bbox_params=alb.BboxParams( format = 'albumentations',
                                                 label_fields=['class_labels']))
#bbox_params specifies parameters for handling bounding boxes in the augmentation pipeline



for partition in ['train', 'test', 'val']:
    for image in os.listdir(os.path.join(aug_data, partition, 'images')):
        img = cv2.imread(os.path.join(aug_data, partition, 'images', image))

        coords = [0, 0, 0.00001, 0.00001]
        label_path = os.path.join(aug_data, partition, 'labels', f'{image.split(".")[0]}.json')

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = json.load(f)

            coords[0] = label['shapes'][0]['points'][0][0]
            coords[1] = label['shapes'][0]['points'][0][1]
            coords[2] = label['shapes'][0]['points'][1][0]
            coords[3] = label['shapes'][0]['points'][1][1]
            w = img.shape[1]
            h = img.shape[0]
            coords = list(np.divide(coords, [w, h, w, h]))

        try:
            for x in range(100):
                augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
                cv2.imwrite(os.path.join(aug_data, 'aug_data', partition, 'images',
                                         f'{image.split(".")[0]}.{x}.jpg'),
                            augmented['image'])
                annotation = {}
                annotation['image'] = image
                if os.path.exists(label_path):
                    if len(augmented['bboxes']) == 0:
                        annotation['bbox'] = [0, 0, 0, 0]
                        annotation['class'] = 0
                        #image has in it no faces
                    else:
                        annotation['bbox'] = augmented['bboxes'][0]
                        annotation['class'] = 1

                else:
                    annotation['bbox'] = [0, 0, 0, 0]
                    annotation['class'] = 0

                with open(os.path.join(aug_data, 'aug_data', partition, 'labels',
                                       f'{image.split(".")[0]}.{x}.json'), 'w') as f:
                    json.dump(annotation, f)

        except Exception as e:
            print(e)
