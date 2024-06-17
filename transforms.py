import cv2
import albumentations as A


#class HorizontalFlipKeepIndexOrder(A.HorizontalFlip):
#    """Flip the input horizontally but keypoint the same keypoints indices."""
#
#    def apply_to_keypoints(self, keypoints, **params):
#        flipped_keypoints = [self.apply_to_keypoint(tuple(keypoint[:4]), **params) + tuple(keypoint[4:]) for keypoint in keypoints]
#        flipped_keypoints.reverse()
#        return flipped_keypoints


def load_transform() -> A.core.composition.Compose:

    return A.Compose([
        A.Sequential([
            A.RandomBrightnessContrast(brightness_limit=1, contrast_limit=1),
            A.VerticalFlip(), # Random change of brightness & contrast
        ], p=1)
    ],
    keypoint_params=A.KeypointParams(format='xy'), # More about keypoint formats used in albumentations library read at https://albumentations.ai/docs/getting_started/keypoints_augmentation/
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bboxes_labels']) # Bboxes should have labels, read more at https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
    )


if __name__ == '__main__':

    input_size = 224
    augmentation_type = "2chkeep"
    input_transform = None
    input_transform = load_transform()#augmentation_type=augmentation_type, augmentation_probability=1.0, input_size=input_size)

