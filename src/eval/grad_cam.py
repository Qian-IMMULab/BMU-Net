import os
import cv2
import numpy as np
from monai.visualize import GradCAM


class BMUGradCAM(GradCAM):
    def _upsample_and_post_process(self, acti_map, img):
        if acti_map.size()[2] == acti_map.size()[3]:
            img_spatial = img[2].shape[2:]
        else:
            img_spatial = img[0].shape[2:]
        acti_map = self.upsampler(img_spatial)(acti_map)
        return self.postprocessing(acti_map)


def show_cam_on_image(
        img: np.ndarray,
        mask: np.ndarray,
        use_rgb: bool = True,
        colormap: int = cv2.COLORMAP_JET,
        image_weight: float = 0.5,
) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    img = np.float32(img) / 255
    if np.max(img) > 1:
        raise Exception("The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}"
        )

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def save_image(patient_name, view, image_name, label, pred, values, ori_img):
    file_path = "./heatmap/{}_{}_{}_{}".format(patient_name, view, label, pred)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_masked = "{}/{}_masked.jpg".format(
        file_path, image_name
    )
    file_ori = "{}/{}_ori.jpg".format(
        file_path, image_name
    )
    cv2.imwrite(file_masked, values)
    cv2.imwrite(file_ori, ori_img)


def crop_image(data_image):
    height, width = data_image.shape[:2]
    crop_width = int(width / 3)
    cropped_image = data_image[:, :-crop_width, :]
    return cropped_image


def show_grad_cam(config, data, path, view, model, label, pred):
    target_layers = ['model1.backbone.7.1.conv2', 'model2.backbone.7.1.conv2',
                     'model3.backbone.7.1.conv2', 'model4.backbone.7.1.conv2']
    masks = []
    image_name = ['mg1', 'mg2', 'us3', 'us4', 'us5', 'us6', 'us7', 'us8']
    for target_layer in target_layers:
        cam = BMUGradCAM(nn_module=model, target_layers=target_layer)
        mask = cam(data)
        masks.append(mask[0])
        masks.append(mask[1])
    patient_name = path[0][0].split('/')[6]
    for j, (a, b, c, d) in enumerate(zip(data, path, masks, image_name)):

        img = cv2.imread(b[0])
        if j in [0, 1]:
            img = cv2.resize(img, (config.bmu.mg_image_w, config.bmu.mg_image_h))
        else:
            img = cv2.resize(img, (config.bmu.us_image_size, config.bmu.us_image_size))
        show_values = show_cam_on_image(
            img, c.permute(1, 2, 0).cpu().numpy()
        )
        save_image(patient_name, view[0], d, label, pred, show_values, img)
