# # import paddlehub as hub
# # import os
# # import util

# # module = hub.Module(name="deeplabv3p_xception65_humanseg")


# # def init(img_src):
# #     try:

# #         input_dict = {"image": img_src}

# #         result = module.segmentation(
# #             data=input_dict, use_gpu=False, visualization=True,  output_dir=util.DEST_FOLDER)

# #         out_src = result[0]['save_path']
# #         return {'succ': True, 'path': out_src}
# #     except Exception:
# #         return {'succ': False}


# import transforms
# import models
# import cv2
# import numpy as np
# import argparse
# import os
# import os.path as osp


# # def comp_2d(image_2d, rate):
# #     height, width = image_2d.shape[:2]

# #     mean_array = np.mean(image_2d, axis=1)
# #     mean_array = mean_array[:, np.newaxis]
# #     mean_array = np.tile(mean_array, width)

# #     cov_mat = image_2d.astype(np.float64) - mean_array
# #     eig_val, eig_vec = np.linalg.eigh(np.cov(cov_mat))
# #     p = np.size(eig_vec, axis=1)
# #     idx = np.argsort(eig_val)
# #     idx = idx[::-1]
# #     eig_vec = eig_vec[:, idx]
# #     numpc = rate
# #     if numpc < p or numpc > 0:
# #         eig_vec = eig_vec[:, range(numpc)]
# #     score = np.dot(eig_vec.T, cov_mat)
# #     recon = np.dot(eig_vec, score) + mean_array
# #     recon_img_mat = np.uint8(np.absolute(recon))
# #     return recon_img_mat


# # def subFile(data):
# #     height, width = data.shape[:2]
# #     a_g = data[:, :, 0]
# #     a_b = data[:, :, 1]
# #     a_r = data[:, :, 2]
# #     rate = 60
# #     g_recon, b_recon, r_recon = comp_2d(
# #         a_g, rate), comp_2d(a_b, rate), comp_2d(a_r, rate)

# #     return cv2.merge([g_recon, b_recon, r_recon])


# def predict(img, model, test_transforms):
#     model.arrange_transform(transforms=test_transforms, mode='test')
#     img, im_info = test_transforms(img)
#     img = np.expand_dims(img, axis=0)
#     result = model.exe.run(
#         model.test_prog,
#         feed={'image': img},
#         fetch_list=list(model.test_outputs.values()))
#     score_map = result[1]
#     score_map = np.squeeze(score_map, axis=0)
#     score_map = np.transpose(score_map, (1, 2, 0))
#     return score_map, im_info


# def recover(img, im_info):
#     keys = list(im_info.keys())
#     for k in keys[::-1]:
#         if k == 'shape_before_resize':
#             h, w = im_info[k][0], im_info[k][1]
#             img = cv2.resize(img, (w, h), cv2.INTER_LINEAR)
#         elif k == 'shape_before_padding':
#             h, w = im_info[k][0], im_info[k][1]
#             img = img[0:h, 0:w]
#     return img


# def bg_replace(score_map, img, bg):
#     h, w, _ = img.shape
#     bg = cv2.resize(bg, (w, h))
#     score_map = np.repeat(score_map[:, :, np.newaxis], 3, axis=2)
#     comb = (score_map * img + (1 - score_map) * bg).astype(np.uint8)
#     # comb = (score_map * img).astype(np.uint8)
#     return comb


# def infer(args):
#     try:
#         resize_h = args['image_shape'][1]
#         resize_w = args['image_shape'][0]

#         test_transforms = transforms.Compose(
#             [transforms.Resize((resize_w, resize_h)),
#              transforms.Normalize()])
#         model = models.load_model(args['model_dir'])

#         # print(args)
#         img = cv2.imread(args['image_path'])
#         score_map, im_info = predict(img, model, test_transforms)
#         score_map = score_map[:, :, 1]
#         score_map = recover(score_map, im_info)
#         bg = cv2.imread(args['background_image_path'])
#         # bg = None
#         save_name = osp.basename(args['image_path'])
#         save_path = osp.join(args['save_dir'], save_name)

#         # print(save_path)
#         result = bg_replace(score_map, img, bg)

#         # result = subFile(result)
#         cv2.imwrite(save_path, result)
#         return {'succ': True, 'path': save_path}

#     except Exception as e:
#         print('error: {}'.format(e))
#         return {'succ': False}
import transforms
import models
import cv2
import numpy as np
import argparse
import os
import os.path as osp
import sys
import convert

from models import paddleSeg


# def predict(img, model, test_transforms):
#     model.arrange_transform(transforms=test_transforms, mode='test')
#     img, im_info = test_transforms(img)
#     img = np.expand_dims(img, axis=0)
#     result = model.exe.run(
#         model.test_prog,
#         feed={'image': img},
#         fetch_list=list(model.test_outputs.values()))
#     score_map = result[1]
#     score_map = np.squeeze(score_map, axis=0)
#     score_map = np.transpose(score_map, (1, 2, 0))
#     return score_map, im_info


# def recover(img, im_info):
#     keys = list(im_info.keys())
#     for k in keys[::-1]:
#         if k == 'shape_before_resize':
#             h, w = im_info[k][0], im_info[k][1]
#             img = cv2.resize(img, (w, h), cv2.INTER_LINEAR)
#         elif k == 'shape_before_padding':
#             h, w = im_info[k][0], im_info[k][1]
#             img = img[0:h, 0:w]
#     return img


# def bg_replace(score_map, img, bg):
#     h, w, _ = img.shape
#     bg = cv2.resize(bg, (w, h))
#     score_map = np.repeat(score_map[:, :, np.newaxis], 3, axis=2)
#     comb = (score_map * img + (1 - score_map) * bg).astype(np.uint8)
#     # comb = (score_map * img).astype(np.uint8)
#     return comb
model = paddleSeg.Predictor('/app/myModel/deploy.yaml')


def get_vis_result_name(img_name):
    ext_pos = img_name.rfind(".")
    img_name_fix = img_name[:ext_pos] + "_" + img_name[ext_pos + 1:]
    return img_name_fix + "_result.png"


def infer(args):
    try:

        model.predict([args['image_path']])
        mask_result_name = get_vis_result_name(args['image_path'])

        mm = convert.MatteMatting(
            args['image_path'], mask_result_name)

        result = mm.save_image()
        save_path = mask_result_name
        cv2.imwrite(save_path, result)
        # result.save(save_path)
        return {'succ': True, 'path': save_path}

    except Exception as e:
        print('error: {}'.format(e))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        return {'succ': False}
