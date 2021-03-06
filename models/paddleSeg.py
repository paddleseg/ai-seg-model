import os
import sys
import ast
import time

import yaml
import cv2

import numpy as np
import paddle.fluid as fluid

from concurrent.futures import ThreadPoolExecutor, as_completed


def get_images_from_dir(img_dir, support_ext=".jpg|.jpeg"):
    if (not os.path.exists(img_dir) or not os.path.isdir(img_dir)):
        raise Exception("Image Directory [%s] invalid" % img_dir)
    imgs = []
    for item in os.listdir(img_dir):
        ext = os.path.splitext(item)[1][1:].strip().lower()
        if (len(ext) > 0 and ext in support_ext):
            item_path = os.path.join(img_dir, item)
            imgs.append(item_path)
    return imgs


def generate_colormap(num_classes):
    color_map = num_classes * [0, 0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3+1] |= (((lab >> 0) & 1) << (7 - j))
            if ((lab >> 0) & 1) << (7 - j) > 0:
                color_map[i * 3+1] = 1

            color_map[i * 3 + 2] |= (((lab >> 1) & 1) << (7 - j))

            color_map[i * 3 + 3] |= (((lab >> 2) & 1) << (7 - j))

            if color_map[i * 3] == 0 and color_map[i * 3 + 1] == 0 and color_map[i * 3 + 2] == 0:
                color_map[i * 3 + 4] = 0
            else:
                color_map[i * 3 + 4] = 255

            j += 1
            lab >>= 3

    color_map = [color_map[i:i + 4] for i in range(0, len(color_map), 4)]
    return color_map


class DeployConfig:
    def __init__(self, conf_file):
        if not os.path.exists(conf_file):
            raise Exception('Config file path [%s] invalid!' % conf_file)

        with open(conf_file) as fp:
            configs = yaml.load(fp, Loader=yaml.FullLoader)
            deploy_conf = configs["DEPLOY"]
            # 1. get eval_crop_size
            self.eval_crop_size = ast.literal_eval(
                deploy_conf["EVAL_CROP_SIZE"])
            # 2. get mean
            self.mean = deploy_conf["MEAN"]
            # 3. get std
            self.std = deploy_conf["STD"]
            # 4. get class_num
            self.class_num = deploy_conf["NUM_CLASSES"]
            # 5. get paddle model and params file path
            self.model_file = os.path.join(deploy_conf["MODEL_PATH"],
                                           deploy_conf["MODEL_FILENAME"])
            self.param_file = os.path.join(deploy_conf["MODEL_PATH"],
                                           deploy_conf["PARAMS_FILENAME"])
            # 6. use_gpu
            self.use_gpu = deploy_conf["USE_GPU"]
            # 7. predictor_mode
            self.predictor_mode = deploy_conf["PREDICTOR_MODE"]
            # 8. batch_size
            self.batch_size = deploy_conf["BATCH_SIZE"]
            # 9. channels
            self.channels = deploy_conf["CHANNELS"]
            # 10. use_pr
            self.use_pr = deploy_conf["USE_PR"]


class ImageReader:
    def __init__(self, configs):
        self.config = configs
        self.threads_pool = ThreadPoolExecutor(configs.batch_size)

    # image processing thread worker
    def process_worker(self, imgs, idx, use_pr=False):
        image_path = imgs[idx]
        im = cv2.imread(image_path, -1)
        if len(im.shape) == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        channels = im.shape[2]
        if channels != 3 and channels != 4:
            print("Only support rgb(gray) or rgba image.")
            return -1
        ori_h = im.shape[0]
        ori_w = im.shape[1]

        # resize to eval_crop_size
        eval_crop_size = self.config.eval_crop_size
        if (ori_h != eval_crop_size[1] or ori_w != eval_crop_size[0]):
            im = cv2.resize(
                im, eval_crop_size, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)

        # if use models with no pre-processing/post-processing op optimizations
        if not use_pr:
            im_mean = np.array(self.config.mean).reshape((3, 1, 1))
            im_std = np.array(self.config.std).reshape((3, 1, 1))
            # HWC -> CHW, don't use transpose((2, 0, 1))
            im = im.swapaxes(1, 2)
            im = im.swapaxes(0, 1)
            im = im[:, :, :].astype('float32') / 255.0
            im -= im_mean
            im /= im_std
        im = im[np.newaxis, :, :, :]
        info = [image_path, im, (ori_w, ori_h)]
        return info

    def process(self, imgs, use_pr=False):
        # print('--------', imgs)
        imgs_data = []
        with ThreadPoolExecutor(max_workers=self.config.batch_size) as exe_pool:
            tasks = [
                exe_pool.submit(self.process_worker, imgs, idx, use_pr)
                for idx in range(len(imgs))
            ]
        for task in as_completed(tasks):
            imgs_data.append(task.result())
        return imgs_data


class Predictor:
    def __init__(self, conf_file):
        self.config = DeployConfig(conf_file)
        self.image_reader = ImageReader(self.config)
        if self.config.predictor_mode == "NATIVE":
            predictor_config = fluid.core.NativeConfig()
            predictor_config.prog_file = self.config.model_file
            predictor_config.param_file = self.config.param_file
            predictor_config.use_gpu = self.config.use_gpu
            predictor_config.device = 0
            predictor_config.fraction_of_gpu_memory = 0
        elif self.config.predictor_mode == "ANALYSIS":
            predictor_config = fluid.core.AnalysisConfig(
                self.config.model_file, self.config.param_file)
            if self.config.use_gpu:
                predictor_config.enable_use_gpu(100, 0)
                predictor_config.switch_ir_optim(True)
                if gflags.FLAGS.trt_mode != "":
                    precision_type = trt_precision_map[gflags.FLAGS.trt_mode]
                    use_calib = (gflags.FLAGS.trt_mode == "int8")
                    predictor_config.enable_tensorrt_engine(
                        workspace_size=1 << 30,
                        max_batch_size=self.config.batch_size,
                        min_subgraph_size=40,
                        precision_mode=precision_type,
                        use_static=False,
                        use_calib_mode=use_calib)
            else:
                predictor_config.disable_gpu()
            predictor_config.switch_specify_input_names(True)
            predictor_config.enable_memory_optim()
        self.predictor = fluid.core.create_paddle_predictor(predictor_config)

    def create_tensor(self, inputs, batch_size, use_pr=False):
        im_tensor = fluid.core.PaddleTensor()
        im_tensor.name = "image"
        if not use_pr:
            im_tensor.shape = [
                batch_size, self.config.channels, self.config.eval_crop_size[1],
                self.config.eval_crop_size[0]
            ]
        else:
            im_tensor.shape = [
                batch_size, self.config.eval_crop_size[1],
                self.config.eval_crop_size[0], self.config.channels
            ]
        im_tensor.dtype = fluid.core.PaddleDType.FLOAT32
        im_tensor.data = fluid.core.PaddleBuf(inputs.ravel().astype("float32"))
        return [im_tensor]

    # save prediction results and visualization them
    def output_result(self, imgs_data, infer_out, use_pr=False):
        for idx in range(len(imgs_data)):
            img_name = imgs_data[idx][0]
            ori_shape = imgs_data[idx][2]
            mask = infer_out[idx]
            if not use_pr:
                mask = np.argmax(mask, axis=0)
            mask = mask.astype('uint8')
            mask_png = mask
            score_png = mask_png[:, :, np.newaxis]
            score_png = np.concatenate([score_png] * 4, axis=2)

            # visualization score png
            color_map = generate_colormap(self.config.class_num)
            for i in range(score_png.shape[0]):
                for j in range(score_png.shape[1]):
                    score_png[i, j] = color_map[score_png[i, j, 0]]

            # save the mask
            # mask of xxx.jpeg will be saved as xxx_jpeg_mask.png
            ext_pos = img_name.rfind(".")
            img_name_fix = img_name[: ext_pos] + "_" + img_name[ext_pos + 1:]
            mask_save_name = img_name_fix + "_mask.png"
            cv2.imwrite(mask_save_name, mask_png, [cv2.CV_8UC1])
            # save the visualized result
            # result of xxx.jpeg will be saved as xxx_jpeg_result.png
            vis_result_name = img_name_fix + "_result.png"
            result_png = score_png
            # if not use_pr:
            result_png = cv2.resize(
                result_png,
                ori_shape,
                fx=0,
                fy=0,
                interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(vis_result_name, result_png, [cv2.CV_8UC4])
            print("save result of [" + img_name + "] done.")

    def predict(self, images):
        # image reader preprocessing time cost
        reader_time = 0
        # inference time cost
        infer_time = 0
        # post_processing: generate mask and visualize it
        post_time = 0
        # total time cost: preprocessing + inference + postprocessing
        total_runtime = 0

        # record starting time point
        total_start = time.time()
        batch_size = self.config.batch_size
        use_pr = self.config.use_pr
        for i in range(0, len(images), batch_size):
            real_batch_size = batch_size
            if i + batch_size >= len(images):
                real_batch_size = len(images) - i
            reader_start = time.time()
            img_datas = self.image_reader.process(images[i:i + real_batch_size],
                                                  use_pr)
            input_data = np.concatenate([item[1] for item in img_datas])
            input_data = self.create_tensor(
                input_data, real_batch_size, use_pr=use_pr)
            reader_end = time.time()
            infer_start = time.time()
            output_data = self.predictor.run(input_data)[0]
            infer_end = time.time()
            output_data = output_data.as_ndarray()
            post_start = time.time()
            self.output_result(img_datas, output_data, use_pr)
            post_end = time.time()
            reader_time += (reader_end - reader_start)
            infer_time += (infer_end - infer_start)
            post_time += (post_end - post_start)

        # finishing process all images
        total_end = time.time()
        # compute whole processing time
        total_runtime = (total_end - total_start)
        print(
            "images_num=[%d],preprocessing_time=[%f],infer_time=[%f],postprocessing_time=[%f],total_runtime=[%f]"
            % (len(images), reader_time, infer_time, post_time, total_runtime))
