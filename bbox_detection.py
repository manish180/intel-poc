import numpy as np
import tensorflow as tf
from PIL import Image
import os
import json
import argparse
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
#from imageai.Detection import ObjectDetection

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


class DetectionModel:

    def __init__(self, model_path, label_path):

        self.model_path = model_path
        self.label_path = label_path
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.category_index = label_map_util.create_category_index_from_labelmap(self.label_path,
                                                                                 use_display_name=True)

    def predict_tf(self, image, preprocess=False):
        if preprocess:
            pass
        boxes = self.run_inference_for_single_image(image)
        return boxes

    def run_inference_for_single_image(self, image):
        with self.detection_graph.as_default():
            with tf.Session() as sess:
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # image_sz = tf.sqeeze(tensor_dict['image_dimensions'],[0])
                    # Reframe is required to translate mask from box coordinates to
                    # image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    # real_num_detection = tf.cast(tensor_dict['imsize'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[1], image.shape[2])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.4), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,
                                       feed_dict={image_tensor: image})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

    def visualize_tf(self, image_np, output_dict):
        i = vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            self.category_index,
            max_boxes_to_draw=10,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            min_score_thresh=0.5,
            line_thickness=1)
        return Image.fromarray(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Bounding Box on Images along with Json File")
    parser.add_argument("-m", "--model",
                        help="Path to the trained model", type=str)
    parser.add_argument("-l", "--label_map",
                        help="Path to the label map ", type=str)
    parser.add_argument("-i", "--image",
                        help="Path to the image directory", type=str)
    parser.add_argument("-o", "--outputImage",
                        help="Path for output detection image file", type=str)
    parser.add_argument("-j", "--outputJson",
                        help="Path for output json file", type=str)
    args = parser.parse_args()

    image_paths = []

    for img in os.listdir(args.image):
        image_paths.append(os.path.join(args.image, img))

    for path in image_paths:
        image_name = (path.split("/")[-1]).split(".")[0]
        print(image_name)
        image = Image.open(path)
        image_np = load_image_into_numpy_array(image)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        detector = DetectionModel(args.model, args.label_map)
        output_dict = detector.predict_tf(image_np_expanded)
        img = detector.visualize_tf(image_np, output_dict)
        img.save(os.path.join(args.outputImage, str(image_name) + ".jpg"))
        imsize = img.size
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.uint8).tolist()
        output_dict['detection_boxes'] = output_dict['detection_boxes'].tolist()
        output_dict['detection_scores'] = output_dict['detection_scores'].tolist()
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'].tolist()
        output_dict.update({'image_size': imsize})
        file_name = os.path.join(args.outputJson, str(image_name) + ".json")

        with open(file_name, 'w') as json_file:
            json.dump(output_dict, json_file)
