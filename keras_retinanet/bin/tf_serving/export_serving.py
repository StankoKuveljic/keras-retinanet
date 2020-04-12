""" A CLI for the RetinaNet detector export. """
import argparse
import logging
import sys

from keras import Model
from keras.layers import Input, Lambda
import tensorflow as tf

from keras_retinanet import models
from keras_retinanet.utils.config import parse_anchor_parameters, \
    read_config_file
from typing import Optional, NamedTuple, Dict
from keras import backend as K
import os
K.set_learning_phase(False)

log = logging.getLogger(__name__)


class AnchorParametersWrap:

    def __init__(self, anchors: Optional = None) -> None:
        self._anchors = anchors

    @property
    def anchors(self) -> Optional:
        return self._anchors

    @property
    def num_anchors(self) -> Optional[int]:
        if self.anchors:
            return self.anchors.num_anchors()
        return None

    @classmethod
    def from_conf(
            cls, conf_path: Optional[str] = None) -> 'AnchorParametersWrap':
        ret: Optional['AnchorParametersWrap'] = None
        if conf_path:
            config = read_config_file(conf_path)
            anchors = parse_anchor_parameters(config)
            ret = cls(anchors)
            log.info(f'loaded anchors from {conf_path}')
        else:
            log.info(f'using default anchors')
            ret = cls()

        return ret


def image_preprocess(input_tensor):
    """ Applies preprocessing on an input string image. """
    ret = tf.io.decode_image(input_tensor, channels=3)
    ret = tf.cast(ret, tf.float32)
    ret -= [103.939, 116.779, 123.68]  # TODO: extract magic numbers
    return ret


def batch_preprocessing(input_tensor):
    """ Applies preprocessing function on an input batch. """
    ret = tf.map_fn(
        image_preprocess,
        input_tensor,
        dtype=tf.float32)
    ret.set_shape([None, None, None, 3])
    return ret


def string_input_model() -> Model:
    """ Creates a model that serves as input for retinanet. """
    model_input = Input(
        shape=[], batch_shape=None, dtype=tf.string, name='input_image')
    preprocess = Lambda(
        batch_preprocessing,
        name='preprocess_image')(model_input)
    model = Model(model_input, preprocess, name='model_input')
    return model


def merge_models(retina_model: Model, input_model: Model) -> Model:
    """ Merges converted model with input string model. """
    retina_outputs = retina_model(input_model.outputs)
    model = Model(input_model.inputs, retina_outputs, name='retina_serving')
    return model


def load_retina(
        weights: str,
        num_classes: int,
        anchors_wrap: AnchorParametersWrap,
        backbone_name: str) -> Model:
    """ Loads retinanet with weights. """
    ret = models.backbone(backbone_name=backbone_name).retinanet(
        num_classes=num_classes, num_anchors=anchors_wrap.num_anchors)
    ret.load_weights(weights)
    ret = models.convert_model(
        model=ret,
        nms=True,
        class_specific_filter=True,
        anchor_params=anchors_wrap.anchors)
    return ret


class ServingSignature(NamedTuple):
    """ Models a metadata for TF serving export. """
    signature_name: str = 'predict'
    input_name: str = 'image_string'
    boxes_output: str = 'boxes'
    scores_output: str = 'scores'
    labels_output: str = 'labels'


class RetinaServingExporter:
    """ Models a retinanet tf serving exporter. """

    def __init__(
            self, model: Model, signature: ServingSignature) -> None:
        self.model = model
        self.signature = signature

    def serving_input(self) -> Dict:
        """ Returns serving input metadata. """
        return {
            self.signature.input_name: self.model.input
        }

    def serving_output(self) -> Dict:
        """ Returns serving output metadata. """
        return {
            self.signature.boxes_output: self.model.output[0],
            self.signature.scores_output: self.model.output[1],
            self.signature.labels_output: self.model.output[2],
        }

    def signature_name(self) -> str:
        """ Default signature name. """
        return self.signature.signature_name

    def export(self, output_dir: str, version: int) -> None:
        """ Exports retinanet model. """
        TFServingExporter(self).export(output_dir, version)


class TFServingExporter:
    """ Models a TF serving simple exporter. """

    def __init__(self, model_serving: RetinaServingExporter) -> None:
        self.model_serving = model_serving

    def export(self, output_dir: str, version: int) -> None:
        """ Exports a versioned model to a directory. """
        export_path = os.path.join(output_dir, str(version))
        if os.path.exists(export_path):
            raise Exception(f'version of model already exists {export_path}')
        with K.get_session() as sess:
            tf.saved_model.simple_save(
                sess,
                export_path,
                self.model_serving.serving_input(),
                self.model_serving.serving_output()
            )
        log.info(f'model exported to {export_path}')


def export_retinanet(args: argparse.Namespace) -> None:
    """ Loads, converts and export retinanet as tf serving model. """
    weights_path = args.weights
    version = args.version
    num_classes = args.num_classes
    export_path = args.output_path
    backbone_name = args.backbone_name
    anchors_file = args.anchors

    anchors = AnchorParametersWrap.from_conf(anchors_file)
    input_model = string_input_model()
    retina_model = load_retina(
        weights_path, num_classes, anchors, backbone_name)
    final_model = merge_models(retina_model, input_model)
    log.info(f'merged retinanet and input model')
    final_model.summary()

    retina_export = RetinaServingExporter(final_model, ServingSignature())
    retina_export.export(export_path, version)


def main() -> int:
    parser = argparse.ArgumentParser(description='RetinaNet argument parser')
    subparsers = parser.add_subparsers(dest='cmd')

    export_model_parser = subparsers.add_parser('export-model')
    export_model_parser.set_defaults(func=export_retinanet)
    export_model_parser.add_argument(
        '--weights',
        type=str,
        dest='weights',
        help='path to weights h5 file.')
    export_model_parser.add_argument(
        '--version',
        type=int,
        dest='version',
        help='model version')
    export_model_parser.add_argument(
        '--classes',
        dest='num_classes',
        type=int,
        help='Number of classes')
    export_model_parser.add_argument(
        '--output',
        dest='output_path',
        type=str,
        help='Output path')
    export_model_parser.add_argument(
        '-b', '--backbone',
        dest='backbone_name',
        type=str,
        default='resnet50',
        help='Resnet backbone')
    export_model_parser.add_argument(
        '-a', '--anchors',
        dest='anchors',
        type=str,
        help='Path to anchors ini file, if not given will use default anchors'
    )

    args = parser.parse_args()
    if args.cmd is None:
        parser.print_help()
        return 1

    try:
        args.func(args)
    except Exception as ex:
        log.exception(f'unable to complete request: {ex}')
        return 2
    except KeyboardInterrupt:
        log.info('request cancelled by user')
        return 2

    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.StreamHandler(sys.stdout)])
    main()
