import argparse
import gc
import os
import random
import sys
import tracemalloc

import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from main_network import Classifier
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter(log_dir='logs')
from tqdm import tqdm
from transformers import ViTModel, ViTImageProcessor
from configuration import Config
import torch
from unixcoder import UniXcoder
import torch.optim as optim
import torch.nn as nn
import numpy as np
import logging
from typing import List, Tuple, Dict
from PIL import Image
import torchvision.transforms as transforms
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
def calculate_recall(tp, fn):

    if tp + fn == 0:
        return 0
    else:
        return tp / (tp + fn)


def calculate_precision(tp, fp):

    if tp + fp == 0:
        return 0
    else:
        return tp / (tp + fp)


def calculate_f1_score(tp, fp, fn):

    recall = calculate_recall(tp, fn)
    precision = calculate_precision(tp, fp)
    if recall + precision == 0:
        return 0
    else:
        return 2 * (precision * recall) / (precision + recall)


def evaluate(labels, outputs, epoch, args):
    theshold = args.set_threshold
    best_data = {"true_positive": 0, "true_negative": 0, "false_positive": 0, "false_negative": 0}
    correct = 0
    if theshold is None:
        theshold = 0.5
        best_threshold = 0.5
        while theshold < 1:
            current_crt = 0
            true_positive = 0
            true_negative = 0
            false_positive = 0
            false_negative = 0
            for label, output in zip(labels, outputs):
                predicted_label = output > theshold
                if predicted_label == label:
                    current_crt += 1
                    if label == 1:
                        true_positive += 1
                    else:
                        true_negative += 1
                else:
                    if label == 1:
                        false_negative += 1
                    else:
                        false_positive += 1

            if current_crt > correct:
                correct = current_crt
                best_threshold = theshold
                best_data["true_negative"] = true_negative
                best_data["true_positive"] = true_positive
                best_data["false_negative"] = false_negative
                best_data["false_positive"] = false_positive
            theshold += 0.05
            break
    else:
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        for label, output in zip(labels, outputs):
            predicted_label = output > float(theshold)
            if predicted_label == label:
                correct += 1
                if label == 1:
                    true_positive += 1
                else:
                    true_negative += 1
            else:
                if label == 1:
                    false_negative += 1
                else:
                    false_positive += 1
            best_threshold = theshold
            best_data["true_negative"] = true_negative
            best_data["true_positive"] = true_positive
            best_data["false_negative"] = false_negative
            best_data["false_positive"] = false_positive

    precision = calculate_precision(best_data["true_positive"], best_data["false_positive"])
    recall = calculate_recall(best_data["true_positive"], best_data["false_negative"])
    f1_score = calculate_f1_score(best_data["true_positive"], best_data["false_positive"], best_data["false_negative"])
    logger.info(f"f*** epoch {epoch} evaluation ***\n"
        f"threshold: {best_threshold}, accuracy: {correct / len(labels)}\n"
        f"precision: {precision}\n"
        f"recall: {recall}\n"
        f"f1-score: {f1_score}"
    )
    output_dir = os.path.join(args.output_dir, "results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    train_info_name = os.path.join(output_dir, f"train_info_{epoch}.txt")
    true_positive = best_data["true_positive"]
    true_negative = best_data["true_negative"]
    false_positive = best_data["false_positive"]
    false_negative = best_data["false_negative"]

    with open(train_info_name, "w", encoding="utf-8") as f:
        info = f"epoch: {epoch}\n" \
               f"Test Accuracy: {correct/len(labels) * 100:.2f}%\n" \
               f"True positive: {true_positive}\n" \
               f"True negative: {true_negative}\n" \
               f"False negative: {false_negative}\n" \
               f"False positive: {false_positive}\n" \
               f"Best threshold: {best_threshold}"
        f.write(info)

    return precision, recall, f1_score


def plot_alpha(writer, mlp, epoch, tag_str):
    """
    plot weight of 'alpha' in tensorboard
    Args:
        writer:  tensorboard writer
        mlp:     the NN model
        epoch:   epoch
        tag_str: the name of the plot window
    """
    for name, param in mlp.named_parameters():
        # 简单写法：想画什么就大概记录下这一层的名字‘block_alpha’，想画权重就是‘weight’，偏移就是'bias'
        if 'weight' in name and param.grad is not None:
            writer.add_histogram(tag=name + tag_str, values=param.grad.clone().cpu().numpy(), global_step=epoch)

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code1_input_tokens,
                 code1_input_ids,
                 code2_input_tokens,
                 code2_input_ids,
                 code1_image_tensor,
                 code2_image_tensor,
                 label

    ):
        self.code1_input_tokens = code1_input_tokens
        self.code1_input_ids = code1_input_ids
        self.code2_input_tokens = code2_input_tokens
        self.code2_input_ids = code2_input_ids
        self.code1_image_tensor = code1_image_tensor
        self.code2_image_tensor = code2_image_tensor
        self.label = label

def convert_examples_to_features(data: List,
                                 tokenizer,
                                 args,
                                 code_file_path: str,
                                 img_file_path: str,
                                 code_filename_extension=".java",
                                 image_filename_extension=".png"
) -> InputFeatures:

    code1_path = os.path.join(code_file_path, data[0]+code_filename_extension)
    img1_path = os.path.join(img_file_path, data[0]+image_filename_extension)
    code2_path = os.path.join(code_file_path, data[1] + code_filename_extension)
    img2_path = os.path.join(img_file_path, data[1] + image_filename_extension)
    with open(code1_path, "r", encoding="utf-8") as f:
        code1_text = f.read()
    with open(code2_path, "r", encoding="utf-8") as f:
        code2_text = f.read()
    code1_input_ids = torch.tensor(tokenizer([code1_text], max_length=512, mode="<encoder-only>"))
    code2_input_ids = torch.tensor(tokenizer([code2_text], max_length=512, mode="<encoder-only>"))

    transform = transforms.Compose([transforms.ToTensor()])
    image1 = Image.open(os.path.join(img1_path)).convert("L")
    code1_image_tensor = transform(image1)
    image2 = Image.open(os.path.join(img2_path)).convert("L")
    code2_image_tensor = transform(image2)

    return InputFeatures(code1_text, code1_input_ids, code2_text, code2_input_ids, code1_image_tensor, code2_image_tensor, data[2])

class CloneDataset(Dataset):
    def __init__(self, tokenizer, args, code_file_path, img_file_path, dataset_file_path):
        self.examples = []
        data = []
        with open(dataset_file_path) as f:
            for line in f:  # The context of line: code1 code2 label
                line = line.strip().split(" ")
                data.append(line)

        for d in tqdm(data, desc="Tokenize", total=len(data)):
            self.examples.append(convert_examples_to_features(d, tokenizer, args, code_file_path, img_file_path))
        if 'train' in dataset_file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("label: {}".format(example.label))
                logger.info("code1_input_tokens: {}".format([x.replace('\u0120', '_') for x in example.code1_input_tokens]))
                logger.info("code1_input_ids: {}".format(' '.join(map(str, example.code1_input_ids))))
                logger.info("code1_input_tokens: {}".format([x.replace('\u0120', '_') for x in example.code2_input_tokens]))
                logger.info("code1_input_ids: {}".format(' '.join(map(str, example.code2_input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        label = self.examples[i].label
        code1_input_ids = self.examples[i].code1_input_ids
        code1_image_tensor = self.examples[i].code1_image_tensor
        code2_input_ids = self.examples[i].code2_input_ids
        code2_image_tensor = self.examples[i].code2_image_tensor

        return (
            (code1_input_ids, code1_image_tensor), (code2_input_ids, code2_image_tensor),
            torch.tensor(int(label)))
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
def main():
    sys.argv = [
        "run.py",
        "--output_dir=codenet/codenet_save_models_mcdm",
        "--vit_model_name_or_path=google/vit-base-patch16-224",
        "--vit_unfrozen_layer=3",
        "--unixcoder_model_name_or_path=microsoft/unixcoder-base",
        "--unixcoder_unfrozen_layer=3",
        "--do_train",
        "--train_data_file=codenet/codenet_pairs/train_pairs.txt",
        "--eval_data_file=codenet/codenet_pairs/test_pairs.txt",
        "--test_data_file=codenet/codenet_pairs/test_pairs.txt",
        "--code_file_path=codenet/codenet_code_all",
        "--image_file_path=codenet/codenet_image_all",
        "--epoch=100",
        "--learning_rate=1e-4",
        "--seed=42",
        "--evaluate_during_training",

    ]
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--test_data_file", default=None, type=str, required=False,
                        help="The input test data file.")
    parser.add_argument("--eval_data_file", default=None, type=str, required=True,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--code_file_path", default=None, type=str, required=True,
                        help="The input code data file.")
    parser.add_argument("--image_file_path", default=None, type=str, required=True,
                        help="The input image data file.")
    parser.add_argument("--train_data_file", default=None, type=str, required=False,
                        help="The input training data file.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters

    parser.add_argument("--unixcoder_model_name_or_path", default=None, type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--unixcoder_model_path", default=None, type=str,
                        help="The model you have been pretrained before.")
    parser.add_argument("--unixcoder_unfrozen_layer", default=3, type=int,
                        help="The layer of UniXcoder can be trained. (From 0 to 12)")
    parser.add_argument("--vit_model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--vit_model_path", default=None, type=str,
                        help="The model you have been pretrained before.")
    parser.add_argument("--vit_unfrozen_layer", default=3, type=int,
                        help="The layer of ViT can be trained. (From 0 to 12)")
    parser.add_argument("--detection_model_path", default=None, type=str,
                        help="The path of model which has been trained before.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument('--set_threshold', type=float, default=None,
                        help="Set the detection threshold.")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--epoch", default=1, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--start_epoch", default=0, type=float,
                        help="The start epoch.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()
    set_seed(args.seed)
    """ vit model frozen"""
    if args.vit_model_path is None:
        vit_model_name = args.vit_model_name_or_path if args.vit_model_name_or_path is not None else "google/vit-base-patch16-224"
        vit_model = ViTModel.from_pretrained(vit_model_name, ignore_mismatched_sizes=True)
    else:
        vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224", ignore_mismatched_sizes=True)
        model_state_dict = vit_model.state_dict()

        # 新的 state_dict 只包含可以匹配的层
        new_state_dict = {}
        pretrained_state_dict = torch.load(args.vit_model_path)
        # 手动过滤匹配的层
        for layer_name, pretrained_param in pretrained_state_dict.items():
            if layer_name in model_state_dict:
                model_param = model_state_dict[layer_name]
                # 检查形状是否匹配
                if pretrained_param.shape == model_param.shape:
                    new_state_dict[layer_name] = pretrained_param
                else:
                    print(f"Skipping {layer_name} due to shape mismatch: {pretrained_param.shape} vs {model_param.shape}")
            else:
                print(f"Skipping {layer_name} as it's not found in the model.")

        # 加载可以匹配的参数
        model_state_dict.update(new_state_dict)
    # vit_model.load_state_dict(model_state_dict)
    vit_feature_extractor = ViTImageProcessor.from_pretrained(args.vit_model_name_or_path if args.vit_model_name_or_path is not None else "google/vit-base-patch16-224", do_resize=False)
    layer_count = 0
    for name, param in vit_model.named_parameters():
        param.requires_grad = False
        layer_count += 1
    counter = 0
    for name, param in vit_model.named_parameters():
        last_unfreeze_layers = 12 - args.vit_unfrozen_layer
        for x in range(last_unfreeze_layers, 12):
            if str(x) in name:
                param.requires_grad = True
        counter += 1
        if layer_count - counter <= 4:
            param.requires_grad = True
    """ unixcoder """
    if args.unixcoder_model_path is None:
        unixcoder_model_name = args.unixcoder_model_name_or_path if args.unixcoder_model_name_or_path is not None else "microsoft/unixcoder-base"
        unixcoder = UniXcoder(unixcoder_model_name)
    else:
        unixcoder = UniXcoder("microsoft/unixcoder-base")
        unixcoder.load_state_dict(torch.load(args.unixcoder_model_path))
    for name, param in unixcoder.named_parameters():
        param.requires_grad = False

    for name, param in unixcoder.named_parameters():
        last_unfreeze_layers = 12 - args.unixcoder_unfrozen_layer
        if "pooler" in name:
            param.requires_grad = True
        for x in range(last_unfreeze_layers, 12):
            if str(x) in name:
                param.requires_grad = True
    """ dataloader """
    code_file_path = args.code_file_path
    image_file_path = args.image_file_path
    train_dataset_path = args.train_data_file
    eval_dataset_path = args.eval_data_file
    config = Config()
    classifier = Classifier(config, vit_model, vit_feature_extractor, unixcoder).to(device)
    if args.detection_model_path is not None:
        classifier.load_state_dict(torch.load(args.detection_model_path))
    # 定义一个fliter，只传入requires_grad=True的模型参数
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, classifier.parameters()), lr=args.learning_rate)
    best_f1 = -1.0
    count = 0
    previous_f1 = -1.0
    """ train """
    if args.do_train:
        num_epoch = args.epoch
        train_dataset = CloneDataset(unixcoder.tokenize, args, code_file_path, image_file_path, train_dataset_path)
        eval_dataset = []
        if args.evaluate_during_training:
            eval_dataset = CloneDataset(unixcoder.tokenize, args, code_file_path, image_file_path, eval_dataset_path)
        for epoch in range(args.start_epoch, num_epoch):
            rea_epoch = epoch
            classifier.train()
            # loss_sum = torch.tensor([0.0]).to(device)
            train_loss = 0
            tr_num = 0
            with tqdm(total=len(train_dataset)) as tbar:
                for code1, code2, label in train_dataset:
                    output = classifier(code1, code2, label, is_train=True)
                    loss = output[1]
                    tr_num += 1
                    tr_loss = loss.item()
                    train_loss += tr_loss
                    avg_loss = round(train_loss/tr_num, 5)
                    # loss_sum += loss.squeeze(0)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    tbar.update(1)
                    tbar.set_description(f"epoch {epoch} loss {avg_loss} train")
                    # if idx != 0 and (idx+1) % 256 == 0:
                    #     writer.add_scalar('Loss/train', (loss_sum.item())/256, (rea_epoch * len(X_train)// 256) + (idx+1)/256)
                    #     del loss_sum
                    #     loss_sum = torch.tensor([0.0]).to(device)
            # eval
            if args.evaluate_during_training:
                classifier.eval()
                with torch.no_grad():
                    outputs = []
                    labels = []
                    for code1, code2, label in tqdm(eval_dataset, desc="eval", total=len(eval_dataset)):
                        output = classifier(code1, code2, label)
                        outputs.append(output)
                        labels.append(label.int())
                    precision, recall, f1_score = evaluate(labels, outputs, rea_epoch, args)
            model_save_path = os.path.join(args.output_dir, "models")
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            model_file_name = os.path.join(model_save_path, f"clone_detection_model_{rea_epoch}.pkl")
            torch.save(classifier.state_dict(), model_file_name)
            if f1_score >= best_f1 and f1_score >= previous_f1:
                count = 0
                torch.save(classifier.state_dict(), os.path.join(model_save_path, "clone_detection_model_best"))
                with open(os.path.join(model_save_path, "best_info.txt"), "w") as f:
                    f.write(str(rea_epoch))
                best_f1 = f1_score
            else:
                count += 1
            if count >= 3:
                logger.info("Early stopping triggered.")
                break
            previous_f1 = f1_score

    if args.do_test:
        test_dataset = CloneDataset(unixcoder.tokenize, args, code_file_path, image_file_path, args.test_data_file)
        classifier.eval()
        with torch.no_grad():
            outputs = []
            labels = []
            for code1, code2, label in tqdm(test_dataset, desc="test", total=len(test_dataset)):
                output = classifier(code1, code2, label)
                outputs.append(output)
                labels.append(label.int())

if __name__ == "__main__":
    main()